#include "AnomalyEngine.h"
#include "AsyncLogger.h"
#include <xmmintrin.h> // For _MM_SET_FLUSH_ZERO_MODE
#include <pmmintrin.h> // For _MM_SET_DENORMALS_ZERO_MODE
#include <algorithm>
#include <cstring>
#include <chrono>
#include <string>
#include <thread>
#include <unordered_map>
#include "OrtsessionConfig.h"
#include <array>
#include <stdexcept>

namespace {
    // Minimal RFC 4648 base64 decoder. Used to unpack render_lut_rgb_b64
    // (768 bytes = 256x3 uint8 RGB) from the ONNX metadata, so the runtime
    // depends on no extra base64 library.
    std::vector<uint8_t> Base64Decode(const std::string& in) {
        static constexpr char kPad = '=';
        std::array<int, 256> T{};
        T.fill(-1);
        const char* alphabet =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; ++i) T[static_cast<unsigned char>(alphabet[i])] = i;

        std::vector<uint8_t> out;
        out.reserve(in.size() * 3 / 4);
        int val = 0, bits = -8;
        for (unsigned char c : in) {
            if (c == kPad) break;
            if (T[c] == -1) continue; // skip whitespace/newlines
            val = (val << 6) + T[c];
            bits += 6;
            if (bits >= 0) {
                out.push_back(static_cast<uint8_t>((val >> bits) & 0xFF));
                bits -= 8;
            }
        }
        return out;
    }
}

AnomalyEngine::AnomalyEngine()
{
    // Initialize ONNX Environment (Warning level, log id)
    env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "AnomalyEngineEnv");
    memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
}

void AnomalyEngine::Initialize(const std::wstring& modelPath, const RT::CpuPartition& cpuPartition)
{
    Ort::SessionOptions sessionOptions;

    // Execution providers + session options: identical policy across all
    // engines. The CPU partition sizes and pins this session's intra-op pool.
    ConfigureOrtSessionOptions(sessionOptions, "Anomaly", cpuPartition);

    // Create the session
    session = std::make_unique<Ort::Session>(*env, modelPath.c_str(), sessionOptions);
    Log::Info("ONNX Session created successfully for Anomaly Detection.");

    // Validate the export contract and read all calibration metadata (thresholds,
    // display range, colormap LUT) from the ONNX file. Rejects non-3.0 or
    // unverified models before any inference is attempted.
    LoadContractMetadata();

    try {
        // Initialize the allocator with default options for node name retrieval
        Ort::AllocatorWithDefaultOptions allocator;

        // Retrieve and store the input node name
        inputName = session->GetInputNameAllocated(0, allocator).get();

        // Retrieve all output node names and store them in the persistent vector
        outputNames.clear();
        for (size_t idx = 0; idx < session->GetOutputCount(); ++idx) {
            outputNames.emplace_back(session->GetOutputNameAllocated(idx, allocator).get());
        }

        // Build the C-style pointer array once to ensure valid memory addresses for the inference call
        outputNamesC.clear();
        outputNamesC.reserve(outputNames.size());
        for (const auto& n : outputNames) {
            outputNamesC.push_back(n.c_str());
        }

        // Extract model input shape and dimensions from TypeInfo
        Ort::TypeInfo typeInfo = session->GetInputTypeInfo(0);
        inputShape = typeInfo.GetTensorTypeAndShapeInfo().GetShape();

        // Handle dynamic batch size by defaulting to 1
        if (inputShape[0] == -1) {
            inputShape[0] = 1;
        }

        modelChannels = inputShape[1];
        modelHeight = inputShape[2];
        modelWidth = inputShape[3];

        // Pre-allocate the input buffer based on the calculated tensor size
        size_t totalElements = 1;
        for (int64_t d : inputShape) {
            totalElements *= d;
        }
        inputTensorValues.assign(totalElements, 0.0f);
        m_splitPlanes.resize(modelChannels);

        // Create the input tensor wrapper ONCE using the pre-allocated buffer;
        // Infer refills the buffer in place and reuses this same Ort::Value
        m_inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo,
            inputTensorValues.data(),
            inputTensorValues.size(),
            inputShape.data(),
            inputShape.size()
        );

        // Define names for the warmup execution
        const char* inputNames[] = { inputName.c_str() };
        const char* warmupOutputNames[] = { outputNames[0].c_str() };

        // Execute Warmup Runs.
        // The first run triggers engine building (TensorRT compilation) and
        // memory allocation; the following ones stabilize the execution path so
        // the first REAL frame does not hit a cold engine. This happens at
        // Configure time, before the camera trigger starts.
        constexpr int kWarmupRuns = 50;
        double firstMs = 0.0;
        double lastMs = 0.0;
        for (int r = 0; r < kWarmupRuns; ++r) {
            auto startTime = std::chrono::steady_clock::now();
            session->Run(Ort::RunOptions{ nullptr }, inputNames, &m_inputTensor, 1, warmupOutputNames, 1);
            lastMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - startTime).count();
            if (r == 0) firstMs = lastMs;
        }

        Log::Info("Warmup completed successfully: {} runs, first {:.1f} ms, last {:.1f} ms. Engine is locked and ready for real-time inference.",
            kWarmupRuns, firstMs, lastMs);
    }
    catch (const Ort::Exception& e) {
        Log::Error("### ERROR DURING WARMUP: {}", e.what());
        throw; // Escalate the error, as an un-warmed engine cannot guarantee production timings
    }
}

void AnomalyEngine::LoadContractMetadata()
{
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::ModelMetadata metadata = session->GetModelMetadata();

    auto lookupString = [&](const char* key) -> std::string {
        auto value = metadata.LookupCustomMetadataMapAllocated(key, allocator);
        return value ? std::string(value.get()) : std::string();
        };
    auto lookupFloat = [&](const char* key, float& target) -> bool {
        auto value = metadata.LookupCustomMetadataMapAllocated(key, allocator);
        if (value) {
            target = std::stof(value.get());
            Log::Info("Metadata '{}' = {}", key, target);
            return true;
        }
        return false;
        };

    // --- Contract gate: refuse anything that is not a verified 3.0 export ---
    // The error message names the tool that fixes it, so an operator who sees
    // it in the log knows exactly what to run.
    exportContract = lookupString("export_contract");
    const std::string verified = lookupString("verified");

    if (exportContract != "3.0") {
        throw std::runtime_error(
            "Anomaly model rejected: export_contract='" +
            (exportContract.empty() ? std::string("<missing>") : exportContract) +
            "', expected '3.0'. Re-export the model with the current export_onnx.py.");
    }
    if (verified != "true") {
        throw std::runtime_error(
            "Anomaly model rejected: metadata 'verified' != 'true' (the model has "
            "not been calibrated). Run calibrate_threshold.py on this .onnx to "
            "write calibrated_threshold / calibration_map_min / calibration_map_max "
            "/ render_lut_rgb_b64 and set verified=true.");
    }

    // normalization=in_graph -> the graph already applies ImageNet normalization
    // (InGraphNormalize). The host must feed RGB in [0,1] and must NOT normalize
    // again, otherwise the score collapses (CONTRACT.md §1.4, double norm).
    const std::string normalization = lookupString("normalization");
    normalizationInGraph = (normalization == "in_graph");
    if (!normalizationInGraph) {
        // Contract 3.0 is always in_graph; if we ever see something else the
        // preprocessing assumption is wrong, so make it loud rather than silent.
        Log::Warning("### WARNING: normalization='{}' (expected 'in_graph' for contract 3.0). "
            "Host will apply ImageNet normalization as a fallback.", normalization);
    }

    // --- Calibration values (no defaults silently accepted for a 3.0 model) ---
    const bool hasMin = lookupFloat("calibration_map_min", mapMin);
    const bool hasMax = lookupFloat("calibration_map_max", mapMax);
    const bool hasThreshold = lookupFloat("calibrated_threshold", scoreThreshold);
    hasPixelThreshold = lookupFloat("calibrated_threshold_pixel", pixelThreshold);

    if (!hasMin || !hasMax || !hasThreshold) {
        throw std::runtime_error(
            "Anomaly model rejected: verified=true but calibration metadata is "
            "incomplete (calibration_map_min/calibration_map_max/calibrated_threshold). "
            "Re-run calibrate_threshold.py.");
    }

    // Guard against a degenerate display range that would divide by zero
    if (mapMax - mapMin <= 0.0f) {
        throw std::runtime_error(
            "Invalid calibration metadata: calibration_map_max must be greater than "
            "calibration_map_min.");
    }

    // --- Shared colormap LUT (render_lut_rgb_b64 -> 256x1 CV_8UC3, RGB) ---
    const std::string lutB64 = lookupString("render_lut_rgb_b64");
    if (lutB64.empty()) {
        throw std::runtime_error(
            "Anomaly model rejected: 'render_lut_rgb_b64' missing. Re-run "
            "calibrate_threshold.py to embed the shared jet colormap.");
    }
    const std::vector<uint8_t> lutBytes = Base64Decode(lutB64);
    if (lutBytes.size() != 256 * 3) {
        throw std::runtime_error(
            "Invalid render_lut_rgb_b64: decoded " + std::to_string(lutBytes.size()) +
            " bytes, expected 768 (256x3 uint8 RGB).");
    }
    // Store as 256x1 CV_8UC3 so it can drive cv::applyColorMap directly. The
    // channel order of the output equals the byte order of the LUT (RGB).
    m_lutRgb.create(256, 1, CV_8UC3);
    std::memcpy(m_lutRgb.data, lutBytes.data(), lutBytes.size());

    Log::Info("Contract 3.0 verified. mapMin={}, mapMax={}, scoreThreshold={}, "
        "pixelThreshold={} (present={}), LUT loaded, normalizationInGraph={}.",
        mapMin, mapMax, scoreThreshold, pixelThreshold, hasPixelThreshold, normalizationInGraph);
}

void AnomalyEngine::Infer(const void* pInputImage, uint32_t width, uint32_t height,
    uint32_t channels, void* pOutputHeatmap, float& outAnomalyScore, std::string& outStatus)
{
    // Wrap raw input buffer into an OpenCV Mat for processing
    cv::Mat rawImg(height, width, CV_8UC3, const_cast<void*>(pInputImage));

    // Resize input to match model's expected dimensions
    cv::resize(rawImg, m_resizeImage,
        cv::Size(static_cast<int>(modelWidth), static_cast<int>(modelHeight)),
        0, 0, cv::INTER_LINEAR);

    // ImageNet statistics, used ONLY in the non-in_graph fallback path
    const float mean[] = { 0.485f, 0.456f, 0.406f };
    const float stddev[] = { 0.229f, 0.224f, 0.225f };
    const size_t planeSize = static_cast<size_t>(modelHeight) * static_cast<size_t>(modelWidth);

    // Convert HWC uint8 -> planar float32 CHW. Contract 3.0 (normalization=in_graph):
    // feed RGB in [0,1] and let the graph's InGraphNormalize apply ImageNet norm.
    // Normalizing on the host as well would double the normalization and collapse
    // the score (CONTRACT.md §1.4). The input buffer is assumed RGB ([D-Q4]),
    // so split plane 0 == R, matching the graph's expected channel order.
    cv::split(m_resizeImage, m_splitPlanes);
    for (int c = 0; c < modelChannels; ++c) {
        // in_graph:      dst = pixel / 255
        // host fallback: dst = pixel * scale - offset  (ImageNet)
        const float scale = normalizationInGraph ? (1.0f / 255.0f)
                                                 : (1.0f / (255.0f * stddev[c]));
        const float offset = normalizationInGraph ? 0.0f : (mean[c] / stddev[c]);

        // Map tensor memory to cv::Mat for efficient pixel-wise operation
        cv::Mat planeF32(static_cast<int>(modelHeight), static_cast<int>(modelWidth), CV_32FC1,
            inputTensorValues.data() + static_cast<size_t>(c) * planeSize);

        m_splitPlanes[c].convertTo(planeF32, CV_32FC1, scale, -offset);
    }

    // Zero-copy: m_inputTensor already wraps inputTensorValues, which the
    // normalization above refilled in place. Nothing to create per frame.
    const char* inputNamesC[] = { inputName.c_str() };
    auto outputTensors = session->Run(Ort::RunOptions{ nullptr },
        inputNamesC, &m_inputTensor, 1, outputNamesC.data(), outputNamesC.size());

    // Identify output indices for score and heatmap upon first execution
    if (scoreIdx < 0) {
        for (size_t idx = 0; idx < outputTensors.size(); ++idx) {
            auto info = outputTensors[idx].GetTensorTypeAndShapeInfo();
            if (info.GetElementCount() == 1 && scoreIdx < 0)    scoreIdx = static_cast<int>(idx);
            else if (info.GetShape().size() >= 3 && mapIdx < 0) mapIdx = static_cast<int>(idx);
        }
        if (scoreIdx < 0)
            throw std::runtime_error("Anomaly model does not expose a scalar score output.");
    }

    // Contract 3.0: the score is FINAL and directly comparable to
    // calibrated_threshold (CONTRACT.md §2.1). Publish it as-is — no runtime
    // min-max, no runtime statistics. The OK/REJECT decision matches
    // calibrate_threshold.py (score >= threshold, Youden).
    const float rawScore = outputTensors[scoreIdx].GetTensorMutableData<float>()[0];
    outAnomalyScore = rawScore;
    outStatus = (rawScore >= scoreThreshold) ? "REJECT" : "OK";

    // Heatmap rendering. Mirrors render_reference.py::render() step by step;
    // each block cites the corresponding line of the Python reference. The
    // reference is the normative spec (tools/parity_check.py verifies parity).
    if (writeHeatmap && pOutputHeatmap && mapIdx >= 0) {
        auto s = outputTensors[mapIdx].GetTensorTypeAndShapeInfo().GetShape();
        int64_t hmH = s[s.size() - 2], hmW = s[s.size() - 1];
        float* pHeatmap = outputTensors[mapIdx].GetTensorMutableData<float>();

        // [Passo 1] render_reference.py:131 — source: the final map from the graph.
        cv::Mat floatHeatmap(static_cast<int>(hmH), static_cast<int>(hmW), CV_32FC1, pHeatmap);

        // [Passo 2/2b] render_reference.py:134,141 — display-normalize with the
        // fixed global range and quantize to uint8:
        // g = round(clamp((map - mapMin)/(mapMax - mapMin), 0, 1) * 255).
        // convertTo saturates to [0,255], which is the clamp; the fused
        // scale/offset is the affine map (mapMin->0, mapMax->255).
        const double alpha = 255.0 / (mapMax - mapMin);
        const double beta = -255.0 * mapMin / (mapMax - mapMin);
        floatHeatmap.convertTo(m_uint8Heatmap, CV_8UC1, alpha, beta);

        // [Passo 3] render_reference.py:144 — resize to the original resolution,
        // INTER_LINEAR, AFTER the normalization ([D-Q2]).
        cv::resize(m_uint8Heatmap, m_largeHeatmap, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);

        // [Passo 4] render_reference.py:148 — colormap via the shared LUT. The
        // LUT is RGB, so applyColorMap emits RGB ([D-Q3]). No library colormap.
        cv::applyColorMap(m_largeHeatmap, m_colorHeat, m_lutRgb);

        // [Passo 5] render_reference.py:152 — alpha blend on the original image
        // (assumed RGB, [D-Q4/Q5]): out = (1-a)*img + a*heatmap.
        cv::addWeighted(rawImg, 1.0 - m_blendAlpha, m_colorHeat, m_blendAlpha, 0.0, m_overlay);

        // [Passo 6] render_reference.py:158 — pred_mask overlay: threshold the
        // FINAL map against calibrated_threshold_pixel, draw red contours
        // (RGB (255,0,0)) thickness 2. Skipped when the model exports no pixel
        // threshold ([D-Q6]).
        if (hasPixelThreshold) {
            // MatExpr comparison yields a CV_8UC1 mask (0/255), same as the
            // reference's (m >= pixel_threshold) at model resolution.
            cv::Mat maskModel = floatHeatmap >= pixelThreshold;
            cv::resize(maskModel, m_maskFull, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(m_maskFull, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            cv::drawContours(m_overlay, contours, -1, cv::Scalar(255, 0, 0), 2);
        }

        // Output is RGB ([D-Q4]); copy the overlay to the shared buffer.
        memcpy(pOutputHeatmap, m_overlay.data, static_cast<size_t>(width) * height * channels);
    }
}