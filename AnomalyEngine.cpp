#include "AnomalyEngine.h"
#include "AsyncLogger.h"
#include <xmmintrin.h> 
#include <pmmintrin.h> 
#include <algorithm>
#include <cstring>
#include <chrono>
#include <string>
#include <thread>
#include <unordered_map>
#include "OrtsessionConfig.h"
#include <array>
#include <stdexcept>

#if defined(ORT_EP_GPU)
#include <cuda_runtime.h>
#endif

namespace {
    // Minimal RFC 4648 base64 decoder...
    std::vector<uint8_t> Base64Decode(const std::string& in) {
        static constexpr char kPad = '=';
        std::array<int, 256> T{};
        T.fill(-1);
        const char* alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; ++i) T[static_cast<unsigned char>(alphabet[i])] = i;

        std::vector<uint8_t> out;
        out.reserve(in.size() * 3 / 4);
        int val = 0, bits = -8;
        for (unsigned char c : in) {
            if (c == kPad) break;
            if (T[c] == -1) continue;
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
    env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "AnomalyEngineEnv");
    memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
}

AnomalyEngine::~AnomalyEngine() {
#if defined(ORT_EP_GPU)
    // Safely free VRAM blocks allocated via CUDA to prevent memory leaks
    if (d_input) cudaFree(d_input);
    if (d_score) cudaFree(d_score);
    if (d_map) cudaFree(d_map);
#endif
}

void AnomalyEngine::Initialize(const std::wstring& modelPath, const RT::CpuPartition& cpuPartition)
{
    Ort::SessionOptions sessionOptions;

    // Check runtime hardware acceleration status to enable/disable CUDA bindings dynamically
    m_useGpuIoBinding = ConfigureOrtSessionOptions(sessionOptions, "Anomaly", cpuPartition);

    session = std::make_unique<Ort::Session>(*env, modelPath.c_str(), sessionOptions);
    Log::Info("ONNX Session created successfully for Anomaly Detection.");

    LoadContractMetadata();

    try {
        Ort::AllocatorWithDefaultOptions allocator;
        inputName = session->GetInputNameAllocated(0, allocator).get();

        outputNames.clear();
        for (size_t idx = 0; idx < session->GetOutputCount(); ++idx) {
            outputNames.emplace_back(session->GetOutputNameAllocated(idx, allocator).get());
        }

        // --- Extrapolate Output Indices BEFORE allocation ---
        for (size_t idx = 0; idx < session->GetOutputCount(); ++idx) {
            auto shape = session->GetOutputTypeInfo(idx).GetTensorTypeAndShapeInfo().GetShape();
            if (shape.size() == 1 || (shape.size() == 2 && shape[1] == 1)) {
                if (scoreIdx < 0) scoreIdx = static_cast<int>(idx);
            }
            else if (shape.size() >= 3) {
                if (mapIdx < 0) mapIdx = static_cast<int>(idx);
            }
        }
        if (scoreIdx < 0) throw std::runtime_error("Anomaly model missing scalar score output.");

        // --- Geometry Extrapolation & Sanitization ---
        Ort::TypeInfo typeInfo = session->GetInputTypeInfo(0);
        inputShape = typeInfo.GetTensorTypeAndShapeInfo().GetShape();
        if (inputShape[0] == -1) inputShape[0] = 1;

        modelChannels = inputShape[1];
        modelHeight = inputShape[2];
        modelWidth = inputShape[3];

        m_scoreShape = session->GetOutputTypeInfo(scoreIdx).GetTensorTypeAndShapeInfo().GetShape();
        for (auto& d : m_scoreShape) if (d < 0) d = 1;

        if (mapIdx >= 0) {
            m_mapShape = session->GetOutputTypeInfo(mapIdx).GetTensorTypeAndShapeInfo().GetShape();
            for (auto& d : m_mapShape) if (d < 0) d = 1;
        }

        // --- Calculate Buffer Sizes ---
        size_t inputElements = 1;
        for (int64_t d : inputShape) inputElements *= d;

        size_t mapElements = 1;
        if (mapIdx >= 0) {
            for (int64_t d : m_mapShape) mapElements *= d;
        }

        // Host allocations (required for pre-processing / fallback)
        inputTensorValues.assign(inputElements, 0.0f);
        m_splitPlanes.resize(modelChannels);
        m_hostScore.assign(1, 0.0f);
        if (mapIdx >= 0) m_hostMap.assign(mapElements, 0.0f);

        // --- Setup I/O Binding ---
        io_binding = std::make_unique<Ort::IoBinding>(*session);

#if defined(ORT_EP_GPU)
        if (m_useGpuIoBinding) {
            // Pre-allocate GPU VRAM strictly for Zero-Copy tensor operation
            if (cudaMalloc(&d_input, inputElements * sizeof(float)) != cudaSuccess) throw std::runtime_error("CUDA Malloc input");
            if (cudaMalloc(&d_score, sizeof(float)) != cudaSuccess) throw std::runtime_error("CUDA Malloc score");
            if (mapIdx >= 0 && cudaMalloc(&d_map, mapElements * sizeof(float)) != cudaSuccess) throw std::runtime_error("CUDA Malloc map");

            Ort::MemoryInfo cudaMemInfo("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);

            m_inputTensor = Ort::Value::CreateTensor<float>(cudaMemInfo, d_input, inputElements, inputShape.data(), inputShape.size());
            io_binding->BindInput(inputName.c_str(), m_inputTensor);

            m_scoreTensor = Ort::Value::CreateTensor<float>(cudaMemInfo, d_score, 1, m_scoreShape.data(), m_scoreShape.size());
            io_binding->BindOutput(outputNames[scoreIdx].c_str(), m_scoreTensor);

            if (mapIdx >= 0) {
                m_mapTensor = Ort::Value::CreateTensor<float>(cudaMemInfo, d_map, mapElements, m_mapShape.data(), m_mapShape.size());
                io_binding->BindOutput(outputNames[mapIdx].c_str(), m_mapTensor);
            }
            Log::Info("GPU I/O Binding established with Zero-Copy VRAM tensors.");
        }
        else
#endif
        {
            // Fallback: CPU Direct Memory Mapping
            m_inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(), inputElements, inputShape.data(), inputShape.size());
            io_binding->BindInput(inputName.c_str(), m_inputTensor);

            m_scoreTensor = Ort::Value::CreateTensor<float>(memoryInfo, m_hostScore.data(), 1, m_scoreShape.data(), m_scoreShape.size());
            io_binding->BindOutput(outputNames[scoreIdx].c_str(), m_scoreTensor);

            if (mapIdx >= 0) {
                m_mapTensor = Ort::Value::CreateTensor<float>(memoryInfo, m_hostMap.data(), mapElements, m_mapShape.data(), m_mapShape.size());
                io_binding->BindOutput(outputNames[mapIdx].c_str(), m_mapTensor);
            }
            Log::Info("CPU I/O Binding established with System RAM mapping.");
        }

        // --- Warmup Runs (Must use IoBinding to compile correct graph segment) ---
        constexpr int kWarmupRuns = 50;
        double firstMs = 0.0, lastMs = 0.0;
        for (int r = 0; r < kWarmupRuns; ++r) {
            auto startTime = std::chrono::steady_clock::now();
            session->Run(Ort::RunOptions{ nullptr }, *io_binding);
            lastMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - startTime).count();
            if (r == 0) firstMs = lastMs;
        }

        Log::Info("Warmup completed: {} runs, first {:.1f} ms, last {:.1f} ms. Engine locked.", kWarmupRuns, firstMs, lastMs);
    }
    catch (const Ort::Exception& e) {
        Log::Error("### ERROR DURING INITIALIZATION: {}", e.what());
        throw;
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
    cv::Mat rawImg(height, width, CV_8UC3, const_cast<void*>(pInputImage));
    cv::resize(rawImg, m_resizeImage, cv::Size(static_cast<int>(modelWidth), static_cast<int>(modelHeight)), 0, 0, cv::INTER_AREA);

    const float mean[] = { 0.485f, 0.456f, 0.406f };
    const float stddev[] = { 0.229f, 0.224f, 0.225f };
    const size_t planeSize = static_cast<size_t>(modelHeight) * static_cast<size_t>(modelWidth);

    cv::split(m_resizeImage, m_splitPlanes);
    for (int c = 0; c < modelChannels; ++c) {
        const float scale = normalizationInGraph ? (1.0f / 255.0f) : (1.0f / (255.0f * stddev[c]));
        const float offset = normalizationInGraph ? 0.0f : (mean[c] / stddev[c]);

        // Write processed data directly into host buffer
        cv::Mat planeF32(static_cast<int>(modelHeight), static_cast<int>(modelWidth), CV_32FC1,
            inputTensorValues.data() + static_cast<size_t>(c) * planeSize);
        m_splitPlanes[c].convertTo(planeF32, CV_32FC1, scale, -offset);
    }

#if defined(ORT_EP_GPU)
    // Synchronous memory transfer: Only triggers if hardware acceleration is active
    if (m_useGpuIoBinding) {
        cudaMemcpy(d_input, inputTensorValues.data(), inputTensorValues.size() * sizeof(float), cudaMemcpyHostToDevice);
    }
#endif

    // Run inference matching exactly the bound VRAM/RAM buffers
    session->Run(Ort::RunOptions{ nullptr }, *io_binding);

#if defined(ORT_EP_GPU)
    if (m_useGpuIoBinding) {
        // Retrieve processed buffers from VRAM
        cudaMemcpy(m_hostScore.data(), d_score, sizeof(float), cudaMemcpyDeviceToHost);
        if (mapIdx >= 0) {
            cudaMemcpy(m_hostMap.data(), d_map, m_hostMap.size() * sizeof(float), cudaMemcpyDeviceToHost);
        }
    }
#endif

    const float rawScore = m_hostScore[0];
    outAnomalyScore = rawScore;
    outStatus = (rawScore >= scoreThreshold) ? "REJECT" : "OK";

    if (writeHeatmap && pOutputHeatmap && mapIdx >= 0) {
        // Point OpenCV directly to our bound host map buffer
        cv::Mat floatHeatmap(static_cast<int>(m_mapShape[2]), static_cast<int>(m_mapShape[3]), CV_32FC1, m_hostMap.data());

        const double alpha = 255.0 / (mapMax - mapMin);
        const double beta = -255.0 * mapMin / (mapMax - mapMin);
        floatHeatmap.convertTo(m_uint8Heatmap, CV_8UC1, alpha, beta);
        cv::resize(m_uint8Heatmap, m_largeHeatmap, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
        cv::applyColorMap(m_largeHeatmap, m_colorHeat, m_lutRgb);
        cv::addWeighted(rawImg, 1.0 - m_blendAlpha, m_colorHeat, m_blendAlpha, 0.0, m_overlay);

        if (hasPixelThreshold) {
            cv::Mat maskModel = floatHeatmap >= pixelThreshold;
            cv::resize(maskModel, m_maskFull, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(m_maskFull, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            cv::drawContours(m_overlay, contours, -1, cv::Scalar(255, 0, 0), 2);
        }
        memcpy(pOutputHeatmap, m_overlay.data, static_cast<size_t>(width) * height * channels);
    }
}