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

    // Read global_min / global_max / threshold from the ONNX metadata and
    // pre-compute the normalized threshold used at inference time
    LoadNormalizationMetadata();

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

void AnomalyEngine::LoadNormalizationMetadata()
{
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::ModelMetadata metadata = session->GetModelMetadata();

    // Different export pipelines embed the same value under different keys,
    // so each stat is looked up through a list of known aliases
    auto lookupFloat = [&](std::initializer_list<const char*> keys, float& target) -> bool {
        for (const char* key : keys) {
            auto value = metadata.LookupCustomMetadataMapAllocated(key, allocator);
            if (value) {
                target = std::stof(value.get());
                Log::Info("Metadata '{}' = {}", key, target);
                return true;
            }
        }
        return false;
        };

    const bool hasMin = lookupFloat({ "calibration_global_min", "min" }, globalMin);
    const bool hasMax = lookupFloat({ "calibration_global_max", "max" }, globalMax);
    const bool hasThreshold = lookupFloat({ "calibrated_threshold", "threshold", "global_threshold" }, rawThreshold);

    if (!hasMin || !hasMax || !hasThreshold) {
        Log::Warning(
            "### WARNING: normalization metadata incomplete (min:{} max:{} threshold:{}). "
            "Falling back to defaults, scores will NOT match the Python model.",
            hasMin, hasMax, hasThreshold);
    }

    // Guard against a degenerate range that would divide by zero
    const float range = globalMax - globalMin;
    if (range <= 0.0f) {
        throw std::runtime_error("Invalid normalization metadata: global_max must be greater than global_min.");
    }

    // Same min-max mapping Anomalib applies in Python: comparing the normalized
    // score against this normalized threshold is equivalent to comparing the raw
    // score against the raw threshold, so the OK/REJECT decision is identical
    normalizedThreshold = (rawThreshold - globalMin) / range;

    Log::Info("Normalization ready: globalMin={}, globalMax={}, rawThreshold={}, normalizedThreshold={}",
        globalMin, globalMax, rawThreshold, normalizedThreshold);
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

    // Image normalization parameters (standard ImageNet statistics)
    const float mean[] = { 0.485f, 0.456f, 0.406f };
    const float stddev[] = { 0.229f, 0.224f, 0.225f };
    const size_t planeSize = static_cast<size_t>(modelHeight) * static_cast<size_t>(modelWidth);

    // Convert HWC format to planar 3xHW format and apply normalization
    cv::split(m_resizeImage, m_splitPlanes);
    for (int c = 0; c < modelChannels; ++c) {
        const float scale = 1.0f / (255.0f * stddev[c]);
        const float offset = mean[c] / stddev[c];

        // Map tensor memory to cv::Mat for efficient pixel-wise operation
        cv::Mat planeF32(static_cast<int>(modelHeight), static_cast<int>(modelWidth), CV_32FC1,
            inputTensorValues.data() + static_cast<size_t>(c) * planeSize);

        // Normalize: dst = (pixel * scale) - offset
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

    // Process raw anomaly score: normalize to [0, 1] using metadata min/max
    const float rawScore = outputTensors[scoreIdx].GetTensorMutableData<float>()[0];
    outAnomalyScore = std::clamp((rawScore - globalMin) / (globalMax - globalMin), 0.0f, 1.0f);
    outStatus = (outAnomalyScore > normalizedThreshold) ? "REJECT" : "OK";

    // Optional: Generate and scale heatmap if requested and supported by model
    if (writeHeatmap && pOutputHeatmap && mapIdx >= 0) {
        auto s = outputTensors[mapIdx].GetTensorTypeAndShapeInfo().GetShape();
        int64_t hmH = s[s.size() - 2], hmW = s[s.size() - 1];
        float* pHeatmap = outputTensors[mapIdx].GetTensorMutableData<float>();
        cv::Mat floatHeatmap(static_cast<int>(hmH), static_cast<int>(hmW), CV_32FC1, pHeatmap);

        // Scale raw heatmap values to uint8 [0, 255] range
        const double alpha = 255.0 / (globalMax - globalMin);
        const double beta = -255.0 * globalMin / (globalMax - globalMin);
        floatHeatmap.convertTo(m_uint8Heatmap, CV_8UC1, alpha, beta);

        // Resize back to original input image size and convert to BGR for output
        cv::resize(m_uint8Heatmap, m_largeHeatmap, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
        cv::cvtColor(m_largeHeatmap, m_rgbHeatmap, cv::COLOR_GRAY2BGR);

        // Copy output buffer to the user-provided destination
        memcpy(pOutputHeatmap, m_rgbHeatmap.data, static_cast<size_t>(width) * height * channels);
    }
}