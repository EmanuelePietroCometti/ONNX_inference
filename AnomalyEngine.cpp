#include "AnomalyEngine.h"
#include <fmt/core.h>
#include <xmmintrin.h> // For _MM_SET_FLUSH_ZERO_MODE
#include <pmmintrin.h> // For _MM_SET_DENORMALS_ZERO_MODE
#include <algorithm>
#include <cstring>
#include <chrono>
#include <string>
#include <thread>
#include <unordered_map>
#include <opencv2/opencv.hpp>

AnomalyEngine::AnomalyEngine()
{
    // Initialize ONNX Environment (Warning level, log id)
    env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "AnomalyEngineEnv");
    memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
}

void AnomalyEngine::Initialize(const std::wstring& modelPath)
{
    Ort::SessionOptions sessionOptions;

    // Threading optimizations
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    bool hardwareAccelerated = false;

#if defined(ORT_EP_GPU)
    // Hardware Acceleration Setup (GPU build, order matters: TRT -> CUDA -> CPU)
    try {
        OrtTensorRTProviderOptions trt_options{};
        trt_options.device_id = 0;
        trt_options.trt_fp16_enable = 1; // Enable FP16 for massive speedup on modern GPUs
        trt_options.trt_engine_cache_enable = 1; // Cache engine to disk for fast consecutive warmups
        trt_options.trt_engine_cache_path = "./trt_cache";

        sessionOptions.AppendExecutionProvider_TensorRT(trt_options);
        hardwareAccelerated = true;
        fmt::print("TensorRT Execution Provider appended successfully.\n");
    }
    catch (const Ort::Exception& e) {
        fmt::print(stderr, "TensorRT not available, falling back to CUDA: {}\n", e.what());
    }

    try {
        OrtCUDAProviderOptions cuda_options{};
        cuda_options.device_id = 0;
        sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
        hardwareAccelerated = true;
        fmt::print("CUDA Execution Provider appended successfully.\n");
    }
    catch (const Ort::Exception& e) {
        fmt::print(stderr, "CUDA not available, falling back to CPU: {}\n", e.what());
    }

#elif defined(ORT_EP_OPENVINO)
    // Hardware Acceleration Setup (OpenVINO build: Intel GPU -> CPU via device AUTO)
    try {
        std::unordered_map<std::string, std::string> ov_options;
        ov_options["device_type"] = "CPU"; // Prefer Intel GPU, fall back to CPU plugin
        sessionOptions.AppendExecutionProvider_OpenVINO_V2(ov_options);
        hardwareAccelerated = true;
        fmt::print("OpenVINO Execution Provider appended successfully.\n");
    }
    catch (const Ort::Exception& e) {
        fmt::print(stderr, "OpenVINO not available, falling back to default CPU: {}\n", e.what());
    }

#elif defined(ORT_EP_CPU)
    // CPU build: no execution provider is appended, the built-in ONNX Runtime
    // CPU EP is used. hardwareAccelerated stays false so the CPU-side
    // optimizations below are applied.
    fmt::print("CPU build: using the default ONNX Runtime CPU Execution Provider.\n");

#else
#error "No execution provider selected: define one of ORT_EP_GPU, ORT_EP_OPENVINO or ORT_EP_CPU."
#endif

    // Dynamic Context Switching based on Execution Provider
    if (hardwareAccelerated) {
        // GPU Mode: Keep threading to 1 to feed the GPU efficiently without CPU context switching
        sessionOptions.SetIntraOpNumThreads(1);
    }
    else {
        // CPU Execution Provider: Maximize x64 architecture utilization
        fmt::print(stderr, "WARNING: Initializing CPU Execution Provider with aggressive optimizations.\n");

        // Enable memory arena to avoid continuous OS memory allocations
        sessionOptions.EnableCpuMemArena();

        // Sequential execution avoids locking overhead on CPU
        sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

        // Thread tuning: If this AnomalyEngine is the ONLY worker, use physical cores.
        // NOTE: If you are spawning multiple AnomalyEngine instances in custom C++ threads,
        // leave this at 1 to prevent OS thread thrashing.
        unsigned int physicalCores = std::thread::hardware_concurrency() / 2;
        sessionOptions.SetIntraOpNumThreads(physicalCores > 0 ? physicalCores : 1);

        // Hardware Math: Enable Flush-To-Zero (FTZ) and Denormals-Are-Zero (DAZ).
        // This prevents catastrophic CPU slowdowns when dealing with near-zero floats in CNNs.
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    }

    // Create the session
    session = std::make_unique<Ort::Session>(*env, modelPath.c_str(), sessionOptions);
    fmt::print("ONNX Session created successfully for Anomaly Detection.\n");

    // Read global_min / global_max / threshold from the ONNX metadata and
    // pre-compute the normalized threshold used at inference time
    LoadNormalizationMetadata();

    try {
        Ort::AllocatorWithDefaultOptions allocator;

        // Extract Input Name dynamically and store it for Infer
        auto inputNamePtr = session->GetInputNameAllocated(0, allocator);
        inputName = inputNamePtr.get();
        const char* inputNames[] = { inputName.c_str() };

        // Extract ALL Output Names dynamically and store them for Infer
        // (Requesting the first one is enough to force the whole graph to execute during warmup)
        outputNames.clear();
        for (size_t idx = 0; idx < session->GetOutputCount(); ++idx) {
            auto outputNamePtr = session->GetOutputNameAllocated(idx, allocator);
            outputNames.emplace_back(outputNamePtr.get());
        }
        const char* warmupOutputNames[] = { outputNames[0].c_str() };

        // Extract Input Shape
        Ort::TypeInfo typeInfo = session->GetInputTypeInfo(0);
        auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> inputShape = tensorInfo.GetShape();

        // Handle dynamic batch dimensions (often exported as -1 in ONNX)
        if (inputShape[0] == -1) {
            inputShape[0] = 1; // Force batch size to 1 for real-time single-image inference
        }

        // Calculate the total number of elements required for the dummy tensor
        size_t totalElements = 1;
        for (int64_t dim : inputShape) {
            totalElements *= dim;
        }

        // Create Dummy Input Tensor (Filled with zeros)
        // Assuming float32 input, which is standard for CNN backbones
        std::vector<float> dummyInput(totalElements, 0.0f);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, dummyInput.data(), dummyInput.size(), inputShape.data(), inputShape.size());

        // Execute Warmup Runs
        // The first run triggers engine building (TensorRT compilation) and
        // memory allocation; the following ones stabilize the execution path so the
        // first REAL frame does not hit a cold engine. This happens at Configure
        // time, before the camera trigger starts.
        constexpr int kWarmupRuns = 50;
        double firstMs = 0.0;
        double lastMs = 0.0;
        for (int r = 0; r < kWarmupRuns; ++r) {
            auto startTime = std::chrono::steady_clock::now();
            session->Run(Ort::RunOptions{ nullptr }, inputNames, &inputTensor, 1, warmupOutputNames, 1);
            lastMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - startTime).count();
            if (r == 0) firstMs = lastMs;
        }

        fmt::print("Warmup completed successfully: {} runs, first {:.1f} ms, last {:.1f} ms. Engine is locked and ready for real-time inference.\n",
            kWarmupRuns, firstMs, lastMs);

    }
    catch (const Ort::Exception& e) {
        fmt::print(stderr, "### ERROR DURING WARMUP: {}\n", e.what());
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
                fmt::print("Metadata '{}' = {}\n", key, target);
                return true;
            }
        }
        return false;
        };

    const bool hasMin = lookupFloat({ "calibration_global_min", "min" }, globalMin);
    const bool hasMax = lookupFloat({ "calibration_global_max", "max" }, globalMax);
    const bool hasThreshold = lookupFloat({ "calibrated_threshold", "threshold", "global_threshold" }, rawThreshold);

    if (!hasMin || !hasMax || !hasThreshold) {
        fmt::print(stderr,
            "### WARNING: normalization metadata incomplete (min:{} max:{} threshold:{}). "
            "Falling back to defaults, scores will NOT match the Python model.\n",
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

    fmt::print("Normalization ready: globalMin={}, globalMax={}, rawThreshold={}, normalizedThreshold={}\n",
        globalMin, globalMax, rawThreshold, normalizedThreshold);
}

void AnomalyEngine::Infer(const void* pInputImage, uint32_t width, uint32_t height, uint32_t channels, void* pOutputHeatmap, float& outAnomalyScore, std::string& outStatus)
{
    const uint8_t* rawImage = static_cast<const uint8_t*>(pInputImage);

    // Dynamically extract input tensor shape from ONNX session
    Ort::TypeInfo typeInfo = session->GetInputTypeInfo(0);
    auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> inputShape = tensorInfo.GetShape();

    const int64_t batchSize = (inputShape[0] == -1) ? 1 : inputShape[0];
    const int64_t modelChannels = inputShape[1];
    const int64_t modelHeight = inputShape[2];
    const int64_t modelWidth = inputShape[3];

    inputShape[0] = batchSize;

    cv::Mat rawImg(height, width, CV_8UC3, const_cast<void*>(pInputImage));

    cv::Mat resizeImage;
    cv::resize(
        rawImg,
        resizeImage,
        cv::Size(static_cast<int>(modelWidth),
            static_cast<int>(modelHeight)),
        0,
        0,
        cv::INTER_LINEAR
    );

    size_t inputTensorSize = batchSize * modelChannels * modelHeight * modelWidth;
    std::vector<float> inputTensorValues(inputTensorSize);

    // Standard image net normalization
    const float mean[] = { 0.485f, 0.456f, 0.406f };
    const float stddev[] = { 0.229f, 0.224f, 0.225f };

    // HWC -> CHW mapping
    size_t chw_base = 0;
    for (int c = 0; c < modelChannels; ++c) {
        float scale = 1.0f / (255.0f * stddev[c]);
        float offset = mean[c] / stddev[c];

        for (int h = 0; h < modelHeight; ++h) {
            const uint8_t* rowPtr = resizeImage.ptr<uint8_t>(h);
            for (int w = 0; w < modelWidth; ++w) {
                float pixel = static_cast<float>(rowPtr[w * modelChannels + c]);
                inputTensorValues[chw_base++] = pixel * scale - offset;
            }
        }
    }

    // Create input tensor for ONNX engine
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo,
        inputTensorValues.data(),
        inputTensorValues.size(),
        inputShape.data(),
        inputShape.size()
    );


    // Run inference using the names extracted from the model graph in Initialize
    const char* inputNamesC[] = { inputName.c_str() };
    std::vector<const char*> outputNamesC;
    outputNamesC.reserve(outputNames.size());
    for (const auto& name : outputNames) {
        outputNamesC.push_back(name.c_str());
    }

    auto outputTensors = session->Run(
        Ort::RunOptions{ nullptr },
        inputNamesC,
        &inputTensor,
        1,
        outputNamesC.data(),
        outputNamesC.size()
    );

    // Post processing results.
    // Do NOT assume the graph output order: identify the scalar score and the
    // spatial anomaly map by inspecting each tensor's shape. Indexing the wrong
    // tensor here means reading garbage dimensions and crashing inside OpenCV.
    int scoreIdx = -1;
    int mapIdx = -1;
    for (size_t idx = 0; idx < outputTensors.size(); ++idx) {
        auto info = outputTensors[idx].GetTensorTypeAndShapeInfo();
        std::vector<int64_t> shape = info.GetShape();

        if (info.GetElementCount() == 1 && scoreIdx < 0) {
            scoreIdx = static_cast<int>(idx); // Single element -> anomaly score
        }
        else if (shape.size() >= 3 && mapIdx < 0) {
            mapIdx = static_cast<int>(idx); // (N,C,H,W) or (N,H,W) -> anomaly map
        }
    }

    if (scoreIdx < 0) {
        throw std::runtime_error("Anomaly model does not expose a scalar score output.");
    }

    // Min-max normalize the raw score with the global stats read from the ONNX
    // metadata, then decide against the normalized threshold: this reproduces
    // exactly the post-processing of the Python (Anomalib) pipeline
    const float rawScore = outputTensors[scoreIdx].GetTensorMutableData<float>()[0];
    outAnomalyScore = std::clamp((rawScore - globalMin) / (globalMax - globalMin), 0.0f, 1.0f);
    outStatus = (outAnomalyScore > normalizedThreshold) ? "REJECT" : "OK";

    // The heatmap is only written if the model actually exposes a spatial map output
    if (pOutputHeatmap && mapIdx >= 0) {
        auto heatmapShape = outputTensors[mapIdx].GetTensorTypeAndShapeInfo().GetShape();
        // Spatial dims are always the last two, regardless of rank (N,C,H,W) or (N,H,W)
        int64_t hmHeight = heatmapShape[heatmapShape.size() - 2];
        int64_t hmWidth = heatmapShape[heatmapShape.size() - 1];
        float* pHeatmap = outputTensors[mapIdx].GetTensorMutableData<float>();

        // Wrap the raw float output into a 1-channel OpenCV Mat
        cv::Mat floatHeatmap(static_cast<int>(hmHeight), static_cast<int>(hmWidth), CV_32FC1, pHeatmap);

        // Normalize with the GLOBAL min/max from the model metadata (not the
        // per-image min/max): the heatmap intensity stays comparable across
        // frames and matches the Python pipeline
        cv::Mat normHeatmap = (floatHeatmap - globalMin) / (globalMax - globalMin);
        cv::Mat uint8Heatmap;
        normHeatmap.convertTo(uint8Heatmap, CV_8UC1, 255.0, 0.0); // convertTo saturates to [0,255]

        // Resize back to original camera resolution
        cv::Mat largeHeatmap;
        cv::resize(uint8Heatmap, largeHeatmap, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);

        // Convert Grayscale to RGB so it matches the IPC expectations
        cv::Mat rgbHeatmap;
        cv::cvtColor(largeHeatmap, rgbHeatmap, cv::COLOR_GRAY2BGR);

        // Copy final data directly to IPC Shared Memory
        size_t mapSize = width * height * channels;
        memcpy(pOutputHeatmap, rgbHeatmap.data, mapSize);
    }
}