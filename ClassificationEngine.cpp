#include "ClassificationEngine.h"
#include <fmt/core.h>
#include <xmmintrin.h> // For _MM_SET_FLUSH_ZERO_MODE
#include <pmmintrin.h> // For _MM_SET_DENORMALS_ZERO_MODE
#include <cstring>
#include <chrono>
#include <string>
#include <thread>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <algorithm>

ClassificationEngine::ClassificationEngine()
{
    env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ClassificationEngineEnv");
    memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
}

void ClassificationEngine::Initialize(const std::wstring& modelPath)
{
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    bool hardwareAccelerated = false;

#if defined(ORT_EP_GPU)
    // Setup hardware acceleration (GPU build: TensorRT -> CUDA -> CPU)
    try {
        OrtTensorRTProviderOptions trt_options{};
        trt_options.device_id = 0;
        trt_options.trt_fp16_enable = 1;
        trt_options.trt_engine_cache_enable = 1;
        trt_options.trt_engine_cache_path = "./trt_cache";
        sessionOptions.AppendExecutionProvider_TensorRT(trt_options);
        hardwareAccelerated = true;
        fmt::print("TensorRT Execution Provider appended successfully for Classification.\n");
    }
    catch (const Ort::Exception& e) {
        fmt::print(stderr, "TensorRT not available, falling back to CUDA: {}\n", e.what());
    }

#elif defined(ORT_EP_OPENVINO)
    // Setup hardware acceleration (OpenVINO build: Intel GPU -> CPU via device AUTO)
    try {
        std::unordered_map<std::string, std::string> ov_options;
        ov_options["device_type"] = "CPU";
        sessionOptions.AppendExecutionProvider_OpenVINO_V2(ov_options);
        hardwareAccelerated = true;
        fmt::print("OpenVINO Execution Provider appended successfully for Classification.\n");
    }
    catch (const Ort::Exception& e) {
        fmt::print(stderr, "OpenVINO not available, falling back to default CPU: {}\n", e.what());
    }

#elif defined(ORT_EP_CPU)
    // CPU build: no execution provider is appended, the built-in ONNX Runtime
    // CPU EP is used. hardwareAccelerated stays false so the CPU-side
    // optimizations below are applied.
    fmt::print("CPU build: using the default ONNX Runtime CPU Execution Provider for Classification.\n");

#else
#error "No execution provider selected: define one of ORT_EP_GPU, ORT_EP_OPENVINO or ORT_EP_CPU."
#endif

    // Dynamic context switching based on the execution provider (same policy as AnomalyEngine)
    if (hardwareAccelerated) {
        // GPU/OpenVINO manage their own threading: one ORT thread avoids context switching
        sessionOptions.SetIntraOpNumThreads(1);
    }
    else {
        // CPU EP: the previous hardcoded SetIntraOpNumThreads(1) forced the whole
        // network onto a single core. Use the physical cores instead.
        fmt::print(stderr, "WARNING: Initializing CPU Execution Provider with aggressive optimizations.\n");

        sessionOptions.EnableCpuMemArena();
        sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

        // NOTE: with multiple concurrent workers (NUM_CONTROL_POINTS > 1) consider
        // dividing the cores among them to avoid thread thrashing
        unsigned int physicalCores = std::thread::hardware_concurrency() / 2;
        sessionOptions.SetIntraOpNumThreads(physicalCores > 0 ? physicalCores : 1);

        // Flush-To-Zero / Denormals-Are-Zero: prevents catastrophic slowdowns on near-zero floats
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    }

    session = std::make_unique<Ort::Session>(*env, modelPath.c_str(), sessionOptions);

    // Extract I/O names dynamically from the model graph and store them for Infer
    // (same pattern as AnomalyEngine: no hardcoded names)
    Ort::AllocatorWithDefaultOptions allocator;
    inputName = session->GetInputNameAllocated(0, allocator).get();

    outputNames.clear();
    for (size_t idx = 0; idx < session->GetOutputCount(); ++idx) {
        outputNames.emplace_back(session->GetOutputNameAllocated(idx, allocator).get());
    }

    // Extract the input shape dynamically from the model graph and store it for Infer
    Ort::TypeInfo typeInfo = session->GetInputTypeInfo(0);
    auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
    inputShape = tensorInfo.GetShape();

    // Handle dynamic dimensions (exported as -1 in ONNX):
    // force batch size to 1 and fall back to 256x256 for dynamic spatial dims
    if (inputShape[0] <= 0) inputShape[0] = 1;
    if (inputShape[2] <= 0) inputShape[2] = 256;
    if (inputShape[3] <= 0) inputShape[3] = 256;

    modelChannels = inputShape[1];
    modelHeight = inputShape[2];
    modelWidth = inputShape[3];

    // Pre-allocate the input tensor buffer reused by every Infer call
    size_t totalElements = 1;
    for (int64_t dim : inputShape) {
        totalElements *= dim;
    }
    inputTensorValues.assign(totalElements, 0.0f);

    // Warmup process using the extracted shape
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorValues.size(), inputShape.data(), inputShape.size());

    const char* inputNames[] = { inputName.c_str() };
    const char* warmupOutputNames[] = { outputNames[0].c_str() };

    // Multiple warmup runs: the first one pays graph compilation and lazy
    // allocations, the following ones stabilize the execution path. This runs
    // at Configure time, BEFORE the camera trigger starts, so the first real
    // frame does not hit a cold engine (the camera cannot be delayed in production)
    constexpr int kWarmupRuns = 50;
    double firstMs = 0.0;
    double lastMs = 0.0;
    for (int r = 0; r < kWarmupRuns; ++r) {
        auto startTime = std::chrono::steady_clock::now();
        session->Run(Ort::RunOptions{ nullptr }, inputNames, &inputTensor, 1, warmupOutputNames, 1);
        lastMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - startTime).count();
        if (r == 0) firstMs = lastMs;
    }

    fmt::print("Classification Warmup completed: {} runs, first {:.1f} ms, last {:.1f} ms.\n",
        kWarmupRuns, firstMs, lastMs);
}

void ClassificationEngine::Infer(const void* pInputImage, uint32_t width, uint32_t height, uint32_t channels,
    void* pOutputHeatmap, float& outAnomalyScore, std::string& outStatus)
{
    // Shape extracted from the model graph in Initialize
    cv::Mat rawImg(height, width, CV_8UC3, const_cast<void*>(pInputImage));
    cv::Mat resizeImage;
    cv::resize(rawImg, resizeImage,
        cv::Size(static_cast<int>(modelWidth), static_cast<int>(modelHeight)),
        0, 0, cv::INTER_LINEAR);

    // Standard ImageNet normalization (Matching AnomalyEngine)
    const float mean[] = { 0.485f, 0.456f, 0.406f };
    const float stddev[] = { 0.229f, 0.224f, 0.225f };

    // Vectorized HWC -> CHW conversion: split the interleaved image into planes,
    // then let convertTo apply scale/offset with SIMD directly into the (reused)
    // tensor buffer. Replaces the previous per-pixel scalar loop.
    const size_t planeSize = static_cast<size_t>(modelHeight) * static_cast<size_t>(modelWidth);
    std::vector<cv::Mat> planes8u;
    cv::split(resizeImage, planes8u);
    for (int c = 0; c < modelChannels; ++c) {
        const float scale = 1.0f / (255.0f * stddev[c]);
        const float offset = mean[c] / stddev[c];
        // Wrapper Mat aliasing the tensor buffer: convertTo writes in place, no copy
        cv::Mat planeF32(static_cast<int>(modelHeight), static_cast<int>(modelWidth), CV_32FC1,
            inputTensorValues.data() + static_cast<size_t>(c) * planeSize);
        planes8u[c].convertTo(planeF32, CV_32FC1, scale, -offset);
    }

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorValues.size(), inputShape.data(), inputShape.size());

    // Use the I/O names extracted from the model graph in Initialize
    const char* inputNamesC[] = { inputName.c_str() };
    std::vector<const char*> outputNamesC;
    outputNamesC.reserve(outputNames.size());
    for (const auto& name : outputNames) {
        outputNamesC.push_back(name.c_str());
    }

    auto outputTensors = session->Run(
        Ort::RunOptions{ nullptr }, inputNamesC, &inputTensor, 1, outputNamesC.data(), outputNamesC.size());

    auto outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

    if (outputShape.size() < 2) {
        throw std::runtime_error("Unexpected output tensor shape from Classification model.");
    }

    // The number of classes is the second dimension [1]
    const int64_t numClasses = outputShape[1];
    float* pProbs = outputTensors[0].GetTensorMutableData<float>();

    // Find the argmax (index of the highest probability) dynamically
    auto maxElementIter = std::max_element(pProbs, pProbs + numClasses);
    int bestClassId = static_cast<int>(std::distance(pProbs, maxElementIter));
    float bestConfidence = *maxElementIter;

    // Repurpose the interface variables to pass data to WorkerONNX
    outAnomalyScore = bestConfidence;
    outStatus = std::to_string(bestClassId);
}