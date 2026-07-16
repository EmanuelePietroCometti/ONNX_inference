#include "ClassificationEngine.h"
#include <fmt/core.h>
#include <xmmintrin.h> // For _MM_SET_FLUSH_ZERO_MODE
#include <pmmintrin.h> // For _MM_SET_DENORMALS_ZERO_MODE
#include <cstring>
#include <chrono>
#include <string>
#include <thread>
#include <unordered_map>
#include <algorithm>
#include "OrtsessionConfig.h"

ClassificationEngine::ClassificationEngine()
{
    env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ClassificationEngineEnv");
    memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
}

void ClassificationEngine::Initialize(const std::wstring& modelPath)
{
    Ort::SessionOptions sessionOptions;

    // Execution providers + session options: identical policy across all engines.
    ConfigureOrtSessionOptions(sessionOptions, "Classification");

    session = std::make_unique<Ort::Session>(*env, modelPath.c_str(), sessionOptions);

    // Extract I/O names dynamically from the model graph and store them for Infer
    // (no hardcoded names)
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
    // force batch size to 1 and fall back to 224x224 for dynamic spatial dims
    if (inputShape[0] <= 0) inputShape[0] = 1;
    if (inputShape[2] <= 0) inputShape[2] = 224;
    if (inputShape[3] <= 0) inputShape[3] = 224;

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
    // allocations, the following ones stabilize the execution path. This runs at
    // Configure time, BEFORE the camera trigger starts, so the first real frame
    // does not hit a cold engine (the camera cannot be delayed in production).
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
    auto t0 = std::chrono::steady_clock::now();

    // Wrap the raw pointer into a cv::Mat without copying data
    cv::Mat rawImg(height, width, CV_8UC3, const_cast<void*>(pInputImage));
    cv::Mat resizeImage;

    // Resize using Nearest Neighbor interpolation for maximum CPU speed
    cv::resize(rawImg, resizeImage,
        cv::Size(static_cast<int>(modelWidth), static_cast<int>(modelHeight)),
        0, 0, cv::INTER_NEAREST);

    auto t1 = std::chrono::steady_clock::now();

    // Standard ImageNet normalization parameters
    const float mean[] = { 0.485f, 0.456f, 0.406f };
    const float stddev[] = { 0.229f, 0.224f, 0.225f };

    // Ensure the split planes vector is pre-allocated to avoid heap allocations
    if (m_splitPlanes.empty()) {
        m_splitPlanes.resize(modelChannels);
    }

    // Vectorized HWC -> CHW conversion
    const size_t planeSize = static_cast<size_t>(modelHeight) * static_cast<size_t>(modelWidth);
    cv::split(resizeImage, m_splitPlanes);

    for (int c = 0; c < modelChannels; ++c) {
        const float scale = 1.0f / (255.0f * stddev[c]);
        const float offset = mean[c] / stddev[c];

        // Wrapper Mat aliasing the contiguous tensor buffer: convertTo writes in-place
        cv::Mat planeF32(static_cast<int>(modelHeight), static_cast<int>(modelWidth), CV_32FC1,
            inputTensorValues.data() + static_cast<size_t>(c) * planeSize);
        m_splitPlanes[c].convertTo(planeF32, CV_32FC1, scale, -offset);
    }

    // Create the input tensor wrapping the pre-allocated and populated buffer
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorValues.size(), inputShape.data(), inputShape.size());

    const char* inputNamesC[] = { inputName.c_str() };

    // Use thread_local to initialize the output names array only once, 
    // avoiding expensive vector allocations during every frame inference
    thread_local std::vector<const char*> outputNamesC;
    if (outputNamesC.empty()) {
        outputNamesC.reserve(outputNames.size());
        for (const auto& name : outputNames) {
            outputNamesC.push_back(name.c_str());
        }
    }

    auto t2 = std::chrono::steady_clock::now();

    // Execute inference
    auto outputTensors = session->Run(
        Ort::RunOptions{ nullptr }, inputNamesC, &inputTensor, 1, outputNamesC.data(), outputNamesC.size());

    auto t3 = std::chrono::steady_clock::now();

    // Validate output shape
    auto outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    if (outputShape.size() < 2) {
        throw std::runtime_error("Unexpected output tensor shape from Classification model.");
    }

    // Post-processing: Extract probabilities and find the argmax (predicted class)
    const int64_t numClasses = outputShape[1];
    float* pProbs = outputTensors[0].GetTensorMutableData<float>();

    auto maxElementIter = std::max_element(pProbs, pProbs + numClasses);
    int bestClassId = static_cast<int>(std::distance(pProbs, maxElementIter));
    float bestConfidence = *maxElementIter;

    auto t4 = std::chrono::steady_clock::now();

    // Assign outputs for the WorkerONNX pipeline
    outAnomalyScore = bestConfidence;
    outStatus = std::to_string(bestClassId);

    // Profiling and metrics
    using ms = std::chrono::duration<double, std::milli>;
    accResize += ms(t1 - t0).count();
    accNorm += ms(t2 - t1).count();
    accRun += ms(t3 - t2).count();
    accPost += ms(t4 - t3).count();
    maxRun = std::max(maxRun, ms(t3 - t2).count());

    if (++frameCount % 100 == 0) {
        fmt::print("avg over {}: resize {:.2f} | norm {:.2f} | RUN {:.2f} (max {:.2f}) | post {:.2f}\n",
            frameCount, accResize / frameCount, accNorm / frameCount,
            accRun / frameCount, maxRun, accPost / frameCount);
    }
}