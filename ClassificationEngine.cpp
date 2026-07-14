#include "ClassificationEngine.h"
#include <fmt/core.h>
#include <cstring>
#include <chrono>
#include <string>
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
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#if defined(ORT_EP_OPENVINO)
    // Setup hardware acceleration (OpenVINO build: Intel GPU -> CPU via device AUTO)
    try {
        std::unordered_map<std::string, std::string> ov_options;
        ov_options["device_type"] = "AUTO:GPU,CPU";
        sessionOptions.AppendExecutionProvider_OpenVINO_V2(ov_options);
        fmt::print("OpenVINO Execution Provider appended successfully for Classification.\n");
    }
    catch (const Ort::Exception& e) {
        fmt::print(stderr, "OpenVINO not available, falling back to default CPU: {}\n", e.what());
    }
#else
    // Setup hardware acceleration (GPU build: TensorRT -> CUDA -> CPU)
    try {
        OrtTensorRTProviderOptions trt_options{};
        trt_options.device_id = 0;
        trt_options.trt_fp16_enable = 1;
        trt_options.trt_engine_cache_enable = 1;
        trt_options.trt_engine_cache_path = "./trt_cache";
        sessionOptions.AppendExecutionProvider_TensorRT(trt_options);
        fmt::print("TensorRT Execution Provider appended successfully for Classification.\n");
    }
    catch (const Ort::Exception& e) {
        fmt::print(stderr, "TensorRT not available, falling back to CUDA: {}\n", e.what());
    }
#endif

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

    // Warmup process using the extracted shape
    size_t totalElements = 1;
    for (int64_t dim : inputShape) {
        totalElements *= dim;
    }
    std::vector<float> dummyInput(totalElements, 0.0f);

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, dummyInput.data(), dummyInput.size(), inputShape.data(), inputShape.size());

    const char* inputNames[] = { inputName.c_str() };
    const char* warmupOutputNames[] = { outputNames[0].c_str() };

    auto startTime = std::chrono::high_resolution_clock::now();
    session->Run(Ort::RunOptions{ nullptr }, inputNames, &inputTensor, 1, warmupOutputNames, 1);
    auto endTime = std::chrono::high_resolution_clock::now();

    fmt::print("Classification Warmup completed in {} ms.\n",
        std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count());
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

    size_t inputTensorSize = 1 * modelChannels * modelHeight * modelWidth;
    std::vector<float> inputTensorValues(inputTensorSize);

    // Standard ImageNet normalization (Matching AnomalyEngine)
    const float mean[] = { 0.485f, 0.456f, 0.406f };
    const float stddev[] = { 0.229f, 0.224f, 0.225f };

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