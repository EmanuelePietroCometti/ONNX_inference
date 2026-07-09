#include "ClassificationEngine.h"
#include <fmt/core.h>
#include <cstring>
#include <chrono>
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

    // Setup hardware acceleration (TensorRT -> CUDA -> CPU)
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

    session = std::make_unique<Ort::Session>(*env, modelPath.c_str(), sessionOptions);

    // Warmup process (Fixed shapes for 256x256 based on the model properties)
    std::vector<int64_t> inputShape = { 1, 3, 256, 256 };
    size_t totalElements = 1 * 3 * 256 * 256;
    std::vector<float> dummyInput(totalElements, 0.0f);

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, dummyInput.data(), dummyInput.size(), inputShape.data(), inputShape.size());

    const char* inputNames[] = { "input_tensor" };
    const char* outputNames[] = { "class_probabilities" };

    auto startTime = std::chrono::high_resolution_clock::now();
    session->Run(Ort::RunOptions{ nullptr }, inputNames, &inputTensor, 1, outputNames, 1);
    auto endTime = std::chrono::high_resolution_clock::now();

    fmt::print("Classification Warmup completed in {} ms.\n",
        std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count());
}

void ClassificationEngine::Infer(const void* pInputImage, uint32_t width, uint32_t height, uint32_t channels,
    void* pOutputHeatmap, float& outAnomalyScore, std::string& outStatus)
{
    // Shape is statically known from the graph properties
    std::vector<int64_t> inputShape = { 1, 3, 256, 256 };
    const int64_t modelHeight = 256;
    const int64_t modelWidth = 256;
    const int64_t modelChannels = 3;

    cv::Mat rawImg(height, width, CV_8UC3, const_cast<void*>(pInputImage));
    cv::Mat resizeImage;
    cv::resize(rawImg, resizeImage, cv::Size(modelWidth, modelHeight), 0, 0, cv::INTER_LINEAR);

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

    // Explicitly define input/output names based on the graph image provided
    const char* inputNames[] = { "input_tensor" };
    const char* outputNames[] = { "class_probabilities" };

    auto outputTensors = session->Run(
        Ort::RunOptions{ nullptr }, inputNames, &inputTensor, 1, outputNames, 1);

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