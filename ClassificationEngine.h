#pragma once

#include "IEngine.h"
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <string>
#include <opencv2/opencv.hpp>

class ClassificationEngine : public IEngine {
public:
    ClassificationEngine();
    ~ClassificationEngine() override = default;

    void Initialize(const std::wstring& modelPath, const RT::CpuPartition& cpuPartition) override;

    // The signature must match IEngine. 
    // For classification: outAnomalyScore -> confidence, outStatus -> class index/name
    void Infer(const void* pInputImage, uint32_t width, uint32_t height, uint32_t channels,
        void* pOutputHeatmap, float& outAnomalyScore, std::string& outStatus) override;

private:
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memoryInfo{ nullptr };

    // I/O names and input shape extracted from the model graph at Initialize time,
    // so Infer never relies on hardcoded names or dimensions
    std::string inputName;
    std::vector<std::string> outputNames;
    std::vector<int64_t> inputShape;
    int64_t modelChannels = 3;
    int64_t modelHeight = 0;
    int64_t modelWidth = 0;

    // Input tensor buffer allocated once at Initialize and reused on every Infer,
    // avoiding a per-frame heap allocation of the whole CHW tensor
    std::vector<float> inputTensorValues;

    // Ort::Value wrapping inputTensorValues, created ONCE at Initialize and
    // reused by every Run: the tensor is just a view over the same buffer, so
    // recreating it per frame is pure overhead (zero-copy policy)
    Ort::Value m_inputTensor{ nullptr };

    double accResize = 0, accNorm = 0, accRun = 0, accPost = 0;
    double maxRun = 0;
    int frameCount = 0;

    // Pre-allocated OpenCV working buffers reused across frames
    std::vector<cv::Mat> m_splitPlanes;
    cv::Mat m_resizeImage;
};