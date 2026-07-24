#pragma once

#include "IEngine.h"
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <string>
#include <opencv2/opencv.hpp>

class ClassificationEngine : public IEngine {
public:
    ClassificationEngine();
    ~ClassificationEngine() override;

    void Initialize(const std::wstring& modelPath, const RT::CpuPartition& cpuPartition) override;
    void Infer(const void* pInputImage, uint32_t width, uint32_t height, uint32_t channels,
        void* pOutputHeatmap, float& outAnomalyScore, std::string& outStatus) override;

private:
    void LoadGraphContract();

    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memoryInfo{ nullptr };

    std::unique_ptr<Ort::IoBinding> io_binding;
    bool m_useGpuIoBinding = false;

    std::string inputName;
    std::vector<std::string> outputNames;
    std::vector<int64_t> inputShape;
    std::vector<int64_t> m_classIdShape;
    std::vector<int64_t> m_confShape;

    int64_t modelChannels = 3;
    int64_t modelHeight = 0;
    int64_t modelWidth = 0;

    std::vector<std::string> m_classNames;
    std::string m_outClassIdName;
    std::string m_outConfidenceName;

    Ort::Value m_inputTensor{ nullptr };
    Ort::Value m_classIdTensor{ nullptr };
    Ort::Value m_confidenceTensor{ nullptr };

    std::vector<float> inputTensorValues;
    std::vector<int64_t> m_hostClassId;
    std::vector<float> m_hostConfidence;

#if defined(ORT_EP_GPU)
    float* d_input = nullptr;
    int64_t* d_classId = nullptr;
    float* d_confidence = nullptr;
#endif

    double accResize = 0, accNorm = 0, accRun = 0, accPost = 0;
    double maxRun = 0;
    int frameCount = 0;

    std::vector<cv::Mat> m_splitPlanes;
    cv::Mat m_resizeImage;
    cv::Mat m_rgbImage;
};