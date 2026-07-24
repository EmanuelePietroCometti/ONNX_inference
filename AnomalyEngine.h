#pragma once
#include "IEngine.h"
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

/**
 * @brief Engine class for anomaly detection using ONNX Runtime.
 * Implements the IEngine interface to perform inference, heatmap generation,
 * and score normalization based on model-specific metadata.
 */
class AnomalyEngine : public IEngine {
public:
    AnomalyEngine();
    ~AnomalyEngine() override;

    void Initialize(const std::wstring& modelPath, const RT::CpuPartition& cpuPartition) override;

    void Infer(const void* pInputImage, uint32_t width, uint32_t height, uint32_t channels,
        void* pOutputHeatmap, float& outAnomalyScore, std::string& outStatus) override;

private:
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memoryInfo{ nullptr };

    // --- I/O Binding Variables ---
    std::unique_ptr<Ort::IoBinding> io_binding;
    bool m_useGpuIoBinding = false; // Resolved at runtime

    Ort::Value m_inputTensor{ nullptr };
    Ort::Value m_scoreTensor{ nullptr };
    Ort::Value m_mapTensor{ nullptr };

    std::string inputName;
    std::vector<std::string> outputNames;

    std::vector<int64_t> inputShape;
    std::vector<int64_t> m_scoreShape;
    std::vector<int64_t> m_mapShape;

    int64_t modelChannels = 3;
    int64_t modelHeight = 0;
    int64_t modelWidth = 0;

    // Standard RAM buffers for OpenCV processing and CPU fallback
    std::vector<float> inputTensorValues;
    std::vector<float> m_hostScore;
    std::vector<float> m_hostMap;
    std::vector<cv::Mat> m_splitPlanes;
    cv::Mat m_resizeImage;

#if defined(ORT_EP_GPU)
    // Raw CUDA device pointers for Zero-Copy GPU execution
    float* d_input = nullptr;
    float* d_score = nullptr;
    float* d_map = nullptr;
#endif

    int scoreIdx = -1;
    int mapIdx = -1;

    bool writeHeatmap = true;
    cv::Mat m_uint8Heatmap;
    cv::Mat m_largeHeatmap;
    cv::Mat m_colorHeat;
    cv::Mat m_overlay;
    cv::Mat m_maskFull;

    // Contract 3.0 metadata
    std::string exportContract;
    bool normalizationInGraph = false;
    float mapMin = 0.0f;
    float mapMax = 1.0f;
    float scoreThreshold = 0.5f;
    float pixelThreshold = 0.0f;
    bool hasPixelThreshold = false;
    cv::Mat m_lutRgb;
    float m_blendAlpha = 0.5f;

    double accResize = 0, accNorm = 0, accRun = 0, accPost = 0;
    double maxRun = 0;
    int frameCount = 0;

    void LoadContractMetadata();
};