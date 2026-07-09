#pragma once

#include "IEngine.h"
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <string>

class ClassificationEngine : public IEngine {
public:
    ClassificationEngine();
    ~ClassificationEngine() override = default;

    void Initialize(const std::wstring& modelPath) override;

    // The signature must match IEngine. 
    // For classification: outAnomalyScore -> confidence, outStatus -> class index/name
    void Infer(const void* pInputImage, uint32_t width, uint32_t height, uint32_t channels,
        void* pOutputHeatmap, float& outAnomalyScore, std::string& outStatus) override;

private:
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memoryInfo{ nullptr };
};