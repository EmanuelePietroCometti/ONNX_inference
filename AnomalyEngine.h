#pragma once


#include "IEngine.h"
#include <onnxruntime_cxx_api.h>
#include <memory>

class AnomalyEngine : public IEngine {
public:
	AnomalyEngine();
	~AnomalyEngine() override = default;

	void Initialize(const std::wstring& modelPath) override;
	void Infer(const void* pInputImage, uint32_t width, uint32_t height, uint32_t channels, void* pOutputHeatmap, float& outAnomalyScore, std::string& outStatus);

private:
	std::unique_ptr<Ort::Env> env;
	std::unique_ptr<Ort::Session> session;
	Ort::MemoryInfo memoryInfo{ nullptr };
};