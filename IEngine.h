#pragma once

#include <string>
#include <vector>
#include <cstdint>

class IEngine {
public:
	virtual ~IEngine() = default;

	// Method that intialize ONNX session with TensorRT, CUDA and CPU fallbacks
	virtual void Initialize(const std::wstring& modelPath) = 0;

	// Method that perform inference 
	virtual void Infer(const void* pInputImage,uint32_t width, uint32_t height, uint32_t channels, void* pOutputHeatmap, float& outAnomalyScore, std::string& outStatus) = 0;
};