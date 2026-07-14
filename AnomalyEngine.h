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

	// I/O names extracted from the model graph at Initialize time,
	// so Infer never relies on hardcoded names
	std::string inputName;
	std::vector<std::string> outputNames;

	// Normalization stats read from the ONNX custom metadata at Initialize time.
	// They replicate Anomalib's min-max post-processing so the C++ scores match
	// the Python model bit-for-bit: norm = clamp((raw - min) / (max - min), 0, 1)
	float globalMin = 0.0f;
	float globalMax = 1.0f;
	float rawThreshold = 0.5f;        // threshold in raw-score space (from metadata)
	float normalizedThreshold = 0.5f; // rawThreshold mapped into [0,1] via globalMin/globalMax

	// Reads the normalization metadata (global_min, global_max, threshold)
	// embedded in the ONNX file and computes normalizedThreshold
	void LoadNormalizationMetadata();
};