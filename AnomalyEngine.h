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
	~AnomalyEngine() override = default;

	/**
	 * @brief Loads the ONNX model, extracts input/output metadata, and initializes the session.
	 * @param modelPath Path to the .onnx model file.
	 */
	void Initialize(const std::wstring& modelPath) override;

	/**
	 * @brief Performs inference on a provided raw image buffer.
	 * @param pInputImage Pointer to the raw input image data.
	 * @param width Image width.
	 * @param height Image height.
	 * @param channels Number of color channels.
	 * @param pOutputHeatmap Pointer to pre-allocated buffer for the output heatmap.
	 * @param outAnomalyScore Reference to store the normalized anomaly score.
	 * @param outStatus Reference to store status messages.
	 */
	void Infer(const void* pInputImage, uint32_t width, uint32_t height, uint32_t channels,
		void* pOutputHeatmap, float& outAnomalyScore, std::string& outStatus) override;

private:
	// ONNX Runtime session environment and handle
	std::unique_ptr<Ort::Env> env;
	std::unique_ptr<Ort::Session> session;
	Ort::MemoryInfo memoryInfo{ nullptr };

	// Input node name
	std::string inputName;

	// Storage for output node names to ensure pointer validity for ORT API calls
	std::vector<std::string> outputNames;
	// Pointers to the strings above, stored for performance in inference loops
	std::vector<const char*> outputNamesC;

	// Model input dimensions and shape configuration
	std::vector<int64_t> inputShape;
	int64_t modelChannels = 3;
	int64_t modelHeight = 0;
	int64_t modelWidth = 0;

	// Pre-allocated buffers to minimize heap allocations during high-frequency inference
	std::vector<float> inputTensorValues;
	std::vector<cv::Mat> m_splitPlanes;
	cv::Mat m_resizeImage;

	// Cache indices for score and map nodes after initial graph inspection
	int scoreIdx = -1;
	int mapIdx = -1;

	// Heatmap processing buffers
	bool writeHeatmap = true;
	cv::Mat m_uint8Heatmap, m_largeHeatmap, m_rgbHeatmap;

	// Normalization statistics parsed from model custom metadata.
	// Used to perform post-processing: normalized = clamp((raw - min) / (max - min), 0, 1)
	float globalMin = 0.0f;
	float globalMax = 1.0f;
	float rawThreshold = 0.5f;         // Threshold in raw score space
	float normalizedThreshold = 0.5f;  // Threshold mapped to [0, 1] range

	double accResize = 0, accNorm = 0, accRun = 0, accPost = 0;
	double maxRun = 0;
	int frameCount = 0;

	/**
	 * @brief Extracts normalization metadata (min, max, threshold) from the ONNX file
	 * and calculates the normalized threshold for classification decisions.
	 */
	void LoadNormalizationMetadata();
};