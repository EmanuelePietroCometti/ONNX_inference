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
	 * @param cpuPartition Core slice owned by this session (sizes/pins the intra-op pool).
	 */
	void Initialize(const std::wstring& modelPath, const RT::CpuPartition& cpuPartition) override;

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

	// Ort::Value wrapping inputTensorValues, created ONCE at Initialize and
	// reused by every Run (zero-copy policy: the buffer is refilled in place)
	Ort::Value m_inputTensor{ nullptr };

	// Cache indices for score and map nodes after initial graph inspection
	int scoreIdx = -1;
	int mapIdx = -1;

	// Heatmap / overlay processing buffers (reused across frames)
	bool writeHeatmap = true;
	cv::Mat m_uint8Heatmap;   // display-normalized map, uint8 [0,255] at model res
	cv::Mat m_largeHeatmap;   // above, resized to original resolution
	cv::Mat m_colorHeat;      // LUT-colored heatmap, RGB
	cv::Mat m_overlay;        // final blended overlay, RGB (written to output buffer)
	cv::Mat m_maskFull;       // pred_mask resized to original resolution

	// --- Contract / calibration metadata (contract 3.0) ---
	// Read once at Initialize from the ONNX custom metadata. Nothing here is
	// computed at runtime: no hardcoded thresholds, no per-frame statistics.
	std::string exportContract;          // must equal "3.0"
	bool normalizationInGraph = false;   // normalization == "in_graph": skip host norm

	// Display-normalization range for the anomaly map (calibration_map_min/max):
	// v = clamp((map - mapMin) / (mapMax - mapMin), 0, 1). See render_reference.py [D-Q1].
	float mapMin = 0.0f;
	float mapMax = 1.0f;

	// Image-level decision threshold, in the FINAL score space of the graph
	// (calibrated_threshold). Compared directly to the raw score (contract 3.0:
	// score is final, CONTRACT.md §2.1).
	float scoreThreshold = 0.5f;

	// Pixel-level threshold on the final map for the pred_mask overlay
	// (calibrated_threshold_pixel). Optional: not all architectures export it.
	float pixelThreshold = 0.0f;
	bool hasPixelThreshold = false;

	// Shared jet colormap (render_lut_rgb_b64): 256x1 CV_8UC3, RGB, decoded from
	// the base64 metadata. Consumed byte-identical to render_reference.py [D-Q3].
	cv::Mat m_lutRgb;

	// Alpha blending constant, out = (1-a)*img + a*heatmap. render_reference.py [D-Q5].
	float m_blendAlpha = 0.5f;

	double accResize = 0, accNorm = 0, accRun = 0, accPost = 0;
	double maxRun = 0;
	int frameCount = 0;

	/**
	 * @brief Loads and validates the contract 3.0 metadata from the ONNX file:
	 * rejects models that are not contract 3.0 or not verified, reads the
	 * calibration thresholds, the display range and the shared colormap LUT.
	 * Throws std::runtime_error (pointing at calibrate_threshold.py) on failure.
	 */
	void LoadContractMetadata();
};