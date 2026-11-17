#pragma once

#include "OrtEngine.h"
#include "Configuration.h"
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

/**
 * @brief Engine class for anomaly detection using ONNX Runtime.
 * Derives from OrtEngine (session + I/O buffers) and adds pre-processing,
 * heatmap generation and score normalization based on model-specific metadata.
 */
class AnomalyEngine : public OrtEngine {
public:
	// Structure grouping model calibration data
	struct Calibration
	{
		std::string normalizationFormula;

		float classificationMinRaw = 0.0f;
		float classificationMaxRaw = 1.0f;
		float classificationThresholdRaw = 0.5f;
		float classificationThresholdNorm = 0.5f;

		float segmentationMinRaw = 0.0f;
		float segmentationMaxRaw = 1.0f;
		float segmentationThresholdRaw = 0.5f;
		float segmentationThresholdNorm = 0.5f;
	};

	explicit AnomalyEngine(OrtRuntime& runtime);
	~AnomalyEngine() override;

	void Initialize(const std::filesystem::path& modelPath, const RT::CpuPartition& cpuPartition, ExecutionProvider executionProvider) override;

	void Infer(const void* pInputImage, uint32_t width, uint32_t height, uint32_t channels,
		void* pOutputHeatmap, float& outAnomalyScore, std::string& outStatus, DWORD idPunto) override;

	// Model-space thresholds (matching raw score and raw segmentation map units)
	float GetClassificationThreshold() const { return m_calibration.classificationThresholdRaw; }
	float GetSegmentationThreshold() const { return m_calibration.segmentationThresholdRaw; }

	// Normalized thresholds (0.0 to 1.0 range): compare against normalized score and map pixel / 255.0f
	float GetClassificationThresholdNormalized() const { return m_calibration.classificationThresholdNorm; }
	float GetSegmentationThresholdNormalized() const { return m_calibration.segmentationThresholdNorm; }

	const Calibration& GetCalibration() const { return m_calibration; }

	// Results from the most recent Infer() call (thread-local to the caller thread)
	float GetLastClassificationScoreRaw() const { return m_lastClassificationScoreRaw; }
	float GetLastClassificationScoreNormalized() const { return m_lastClassificationScoreNorm; }
	const std::vector<float>& GetLastSegmentationMapRaw() const { return m_hostSegmentationMap; }
	const cv::Mat& GetLastSegmentationMapNormalizedU8() const { return m_segmentationMapU8; }

private:
	void DescribeIo(IoBuffers& io) override;

	std::string m_inputName;
	std::vector<std::string> m_outputNames;

	std::vector<int64_t> m_inputShape;
	std::vector<int64_t> m_classificationScoreShape;
	std::vector<int64_t> m_segmentationMapShape;

	int64_t m_modelChannels = 3;
	int64_t m_modelHeight = 0;
	int64_t m_modelWidth = 0;

	// Standard RAM buffers for OpenCV processing and CPU fallback
	std::vector<float> m_inputTensorValues;
	std::vector<float> m_hostClassificationScore;
	std::vector<float> m_hostSegmentationMap;
	std::vector<cv::Mat> m_splitPlanes;
	cv::Mat m_resizeImage;

	// Antialiased bilinear resize (host-side)
	// Replaces cv::INTER_AREA, which does NOT match the training/reference resize
	// (torchvision/PIL antialiased bilinear) and inflated every anomaly score. This
	// is the SAME explicit separable triangle-filter algorithm as the Python
	// reference (inference_simulation/src/antialias_resize.py) — keep the two in
	// sync. Coefficients depend only on (src size -> model size), so they are built
	// once and reused while the frame size is constant.
	int m_rsInW = 0, m_rsInH = 0;               // Source size the coefficients were computed for
	int m_hK = 0, m_vK = 0;                     // Kernel taps per output pixel (horizontal / vertical axes)
	std::vector<int> m_hBounds, m_vBounds;      // First source index mapped per output pixel
	std::vector<double> m_hWeights, m_vWeights; // Normalized interpolation weights: [out_size * ksize]
	cv::Mat m_hpass;                            // Intermediate horizontal pass buffer: [inH x modelWidth] CV_8U
	void BuildResizeCoeffs(int inW, int inH);
	void ResizeAntialias(const cv::Mat& src, cv::Mat& dst);

	int m_classificationOutputIdx = -1;
	int m_segmentationOutputIdx = -1;

	// Segmentation map pipeline:
	// Raw model float values -> normalized uint8 (single-channel) -> resized to target ROI.
	// Output to caller is formatted as 3-channel BGR.
	bool m_writeSegmentationMap = true;
	cv::Mat m_segmentationMapU8;       // Normalized map at model resolution (CV_8UC1)
	cv::Mat m_segmentationMapRoiU8;    // Normalized map scaled to source ROI size (CV_8UC1)
	cv::Mat m_bgrImage;

	bool m_convertBgrToRgb = true;

	// Calibration parsed from ONNX model metadata (see Calibration struct in public section)
	Calibration m_calibration;
	bool m_centeredNormalization = false;   // true: anomalib_centered, false: minmax
	double m_segmentationMapAlpha = 0.0;    // Raw-to-uint8 conversion factor: pixel = alpha * raw + beta
	double m_segmentationMapBeta = 0.0;

	// Cached inference results from the last execution
	float m_lastClassificationScoreRaw = 0.0f;
	float m_lastClassificationScoreNorm = 0.0f;

	double m_accResize = 0, m_accNorm = 0, m_accRun = 0, m_accPost = 0;
	double m_maxRun = 0;
	int m_frameCount = 0;
	IniConfiguration& config;

	void LoadContractMetadata();
};