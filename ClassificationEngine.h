#pragma once

#include "OrtEngine.h"
#include "Configuration.h"
#include "ContractMetadata.h"
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

/**
 * @brief Engine class for image classification using ONNX Runtime.
 * Derives from OrtEngine (session + I/O buffers) and adds pre-processing
 * and the class_id -> label mapping read from the model metadata.
 */
class ClassificationEngine : public OrtEngine {
public:
	explicit ClassificationEngine(OrtRuntime& runtime);
	~ClassificationEngine() override;

	void Initialize(const std::filesystem::path& modelPath, const RT::CpuPartition& cpuPartition, ExecutionProvider executionProvider) override;
	void Infer(const void* pInputImage, uint32_t width, uint32_t height, uint32_t channels,
		void* pOutputHeatmap, float& outAnomalyScore, std::string& outStatus, DWORD idPunto) override;
	std::string GetNamesJSONString() const { return m_classNamesJSONString; }
private:
	void DescribeIo(IoBuffers& io) override;
	void LoadGraphContract();

	std::string inputName;
	std::vector<std::string> outputNames;
	std::vector<int64_t> inputShape;
	std::vector<int64_t> m_classIdShape;
	std::vector<int64_t> m_confShape;

	int64_t modelChannels = 3;
	int64_t modelHeight = 0;
	int64_t modelWidth = 0;

	std::vector<std::string> m_classNames;
	std::string m_classNamesJSONString;
	std::string m_outClassIdName;
	std::string m_outConfidenceName;

	// Resize flag for cv::resize, taken from the model contract (preproc_resize_interpolation)
	int m_resizeInterpolation = cv::INTER_LINEAR;

	// Host buffers: IoBuffers binds them directly (CPU/OpenVINO) or copies them to/from the device
	std::vector<float> inputTensorValues;
	std::vector<int64_t> m_hostClassId;
	std::vector<float> m_hostConfidence;

	double accResize = 0, accNorm = 0, accRun = 0, accPost = 0;
	double maxRun = 0;
	int frameCount = 0;

	std::vector<cv::Mat> m_splitPlanes;
	cv::Mat m_resizeImage;
	cv::Mat m_rgbImage;
	cv::Mat m_bgrImage;

	IniConfiguration& config;
};