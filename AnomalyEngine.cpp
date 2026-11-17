#include "AnomalyEngine.h"
#include "IoBuffers.h"
#include "AsyncLogger.h"
#include <xmmintrin.h> 
#include <pmmintrin.h> 
#include <algorithm>
#include <cmath>
#include <cstring>
#include <chrono>
#include <string>
#include <thread>
#include <unordered_map>
#include <array>
#include <stdexcept>

namespace {
	// Mirrors anomaly_export/postprocess.py::normalize()
	//   minmax:             y = clamp((x - min) / (max - min), 0, 1)
	//   anomalib_centered:  y = clamp((x - threshold) / (max - min) + 0.5, 0, 1)
	inline float NormalizeToUnitRange(double raw, double threshold, double minRaw, double maxRaw, bool centered)
	{
		const double span = maxRaw - minRaw;
		const double y = centered ? (raw - threshold) / span + 0.5 : (raw - minRaw) / span;
		return static_cast<float>(std::clamp(y, 0.0, 1.0));
	}
}

AnomalyEngine::AnomalyEngine(OrtRuntime& runtime)
	: OrtEngine(runtime, "Anomaly"), config(IniConfiguration::Instance())
{
}

AnomalyEngine::~AnomalyEngine()
{
}

void AnomalyEngine::DescribeIo(IoBuffers& io)
{
	io.AddInput(m_inputName, m_inputTensorValues.data(), m_inputTensorValues.size() * sizeof(float),
		m_inputShape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

	io.AddOutput(m_outputNames[m_classificationOutputIdx], m_hostClassificationScore.data(), sizeof(float),
		m_classificationScoreShape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

	if (m_segmentationOutputIdx >= 0) {
		io.AddOutput(m_outputNames[m_segmentationOutputIdx], m_hostSegmentationMap.data(),
			m_hostSegmentationMap.size() * sizeof(float), m_segmentationMapShape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
	}
}

void AnomalyEngine::Initialize(const std::filesystem::path& modelPath, const RT::CpuPartition& cpuPartition, ExecutionProvider executionProvider)
{
	try {
		// Provider selection + session (OrtEngine)
		CreateSession(modelPath, cpuPartition, executionProvider);
		Log::Info("ONNX Session created successfully for Anomaly Detection.");

		Ort::AllocatorWithDefaultOptions allocator;
		m_inputName = session->GetInputNameAllocated(0, allocator).get();

		m_outputNames.clear();
		for (size_t idx = 0; idx < session->GetOutputCount(); ++idx) {
			m_outputNames.emplace_back(session->GetOutputNameAllocated(idx, allocator).get());
		}

		// Extrapolate output indices BEFORE allocation
		for (size_t idx = 0; idx < session->GetOutputCount(); ++idx) {
			auto shape = session->GetOutputTypeInfo(idx).GetTensorTypeAndShapeInfo().GetShape();
			if (shape.size() == 1 || (shape.size() == 2 && shape[1] == 1)) {
				if (m_classificationOutputIdx < 0) m_classificationOutputIdx = static_cast<int>(idx);
			}
			else if (shape.size() >= 3) {
				if (m_segmentationOutputIdx < 0) m_segmentationOutputIdx = static_cast<int>(idx);
			}
		}
		if (m_classificationOutputIdx < 0) throw std::runtime_error("Anomaly model missing scalar score output.");

		// Geometry extrapolation & sanitization
		Ort::TypeInfo typeInfo = session->GetInputTypeInfo(0);
		m_inputShape = typeInfo.GetTensorTypeAndShapeInfo().GetShape();
		if (m_inputShape[0] == -1) m_inputShape[0] = 1;

		m_modelChannels = m_inputShape[1];
		m_modelHeight = m_inputShape[2];
		m_modelWidth = m_inputShape[3];

		// Contract: version, calibration, preprocessing (interpolation, colour order). Throws when the
		// model is not one this engine can run exactly as it was trained.
		const Contract::AnomalyContract contract = Contract::LoadAnomalyContract(
			Contract::MetadataReader::FromSession(*session), m_modelWidth, m_modelHeight);
		m_calibration = contract.calibration;
		m_centeredNormalization = contract.centeredNormalization;
		m_segmentationMapAlpha = contract.segmentationMapAlpha;
		m_segmentationMapBeta = contract.segmentationMapBeta;
		m_resizeInterpolation = contract.preprocessing.cvInterpolation;
		m_convertBgrToRgb = contract.preprocessing.bgrToRgb;

		m_classificationScoreShape = session->GetOutputTypeInfo(m_classificationOutputIdx).GetTensorTypeAndShapeInfo().GetShape();
		for (auto& d : m_classificationScoreShape) if (d < 0) d = 1;

		if (m_segmentationOutputIdx >= 0) {
			m_segmentationMapShape = session->GetOutputTypeInfo(m_segmentationOutputIdx).GetTensorTypeAndShapeInfo().GetShape();
			for (auto& d : m_segmentationMapShape) if (d < 0) d = 1;
		}

		// Calculate buffer sizes
		size_t inputElements = 1;
		for (int64_t d : m_inputShape) inputElements *= d;

		size_t segmentationMapElements = 1;
		if (m_segmentationOutputIdx >= 0) {
			for (int64_t d : m_segmentationMapShape) segmentationMapElements *= d;
		}

		// Host allocations (required for pre-processing / fallback)
		m_inputTensorValues.assign(inputElements, 0.0f);
		m_splitPlanes.resize(m_modelChannels);
		m_hostClassificationScore.assign(1, 0.0f);
		if (m_segmentationOutputIdx >= 0) m_hostSegmentationMap.assign(segmentationMapElements, 0.0f);

		// I/O binding: the tensors are the ones listed in DescribeIo (OrtEngine picks where they live)
		CreateIoBuffers();

		// Warmup runs
		double firstMs = 0.0, lastMs = 0.0;
		for (int r = 0; r < config.GetONNXWarmUps(); ++r) {
			auto startTime = std::chrono::steady_clock::now();
			Io().Run();
			lastMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - startTime).count();
			if (r == 0) firstMs = lastMs;
		}

		Log::Info("Warmup completed: {} runs, first {:.1f} ms, last {:.1f} ms. Engine locked.", config.GetONNXWarmUps(), firstMs, lastMs);
	}
	catch (const std::exception& e) {   // Ort::Exception derives from it
		Log::Error("### ERROR DURING INITIALIZATION: {}", e.what());
		throw std::runtime_error(std::string("Model initialization failed (TensorRT/CUDA): ") + e.what());
	}
	catch (...)
	{
		Log::Error("### ERROR DURING INITIALIZATION: unknown error");
		throw std::runtime_error(std::string("Model initialization failed (TensorRT/CUDA): unknown error"));
	}
}

void AnomalyEngine::Infer(const void* pInputImage, uint32_t width, uint32_t height,
	uint32_t channels, void* pOutputHeatmap, float& outAnomalyScore, std::string& outStatus, DWORD idPunto)
{
	int type = 0;
	switch (channels)
	{
	case 1:
		type = CV_8UC1;
		break;
	case 3:
		type = CV_8UC3;
		break;
	default:
		throw std::invalid_argument("OpenCV ERROR: Unsupported number of channels (" + std::to_string(channels) + "). Expected 1 or 3.");
	}
	cv::Mat rawImg(height, width, type, const_cast<void*>(pInputImage));
	// Resize with the interpolation declared by the contract (preproc_resize_interpolation).
	// The exporter only accepts a plain cv::resize (stretch, no crop, no antialias), so this is the
	// same operation the model saw in training.
	const cv::Mat* pSrc = &rawImg;
	if (channels == 1) {
		cv::cvtColor(rawImg, m_bgrImage, cv::COLOR_GRAY2BGR);
		pSrc = &m_bgrImage;
	}
	cv::resize(*pSrc, m_resizeImage, cv::Size(static_cast<int>(m_modelWidth), static_cast<int>(m_modelHeight)),
		0, 0, m_resizeInterpolation);

	// The model expects RGB (ImageNet order); OpenCV holds BGR. Swap in place so
	// cv::split below yields planes [R, G, B] aligned with the graph and mean/std.
	if (m_convertBgrToRgb && m_modelChannels == 3) {
		cv::cvtColor(m_resizeImage, m_resizeImage, cv::COLOR_BGR2RGB);
	}

	// normalize_inside_graph is 'true' by contract: the graph subtracts mean and divides by std,
	// host only converts uint8 [0, 255] to float [0, 1] (preproc_scale = 1/255)
	const float scale = 1.0f / 255.0f;
	const size_t planeSize = static_cast<size_t>(m_modelHeight) * static_cast<size_t>(m_modelWidth);

	cv::split(m_resizeImage, m_splitPlanes);
	for (int c = 0; c < m_modelChannels; ++c) {
		// Writes directly into the host buffer
		cv::Mat planeF32(static_cast<int>(m_modelHeight), static_cast<int>(m_modelWidth), CV_32FC1, m_inputTensorValues.data() + static_cast<size_t>(c) * planeSize);
		m_splitPlanes[c].convertTo(planeF32, CV_32FC1, scale);
	}

	// Host -> device, inference, device -> host. On CPU/OpenVINO the two copies do nothing.
	Io().Upload();
	Io().Run();
	Io().Download();

	// Classification. Decision is made in raw units: this matches the Python logic
		// (normalized score >= normalized threshold) because normalization is strictly monotonic
		// (they only diverge if a min-max threshold falls outside [score_min, score_max]).
	const float rawScore = m_hostClassificationScore[0];
	outAnomalyScore = rawScore;
	m_lastClassificationScoreRaw = rawScore;
	m_lastClassificationScoreNorm = NormalizeToUnitRange(rawScore,
		m_calibration.classificationThresholdRaw,
		m_calibration.classificationMinRaw, m_calibration.classificationMaxRaw,
		m_centeredNormalization);
	outStatus = (rawScore >= m_calibration.classificationThresholdRaw) ? "REJECT" : "OK";

	// Segmentation: normalized map, 24-bit output
	if (m_writeSegmentationMap && pOutputHeatmap && m_segmentationOutputIdx >= 0) {
		// Convert raw float map -> uint8 using the postprocess.normalize() formula
		cv::Mat rawSegmentationMap(static_cast<int>(m_segmentationMapShape[2]), static_cast<int>(m_segmentationMapShape[3]), CV_32FC1, m_hostSegmentationMap.data());
		rawSegmentationMap.convertTo(m_segmentationMapU8, CV_8UC1, m_segmentationMapAlpha, m_segmentationMapBeta);

		// Resize to match the ROI dimensions
		cv::resize(m_segmentationMapU8, m_segmentationMapRoiU8, cv::Size(static_cast<int>(width), static_cast<int>(height)), 0, 0, cv::INTER_LINEAR);

		// 24-bit (BGR, three identical channels) written directly into the shared buffer:
		// Dimensions and type match, preventing cvtColor from reallocating
		cv::Mat segmentationMapBgr(static_cast<int>(height), static_cast<int>(width), CV_8UC3, pOutputHeatmap);
		cv::cvtColor(m_segmentationMapRoiU8, segmentationMapBgr, cv::COLOR_GRAY2BGR);
	}
}