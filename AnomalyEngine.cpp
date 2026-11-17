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
	// Major version of the contract supported by this engine (exporter: CONTRACT_VERSION)
	constexpr int kSupportedContractMajor = 1;

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

		LoadContractMetadata();

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

void AnomalyEngine::LoadContractMetadata()
{
	Ort::AllocatorWithDefaultOptions allocator;
	Ort::ModelMetadata metadata = session->GetModelMetadata();

	auto lookupString = [&](const char* key) -> std::string {
		auto value = metadata.LookupCustomMetadataMapAllocated(key, allocator);
		return value ? std::string(value.get()) : std::string();
		};
	auto requireFloat = [&](const char* key) -> float {
		const std::string text = lookupString(key);
		if (text.empty()) {
			throw std::runtime_error(std::string("Anomaly model rejected: calibration metadata '") + key +
				"' is missing. Re-run calibrate.py.");
		}
		const float parsed = std::stof(text);
		Log::Info("Metadata '{}' = {}", key, parsed);
		return parsed;
		};

	// Verify contract version presence
	const std::string contractVersion = lookupString("contract_version");
	if (contractVersion.empty()) {
		throw std::runtime_error("Anomaly model rejected: metadata 'contract_version' is missing.");
	}
	int contractMajor = -1;
	try {
		contractMajor = std::stoi(contractVersion.substr(0, contractVersion.find('.')));
	}
	catch (const std::exception&) {
		Log::Error("### ERROR: failed to parse contract_version '{}'", contractVersion);
		contractMajor = -1;
	}
	catch (...)
	{
		Log::Error("### ERROR: unknown error while parsing contract_version '{}'", contractVersion);
		contractMajor = -1;
	}
	if (contractMajor != kSupportedContractMajor) {
		throw std::runtime_error("Anomaly model rejected: contract_version '" + contractVersion +
			"' is not supported (expected major " + std::to_string(kSupportedContractMajor) + ").");
	}
	Log::Info("ONNX Contract Version validated: {}", contractVersion);

	const std::string calibratedStatus = lookupString("calibrated");
	if (calibratedStatus != "true")
	{
		throw std::runtime_error("Anomaly model rejected: metadata 'calibrated' is '" + calibratedStatus +
			"' (expected 'true'). Run calibrate.py on the model.");
	}
	// Normalization check: updated key to match new Python exporter
	const std::string normalization = lookupString("normalize_inside_graph");
	if (normalization != "true") {
		throw std::runtime_error("Anomaly model rejected: normalize_inside_graph='" + normalization +
			"' (the contract requires 'true').");
	}

	// Channel order: the graph is trained on RGB (ImageNet). OpenCV decodes BMPs
	// as BGR, so honor preproc_color_conversion and swap channels before feeding
	// the model. Missing key defaults to bgr2rgb (all our exporter's models).
	const std::string colorConversion = lookupString("preproc_color_conversion");
	m_convertBgrToRgb = colorConversion.empty() || colorConversion == "bgr2rgb";
	Log::Info("Metadata 'preproc_color_conversion' = '{}' -> BGR2RGB swap {}.", colorConversion.empty() ? "(missing, default)" : colorConversion, m_convertBgrToRgb ? "ENABLED" : "disabled");

	m_calibration.classificationThresholdRaw = requireFloat("image_threshold_raw");
	m_calibration.segmentationThresholdRaw = requireFloat("pixel_threshold_raw");
	m_calibration.classificationMinRaw = requireFloat("score_min_raw");
	m_calibration.classificationMaxRaw = requireFloat("score_max_raw");
	m_calibration.segmentationMinRaw = requireFloat("map_min_raw");
	m_calibration.segmentationMaxRaw = requireFloat("map_max_raw");

	const double classificationSpan = static_cast<double>(m_calibration.classificationMaxRaw) - m_calibration.classificationMinRaw;
	const double segmentationSpan = static_cast<double>(m_calibration.segmentationMaxRaw) - m_calibration.segmentationMinRaw;
	if (classificationSpan <= 0.0) {
		throw std::runtime_error("Invalid calibration metadata: score_max_raw must be greater than score_min_raw.");
	}
	if (segmentationSpan <= 0.0) {
		throw std::runtime_error("Invalid calibration metadata: map_max_raw must be greater than map_min_raw.");
	}


	const std::string normFormula = lookupString("normalization_formula");
	if (normFormula == "anomalib_centered") {
		m_centeredNormalization = true;
	}
	else if (normFormula == "minmax") {
		m_centeredNormalization = false;
	}
	else {
		throw std::runtime_error("Anomaly model rejected: normalization_formula '" + normFormula +
			"' is missing or not supported (expected 'minmax' or 'anomalib_centered').");
	}
	m_calibration.normalizationFormula = normFormula;

	// Normalized thresholds, following the same rule as postprocess.normalized_threshold():
	//	- anomalib_centered always maps the threshold to 0.5.
	//	- minmax maps the threshold linearly between minRaw and maxRaw.
	auto thresholdNorm = [&](double threshold, double minRaw, double maxRaw) -> float {
		if (m_centeredNormalization) {
			return 0.5f;
		}
		const double span = maxRaw - minRaw;
		if (std::abs(span) < 1e-9) {
			return 0.0f;
		}
		return static_cast<float>(std::clamp((threshold - minRaw) / span, 0.0, 1.0));
		};

	m_calibration.classificationThresholdNorm = thresholdNorm(
		m_calibration.classificationThresholdRaw,
		m_calibration.classificationMinRaw,
		m_calibration.classificationMaxRaw
	);

	m_calibration.segmentationThresholdNorm = thresholdNorm(
		m_calibration.segmentationThresholdRaw,
		m_calibration.segmentationMinRaw,
		m_calibration.segmentationMaxRaw
	);

	// Cross-check against the *_threshold_norm metadata values written by the exporter:
	// if they are stale or drifted, the derived value takes precedence.
	auto warnIfStale = [&](const char* key, float derived) {
		const std::string stored = lookupString(key);
		if (stored.empty()) {
			return;
		}
		const float storedValue = std::stof(stored);
		if (std::fabs(storedValue - derived) > 1e-4f) {
			Log::Warning("Metadata '{}' = {} differs from the value derived from the raw calibration ({}); using the derived value.",
				key, storedValue, derived);
		}
		};

	warnIfStale("image_threshold_norm", m_calibration.classificationThresholdNorm);
	warnIfStale("pixel_threshold_norm", m_calibration.segmentationThresholdNorm);

	m_segmentationMapAlpha = 255.0 / segmentationSpan;
	m_segmentationMapBeta = m_centeredNormalization
		? 255.0 * (0.5 - static_cast<double>(m_calibration.segmentationThresholdRaw) / segmentationSpan)
		: -255.0 * static_cast<double>(m_calibration.segmentationMinRaw) / segmentationSpan;

	Log::Info("Contract limits verified (formula '{}'). classification: min={}, max={}, threshold={} (norm {}); "
		"segmentation: min={}, max={}, threshold={} (norm {}); map->uint8 alpha={:.5f}, beta={:.5f}.",
		m_calibration.normalizationFormula,
		m_calibration.classificationMinRaw, m_calibration.classificationMaxRaw,
		m_calibration.classificationThresholdRaw, m_calibration.classificationThresholdNorm,
		m_calibration.segmentationMinRaw, m_calibration.segmentationMaxRaw,
		m_calibration.segmentationThresholdRaw, m_calibration.segmentationThresholdNorm,
		m_segmentationMapAlpha, m_segmentationMapBeta);
}

namespace {
	// Triangle-filter (bilinear) resample coefficients for ONE axis. Mirrors
	// inference_simulation/src/antialias_resize.py::precompute_coeffs bit-for-bit:
	// on a downscale the filter support widens by the scale factor (antialiasing).
	void PrecomputeCoeffs(int inSize, int outSize,
		std::vector<int>& bounds, std::vector<double>& weights, int& ksize)
	{
		const double scale = static_cast<double>(inSize) / static_cast<double>(outSize);
		const double filterscale = scale >= 1.0 ? scale : 1.0;   // widen on downscale
		const double support = 1.0 * filterscale;                // bilinear half-width
		const double ss = 1.0 / filterscale;

		ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
		bounds.assign(outSize, 0);
		weights.assign(static_cast<size_t>(outSize) * ksize, 0.0);

		for (int o = 0; o < outSize; ++o) {
			const double center = (o + 0.5) * scale;
			int xmin = static_cast<int>(center - support + 0.5);
			if (xmin < 0) xmin = 0;
			int xmax = static_cast<int>(center + support + 0.5);
			if (xmax > inSize) xmax = inSize;
			const int n = xmax - xmin;
			double total = 0.0;
			for (int t = 0; t < n; ++t) {
				double w = 1.0 - std::abs((xmin + t - center + 0.5) * ss);  // bilinear_filter
				if (w < 0.0) w = 0.0;
				weights[static_cast<size_t>(o) * ksize + t] = w;
				total += w;
			}
			if (total > 0.0)
				for (int t = 0; t < n; ++t) weights[static_cast<size_t>(o) * ksize + t] /= total;
			bounds[o] = xmin;
		}
	}

	inline uint8_t RoundClipU8(double v) {
		double r = std::floor(v + 0.5);   // round half up (pixels are non-negative)
		if (r < 0.0) r = 0.0;
		if (r > 255.0) r = 255.0;
		return static_cast<uint8_t>(r);
	}
}

void AnomalyEngine::BuildResizeCoeffs(int inW, int inH)
{
	PrecomputeCoeffs(inW, static_cast<int>(m_modelWidth), m_hBounds, m_hWeights, m_hK);
	PrecomputeCoeffs(inH, static_cast<int>(m_modelHeight), m_vBounds, m_vWeights, m_vK);
	m_rsInW = inW;
	m_rsInH = inH;
}

// Antialiased bilinear resize of a CV_8UC3 image to (modelWidth x modelHeight).
// Separable: horizontal pass -> uint8 round/clip -> vertical pass -> uint8. Double
// accumulation to match the Python reference exactly.
void AnomalyEngine::ResizeAntialias(const cv::Mat& src, cv::Mat& dst)
{
	const int inH = src.rows, inW = src.cols;
	const int outW = static_cast<int>(m_modelWidth), outH = static_cast<int>(m_modelHeight);
	if (inW != m_rsInW || inH != m_rsInH) BuildResizeCoeffs(inW, inH);

	// Horizontal: [inH x inW] -> [inH x outW]
	m_hpass.create(inH, outW, CV_8UC3);
	for (int y = 0; y < inH; ++y) {
		const uint8_t* srow = src.ptr<uint8_t>(y);
		uint8_t* hrow = m_hpass.ptr<uint8_t>(y);
		for (int o = 0; o < outW; ++o) {
			const int s = m_hBounds[o];
			const int avail = std::min(m_hK, inW - s);
			const double* w = &m_hWeights[static_cast<size_t>(o) * m_hK];
			double a0 = 0.0, a1 = 0.0, a2 = 0.0;
			for (int t = 0; t < avail; ++t) {
				const uint8_t* px = srow + static_cast<size_t>(s + t) * 3;
				const double wt = w[t];
				a0 += wt * px[0]; a1 += wt * px[1]; a2 += wt * px[2];
			}
			uint8_t* op = hrow + static_cast<size_t>(o) * 3;
			op[0] = RoundClipU8(a0); op[1] = RoundClipU8(a1); op[2] = RoundClipU8(a2);
		}
	}

	// Vertical: [inH x outW] -> [outH x outW]
	dst.create(outH, outW, CV_8UC3);
	for (int o = 0; o < outH; ++o) {
		const int s = m_vBounds[o];
		const int avail = std::min(m_vK, inH - s);
		const double* w = &m_vWeights[static_cast<size_t>(o) * m_vK];
		uint8_t* drow = dst.ptr<uint8_t>(o);
		for (int x = 0; x < outW; ++x) {
			double a0 = 0.0, a1 = 0.0, a2 = 0.0;
			for (int t = 0; t < avail; ++t) {
				const uint8_t* px = m_hpass.ptr<uint8_t>(s + t) + static_cast<size_t>(x) * 3;
				const double wt = w[t];
				a0 += wt * px[0]; a1 += wt * px[1]; a2 += wt * px[2];
			}
			uint8_t* op = drow + static_cast<size_t>(x) * 3;
			op[0] = RoundClipU8(a0); op[1] = RoundClipU8(a1); op[2] = RoundClipU8(a2);
		}
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
	// Antialiased bilinear downscale to the model size — matches training/reference
	// (inference_simulation) exactly. cv::INTER_AREA does NOT and inflated scores.
	const cv::Mat* pSrc = &rawImg;
	if (channels == 1) {
		cv::cvtColor(rawImg, m_bgrImage, cv::COLOR_GRAY2BGR);
		pSrc = &m_bgrImage;
	}
	ResizeAntialias(*pSrc, m_resizeImage);

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