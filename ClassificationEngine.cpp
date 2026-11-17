#include "ClassificationEngine.h"
#include "IoBuffers.h"
#include "AsyncLogger.h"
#include <xmmintrin.h> 
#include <pmmintrin.h> 
#include <cstring>
#include <chrono>
#include <string>
#include <thread>
#include <unordered_map>
#include <algorithm>
#include <stdexcept>

ClassificationEngine::ClassificationEngine(OrtRuntime& runtime)
	: OrtEngine(runtime, "Classification"), config(IniConfiguration::Instance())
{
}

ClassificationEngine::~ClassificationEngine()
{
}

void ClassificationEngine::DescribeIo(IoBuffers& io)
{
	io.AddInput(inputName, inputTensorValues.data(), inputTensorValues.size() * sizeof(float),
		inputShape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

	io.AddOutput(m_outClassIdName, m_hostClassId.data(), sizeof(int64_t),
		m_classIdShape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);

	io.AddOutput(m_outConfidenceName, m_hostConfidence.data(), sizeof(float),
		m_confShape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
}

void ClassificationEngine::LoadGraphContract()
{
	// Metadata contract (shared with the anomaly engine): version, class names, normalisation, imgsz,
	// colour order and the resize interpolation. Throws when the model cannot be run as trained.
	const Contract::ClassificationContract contract = Contract::LoadClassificationContract(
		Contract::MetadataReader::FromSession(*session), modelWidth, modelHeight);
	m_classNames = contract.classNames;
	m_classNamesJSONString = contract.namesJson;
	m_resizeInterpolation = contract.preprocessing.cvInterpolation;

	// Output resolution
	// Resolve by name rather than by position: the export can emit class_id,
	// confidence, probs and class_name in an order that is not guaranteed.
	const bool hasClassId =
		std::find(outputNames.begin(), outputNames.end(), "class_id") != outputNames.end();
	const bool hasConfidence =
		std::find(outputNames.begin(), outputNames.end(), "confidence") != outputNames.end();

	if (!hasClassId || !hasConfidence) {
		throw std::runtime_error(
			"The ONNX graph does not expose both 'class_id' and 'confidence'. "
			"Export it with export_onnx.py, which folds argmax/max into the graph.");
	}

	m_outClassIdName = "class_id";
	m_outConfidenceName = "confidence";

	Log::Info("Classification graph: outputs '{}' and '{}' found, batch {}.", m_outClassIdName, m_outConfidenceName, inputShape[0]);
}


void ClassificationEngine::Initialize(const std::filesystem::path& modelPath, const RT::CpuPartition& cpuPartition, ExecutionProvider executionProvider)
{
	// Provider selection + session (OrtEngine)
	CreateSession(modelPath, cpuPartition, executionProvider);

	Ort::AllocatorWithDefaultOptions allocator;
	inputName = session->GetInputNameAllocated(0, allocator).get();

	outputNames.clear();
	for (size_t idx = 0; idx < session->GetOutputCount(); ++idx) {
		outputNames.emplace_back(session->GetOutputNameAllocated(idx, allocator).get());
	}

	Ort::TypeInfo typeInfo = session->GetInputTypeInfo(0);
	auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
	inputShape = tensorInfo.GetShape();

	if (inputShape[0] <= 0) inputShape[0] = 1;
	if (inputShape[2] <= 0 || inputShape[3] <= 0) {
		const int64_t imgsz = Contract::MetadataReader::FromSession(*session).RequireInt("imgsz");
		if (inputShape[2] <= 0) inputShape[2] = imgsz;
		if (inputShape[3] <= 0) inputShape[3] = imgsz;
	}

	modelChannels = inputShape[1];
	modelHeight = inputShape[2];
	modelWidth = inputShape[3];

	LoadGraphContract();

	// Cache output shapes dynamically
	for (size_t i = 0; i < session->GetOutputCount(); i++) {
		std::string name = session->GetOutputNameAllocated(i, allocator).get();
		auto shape = session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
		for (auto& d : shape) if (d < 0) d = 1;
		if (name == m_outClassIdName) m_classIdShape = shape;
		if (name == m_outConfidenceName) m_confShape = shape;
	}

	size_t totalElements = 1;
	for (int64_t dim : inputShape) totalElements *= dim;

	inputTensorValues.assign(totalElements, 0.0f);
	m_hostClassId.assign(1, 0);
	m_hostConfidence.assign(1, 0.0f);

	// I/O binding: the tensors are the ones listed in DescribeIo (OrtEngine picks where they live)
	CreateIoBuffers();

	double firstMs = 0.0, lastMs = 0.0;
	for (int r = 0; r < config.GetONNXWarmUps(); ++r) {
		auto startTime = std::chrono::steady_clock::now();
		Io().Run();
		lastMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - startTime).count();
		if (r == 0) firstMs = lastMs;
	}
	Log::Info("Classification Warmup completed: {} runs, first {:.1f} ms, last {:.1f} ms.", config.GetONNXWarmUps(), firstMs, lastMs);
}

void ClassificationEngine::Infer(const void* pInputImage, uint32_t width, uint32_t height, uint32_t channels,
	void* pOutputHeatmap, float& outAnomalyScore, std::string& outStatus, DWORD idPunto)
{
	auto t0 = std::chrono::steady_clock::now();
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

	const cv::Mat* pSrc = &rawImg;
	if (channels == 1) {
		cv::cvtColor(rawImg, m_bgrImage, cv::COLOR_GRAY2BGR);
		pSrc = &m_bgrImage;
	}
	cv::resize(*pSrc, m_resizeImage, cv::Size(static_cast<int>(modelWidth), static_cast<int>(modelHeight)), 0, 0, m_resizeInterpolation);
	cv::cvtColor(m_resizeImage, m_rgbImage, cv::COLOR_BGR2RGB);

	auto t1 = std::chrono::steady_clock::now();
	if (m_splitPlanes.empty()) m_splitPlanes.resize(modelChannels);

	const size_t planeSize = static_cast<size_t>(modelHeight) * static_cast<size_t>(modelWidth);
	cv::split(m_rgbImage, m_splitPlanes);

	for (int c = 0; c < modelChannels; ++c) {
		cv::Mat planeF32(static_cast<int>(modelHeight), static_cast<int>(modelWidth), CV_32FC1,
			inputTensorValues.data() + static_cast<size_t>(c) * planeSize);
		m_splitPlanes[c].convertTo(planeF32, CV_32FC1);
	}

	auto t2 = std::chrono::steady_clock::now();

	// Host -> device, inference, device -> host. On CPU/OpenVINO the two copies do nothing.
	Io().Upload();
	Io().Run();
	Io().Download();

	auto t3 = std::chrono::steady_clock::now();

	const int64_t classId = m_hostClassId[0];
	const float confidence = m_hostConfidence[0];

	if (classId < 0 || classId >= static_cast<int64_t>(m_classNames.size())) {
		throw std::runtime_error("class_id " + std::to_string(classId) + " out of range.");
	}

	auto t4 = std::chrono::steady_clock::now();

	outAnomalyScore = confidence;
	outStatus = m_classNames[static_cast<size_t>(classId)];

	using ms = std::chrono::duration<double, std::milli>;
	accResize += ms(t1 - t0).count();
	accNorm += ms(t2 - t1).count();
	accRun += ms(t3 - t2).count();
	accPost += ms(t4 - t3).count();
	maxRun = std::max(maxRun, ms(t3 - t2).count());

	if (++frameCount % 100 == 0) {
		Log::Info("avg over 100: resize {:.2f} | norm {:.2f} | RUN {:.2f} (max {:.2f}) | post {:.2f}",
			accResize / 100, accNorm / 100, accRun / 100, maxRun, accPost / 100);
		accResize = accNorm = accRun = accPost = maxRun = 0.0;
	}
}