#include "WorkerONNX.h"
#include "ClassificationEngine.h"
#include "AsyncLogger.h"
#include "RealTimeConfig.h"
#include <chrono>
#include <fmt/core.h>
#include <fmt/xchar.h>
#include <string>
#include <vector>
#include <cstdint>

namespace fs = std::filesystem;

bool WorkerONNX::verifyONNXModelPathAndEngine(const fs::path& filePath)
{
	WORD packageRelease = PackageInfo::kRelease;
	std::string suffix;
	if (packageRelease == 1 && executionProvider == ExecutionProvider::TensorRT)
	{
		suffix = "_ctx.onnx";
	}
	else
	{
		suffix = ".onnx";
	}
	
	if (!fs::exists(filePath) || !fs::is_regular_file(filePath))
	{
		Log::Error("ONNX model file is invalid: {}", filePath.string());
		return false;
	}

	std::string extension = filePath.extension().string();
	if (extension != ".onnx" && extension != ".ONNX")
	{
		Log::Error("ONNX model file has invalid extension: {}", filePath.string());
		return false;
	}

	std::string filename = filePath.filename().string();
#if __cplusplus >= 202002L
	if (!filename.ends_with(suffix))
	{
		Log::Error("ONNX model file has invalid filename: {}", filePath.string());
		return false;
	}
#else
	if (filename.length() <= suffix.length() || filename.compare(filename.length() - suffix.length(), suffix.length(), suffix) != 0)
	{
		Log::Error("ONNX model file has invalid filename: {}", filePath.string());
		return false;
	}
#endif
	
	std::string engineStem = filePath.stem().string();
	const std::string ctxSuffix = "_ctx";
	if (engineStem.size() > ctxSuffix.size() && engineStem.compare(engineStem.size() - ctxSuffix.size(), ctxSuffix.size(), ctxSuffix) == 0)
	{
		engineStem.erase(engineStem.size() - ctxSuffix.size());
	}

	fs::path enginePath = filePath.parent_path() / (engineStem + ".engine");
	if (!fs::exists(enginePath) || !fs::is_regular_file(enginePath))
	{
		Log::Error("Engine file not found: {}", enginePath.string());
		return false;
	}
	return true;
}

WorkerONNX::WorkerONNX(OrtRuntime& runtime, PTcontrolPoint pPoint, unsigned workerIndex, unsigned workerCount, ExecutionProvider executionProvider) :
	ortRuntime(runtime),
	controlPointData(pPoint),
	workerIndex(workerIndex),
	cpuPartition(RT::ComputeCpuPartition(workerIndex, workerCount)),
	executionProvider(executionProvider),
	hLocalMutex(NULL),
	hEventReady(NULL),
	hEventResults(NULL),
	hMMFImage(NULL),
	pImageBuffer(NULL),
	hMMFResImage(NULL),
	pResImageBuffer(NULL)
{
	try
	{
		InitializeLocalPC();
	}
	catch (...)
	{
		ReleaseResources();
		throw;
	}
}

WorkerONNX::~WorkerONNX()
{
	Stop();
	ReleaseResources();
}

void WorkerONNX::Start()
{
	workerThread = std::thread(&WorkerONNX::InferenceLoop, this);
}

void WorkerONNX::Stop()
{
	DWORD waitRes = WaitForSingleObject(hLocalMutex, INFINITE);
	if (waitRes == WAIT_OBJECT_0 || waitRes == WAIT_ABANDONED)
	{
		Log::Info(">> Worker {} shutdown pending...", controlPointData->idPunto);
		controlPointData->status = PointState::QUIT;
		ReleaseMutex(hLocalMutex);
	}

	// If the thread is blocked, wake it up and wait for it to join
	if (hEventReady) {
		SetEvent(hEventReady);
	}
	if (workerThread.joinable()) {
		workerThread.join();
	}
	Log::Info(">> Worker {} stopped!", controlPointData->idPunto);
}

void WorkerONNX::MarkAsConfigured()
{
	DWORD waitRes = WaitForSingleObject(hLocalMutex, INFINITE);
	if (waitRes == WAIT_OBJECT_0 || waitRes == WAIT_ABANDONED)
	{
		controlPointData->status = PointState::CONFIGURED;
		Log::Info("Worker {} CONFIGURED", controlPointData->idPunto);
		ReleaseMutex(hLocalMutex);
	}
}

void WorkerONNX::MarkAsError()
{
	DWORD waitRes = WaitForSingleObject(hLocalMutex, INFINITE);
	if (waitRes == WAIT_OBJECT_0 || waitRes == WAIT_ABANDONED)
	{
		controlPointData->status = PointState::ERROR_DETECTED;
		Log::Error("Worker {} ERROR_DETECTED", controlPointData->idPunto);
		ReleaseMutex(hLocalMutex);
	}
}

void WorkerONNX::ReleaseResources()
{
	if (pImageBuffer)
	{
		UnmapViewOfFile(pImageBuffer);
		pImageBuffer = NULL;
	}
	if (pResImageBuffer)
	{
		UnmapViewOfFile(pResImageBuffer);
		pResImageBuffer = NULL;
	}
	if (hMMFImage)
	{
		CloseHandle(hMMFImage);
		hMMFImage = NULL;
	}
	if (hMMFResImage)
	{
		CloseHandle(hMMFResImage);
		hMMFResImage = NULL;
	}
	if (hLocalMutex)
	{
		CloseHandle(hLocalMutex);
		hLocalMutex = NULL;
	}
	if (hEventReady)
	{
		CloseHandle(hEventReady);
		hEventReady = NULL;
	}
	if (hEventResults)
	{
		CloseHandle(hEventResults);
		hEventResults = NULL;
	}
}

void WorkerONNX::InferenceLoop()
{
	try {
		// TIME_CRITICAL priority + first core of this worker's slice: the worker
		// must preempt everything else the instant hEventReady fires, and its ORT
		// pool threads own the remaining slice cores (see RealTimeConfig.h)
		RT::ConfigureInferenceThread(workerIndex, cpuPartition);

		Log::Info(">> Worker thread started for control point: {}", controlPointData->idPunto);
		bool shouldQuit = false;

		// Reused across frames: statusStr keeps its heap capacity after the first
		// frame, so the steady-state loop performs no allocations at all
		std::string statusStr;
		float modelClassificationThreshold = 0.0f;
		float modelSegmentationThreshold = 0.0f;
		float modelClassificationThresholdNorm = 0.0f;
		float modelSegmentationThresholdNorm = 0.0f;
		std::string classNames;
		const AnomalyEngine* anomalyEngine = dynamic_cast<const AnomalyEngine*>(aiEngine.get());
		if (const AnomalyEngine* anomaly = anomalyEngine)
		{
			modelClassificationThreshold = anomaly->GetClassificationThreshold();
			modelSegmentationThreshold = anomaly->GetSegmentationThreshold();
			modelClassificationThresholdNorm = anomaly->GetClassificationThresholdNormalized();
			modelSegmentationThresholdNorm = anomaly->GetSegmentationThresholdNormalized();
		}
		else if (const auto* classification = dynamic_cast<const ClassificationEngine*>(aiEngine.get()))
		{
			classNames = classification->GetNamesJSONString();
		}
		while (!shouldQuit) {
			bool inferenceError = false;

			// Thread waiting for the signal eventReady
			WaitForSingleObject(hEventReady, INFINITE);

			// Lock acquire for state check
			DWORD waitRes = WaitForSingleObject(hLocalMutex, INFINITE);
			if (waitRes == WAIT_OBJECT_0 || waitRes == WAIT_ABANDONED) {
				if (controlPointData->status == PointState::QUIT || controlPointData->status == PointState::UPDATE_PENDING) {
					shouldQuit = true;
				}
				ReleaseMutex(hLocalMutex);

				if (shouldQuit) break;

				float anomalyScore = 0.0f;			// filled only by ANOMALY engines
				float confidence = 0.0f;     // filled only by CLASSIFICATION engines
				// Infer() writes its single scalar result into the slot matching the engine type
				float* output = nullptr;
				switch (controlPointData->inferenceType) {
				case InferenceType::ANOMALY:
					output = &anomalyScore;
					break;
				case InferenceType::CLASSIFICATION:
					output = &confidence;
					break;
				default:
					break;   // unsupported type: scoreOut stays null, handled as error below
				}
				statusStr.assign("ERROR");

				// Measure the end-to-end inference latency
				const auto inferStart = std::chrono::steady_clock::now();

				try {
					if (aiEngine && pImageBuffer != nullptr && pResImageBuffer != nullptr) {
						const uint8_t* pRawInput = static_cast<const uint8_t*>(pImageBuffer);
						uint8_t* pRawHeatmap = static_cast<uint8_t*>(pResImageBuffer);

						// Execute hardware-accelerated inference
						aiEngine->Infer(pRawInput,
							controlPointData->sizeX,
							controlPointData->sizeY,
							controlPointData->bpp / 8,
							pRawHeatmap,
							*output,
							statusStr,
							controlPointData->idPunto);
					}
					else {
						// Engine or shared buffers missing: the frame cannot be
						// processed, report it as an error instead of publishing
						// a stale/invalid result
						inferenceError = true;
					}
				}
				catch (const std::exception& e) 
				{
					Log::Error("### AI INFERENCE EXCEPTION ON CP {}: {}", controlPointData->idPunto, e.what());
					inferenceError = true;
				}
				catch (...)
				{
					Log::Error("### AI INFERENCE EXCEPTION ON CP {}: {}", controlPointData->idPunto, "unknown error");
					inferenceError = true;
				}
				float anomalyScoreNorm = 0.0f;
				// Normalized score of infered frame
				if (!inferenceError && anomalyEngine) {
					anomalyScoreNorm = anomalyEngine->GetLastClassificationScoreNormalized();
				}
				const double inferMs = std::chrono::duration<double, std::milli>(
					std::chrono::steady_clock::now() - inferStart).count();

				// On error, force the published values to the error sentinel BEFORE
				// building the JSON. Infer() may throw AFTER it already wrote
				// statusStr="OK"/"REJECT" and anomalyScore (e.g. the score path
				// succeeded but heatmap rendering threw): without this reset the
				// ANOMALY JSON below would carry that stale "valid" result while the
				// state is ERROR_DETECTED — an inconsistency a consumer could act on.
				if (inferenceError) {
					statusStr.assign("ERROR");
					anomalyScore = 0.0f;
				}

				// The result JSON has a fixed schema: build it with fmt straight
				// onto the stack. No nlohmann DOM, no heap allocation, no wstring
				// temporary — the hot path only formats and copies
				char jsonBuf[512];
				size_t jsonLen = 0;

				if (controlPointData->inferenceType == InferenceType::ANOMALY) {
					const auto out = fmt::format_to_n(jsonBuf, sizeof(jsonBuf) - 1,
						R"({{"anomaly_score":{:.6f},"anomaly_score_norm":{:.6f},)"
						R"("classificationThreshold":{:.6f},"classificationThresholdNorm":{:.6f},)"
						R"("segmentationThreshold":{:.6f},"segmentationThresholdNorm":{:.6f},"status":"{}"}})",
						anomalyScore, anomalyScoreNorm,
						modelClassificationThreshold, modelClassificationThresholdNorm,
						modelSegmentationThreshold, modelSegmentationThresholdNorm, statusStr);
					jsonLen = out.size;
					jsonLen = out.size;
				}
				else if (controlPointData->inferenceType == InferenceType::CLASSIFICATION) {
					// statusStr carries the class id as text; on error publish -1
					const auto out = inferenceError
						? fmt::format_to_n(jsonBuf, sizeof(jsonBuf) - 1,
							R"({{"class_id":"-1","confidence":0.0,"names":{}}})", classNames)
						: fmt::format_to_n(jsonBuf, sizeof(jsonBuf) - 1,
							R"({{"class_id":"{}","confidence":{:.6f},"names":{}}})", statusStr, confidence, classNames);
					jsonLen = out.size;
				}
				jsonLen = (jsonLen < sizeof(jsonBuf)) ? jsonLen : sizeof(jsonBuf) - 1;
				jsonBuf[jsonLen] = '\0';

				// SYNCHRONIZE STATE AND JSON
				DWORD resWait = WaitForSingleObject(hLocalMutex, INFINITE);
				if (resWait == WAIT_OBJECT_0 || resWait == WAIT_ABANDONED) {
					if (!inferenceError) {
						controlPointData->results.state = InferenceState::RESULT_READY;
					}
					else {
						controlPointData->results.state = InferenceState::ERROR_DETECTED;
					}

					// The JSON is pure ASCII: widen it char-by-char directly into
					// the shared buffer, without any intermediate std::wstring
					constexpr size_t kJsonCapacity = sizeof(controlPointData->results.json) / sizeof(TCHAR);
					const size_t copyLen = (jsonLen < kJsonCapacity - 1) ? jsonLen : kJsonCapacity - 1;
					for (size_t k = 0; k < copyLen; ++k) {
						controlPointData->results.json[k] = static_cast<TCHAR>(static_cast<unsigned char>(jsonBuf[k]));
					}
					controlPointData->results.json[copyLen] = 0;

					// Release Mutex BEFORE signaling, so the producer never wakes
					// up just to block on the mutex we still hold
					ReleaseMutex(hLocalMutex);

					// Signal the producer immediately that data is ready
					SetEvent(hEventResults);
				}
				Log::Info("Worker {} | infer {:.1f} ms | {}", controlPointData->idPunto,
					inferMs, fmt::string_view(jsonBuf, jsonLen));
			}
		}
		Log::Info(">> Worker thread stopped for control point: {}", controlPointData->idPunto);
	}
	catch (const std::exception& e) {
		Log::Error("### FATAL ERROR in worker thread of CP {}: {}", controlPointData->idPunto, e.what());
		MarkAsError();
	}
	catch (...) {
		Log::Error("### FATAL NON-STANDARD ERROR in worker thread of CP {}", controlPointData->idPunto);
		MarkAsError();
	}
}

void WorkerONNX::InitializeLocalPC()
{
	if (controlPointData->bpp != 8 && controlPointData->bpp != 24)
	{
		controlPointData->status = PointState::ERROR_DETECTED;
		Log::Error("### ERROR: invalid bpp value {}", controlPointData->bpp);
		throw std::runtime_error("Invalid bpp value!");
	}

	if (controlPointData->sizeX == 0 || controlPointData->sizeY == 0)
	{
		controlPointData->status = PointState::ERROR_DETECTED;
		Log::Error("### ERROR: invalid size value ({}x{})", controlPointData->sizeX, controlPointData->sizeY);
		throw std::runtime_error("Invalid size value!");
	}

	// Keep local handles 
	hLocalMutex = OpenMutexW(MUTEX_ALL_ACCESS, FALSE, controlPointData->mutexName);
	hEventReady = OpenEventW(EVENT_ALL_ACCESS, FALSE, controlPointData->eventReadyName);
	hEventResults = OpenEventW(EVENT_ALL_ACCESS, FALSE, controlPointData->resultsEventName);
	if (hLocalMutex == NULL || hEventReady == NULL || hEventResults == NULL)
	{
		Log::Error("### ERROR: point objects not found (error {})", GetLastError());
		throw std::runtime_error("Point mutex/events not available!");
	}

	// Map image buffer for this point
	int nPayload = controlPointData->sizeX * controlPointData->sizeY * controlPointData->bpp / 8;
	std::wstring mmfName = fmt::format(IPC_NAME_PREFIX L"MMF_{}_IMAGE", controlPointData->idPunto);

	hMMFImage = CreateFileMappingW(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, nPayload, mmfName.c_str());
	if (hMMFImage == NULL) {
		throw std::runtime_error("Failed to create or open the image MMF!");
	}

	pImageBuffer = MapViewOfFile(hMMFImage, FILE_MAP_READ, 0, 0, nPayload);
	if (pImageBuffer == NULL) {
		DWORD dwErr = GetLastError();
		CloseHandle(hMMFImage);
		hMMFImage = NULL;
		Log::Error("### ERROR: MapViewOfFile failed with code: {}", dwErr);
		throw std::runtime_error("MapViewOfFile failed!");
	}


	std::wstring mmfResName = fmt::format(IPC_NAME_PREFIX L"MMF_{}_RESIMAGE", controlPointData->idPunto);
	const size_t nResPayload = (size_t)controlPointData->sizeX * controlPointData->sizeY * 3;   // anomaly map 24 bit (BGR, rows packed width*3, no padding)
	hMMFResImage = CreateFileMappingW(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, static_cast<DWORD>(nResPayload), mmfResName.c_str());
	if (hMMFResImage == NULL) {
		Log::Error("### ERROR: CreateFileMapping (result image) failed with code: {}", GetLastError());
		controlPointData->status = PointState::ERROR_DETECTED;	
		throw std::runtime_error("Failed to create or open the result image MMF!");
	}

	pResImageBuffer = MapViewOfFile(hMMFResImage, FILE_MAP_WRITE, 0, 0, nResPayload);
	if (pResImageBuffer == NULL) {
		DWORD dwErr = GetLastError();
		CloseHandle(hMMFResImage);
		hMMFResImage = NULL;
		Log::Error("### ERROR: MapViewOfFile failed with code: {}", dwErr);
		controlPointData->status = PointState::ERROR_DETECTED;
		throw std::runtime_error("MapViewOfFile failed!");
	}

	std::filesystem::path modelPath(controlPointData->pathModello);
	
	switch (controlPointData->inferenceType) {
	case InferenceType::ANOMALY:
		Log::Info("Initialize ONNX engine for ANOMALY!");
		aiEngine = std::make_unique<AnomalyEngine>(ortRuntime);
		if (!verifyONNXModelPathAndEngine(modelPath))
		{
			Log::Error("### ERROR: ONNX model file or engine file not found or invalid for control point {}!", controlPointData->idPunto);
			controlPointData->status = PointState::ERROR_DETECTED;
			throw std::runtime_error("ONNX model file or engine file not found or invalid for control point!");
		}
		aiEngine->Initialize(modelPath, cpuPartition, executionProvider);
		break;
	case InferenceType::CLASSIFICATION:
		Log::Info("Initialize ONNX engine for CLASSIFICATION!");
		aiEngine = std::make_unique<ClassificationEngine>(ortRuntime);
		if (!verifyONNXModelPathAndEngine(modelPath))
		{
			Log::Error("### ERROR: ONNX model file or engine file not found or invalid for control point {}!", controlPointData->idPunto);
			controlPointData->status = PointState::ERROR_DETECTED;
			throw std::runtime_error("ONNX model file or engine file not found or invalid for control point!");
		}
		aiEngine->Initialize(modelPath, cpuPartition, executionProvider);
		break;
	case InferenceType::OBJECT_DETECTION:
		Log::Info("Initialize ONNX engine for OBJECT DETECTION!");
		// engine = std::make_unique<ObjectDetectionEngine>();
		break;
	default:
		throw std::runtime_error("Unsupported Inference Type!");
	}
}