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

WorkerONNX::WorkerONNX(PTcontrolPoint pPoint, unsigned workerIndex, unsigned workerCount) :
	controlPointData(pPoint),
	workerIndex(workerIndex),
	cpuPartition(RT::ComputeCpuPartition(workerIndex, workerCount)),
	hLocalMutex(NULL),
	hEventReady(NULL), 
	hEventResults(NULL), 
	hMMFImage(NULL), 
	pImageBuffer(NULL),
	hMMFResImage(NULL),
	pResImageBuffer(NULL)
{
	InitializeLocalPC();
}

WorkerONNX::~WorkerONNX()
{
	Stop();
	if (pImageBuffer) UnmapViewOfFile(pImageBuffer);
	if (pResImageBuffer) UnmapViewOfFile(pResImageBuffer);
	if (hMMFImage) CloseHandle(hMMFImage);
	if (hLocalMutex) CloseHandle(hLocalMutex);
	if (hEventReady) CloseHandle(hEventReady);
	if (hEventResults) CloseHandle(hEventResults);
	if (hMMFResImage) CloseHandle(hMMFResImage);
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

void WorkerONNX::InferenceLoop()
{
	// TIME_CRITICAL priority + first core of this worker's slice: the worker
	// must preempt everything else the instant hEventReady fires, and its ORT
	// pool threads own the remaining slice cores (see RealTimeConfig.h)
	RT::ConfigureInferenceThread(workerIndex, cpuPartition);

	Log::Info(">> Worker thread started for control point: {}", controlPointData->idPunto);
	bool shouldQuit = false;
	uint32_t frameCounter = 0; // Local counter for throttling logs

	// Reused across frames: statusStr keeps its heap capacity after the first
	// frame, so the steady-state loop performs no allocations at all
	std::string statusStr;

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

			float anomalyScore = 0.0f;
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
						anomalyScore,
						statusStr);
				}
				else {
					// Engine or shared buffers missing: the frame cannot be
					// processed, report it as an error instead of publishing
					// a stale/invalid result
					inferenceError = true;
				}
			}
			catch (const std::exception& e) {
				Log::Error("### AI INFERENCE EXCEPTION ON CP {}: {}", controlPointData->idPunto, e.what());
				inferenceError = true;
			}

			const double inferMs = std::chrono::duration<double, std::milli>(
				std::chrono::steady_clock::now() - inferStart).count();

			// The result JSON has a fixed schema: build it with fmt straight
			// onto the stack. No nlohmann DOM, no heap allocation, no wstring
			// temporary — the hot path only formats and copies
			char jsonBuf[512];
			size_t jsonLen = 0;

			if (controlPointData->inferenceType == InferenceType::ANOMALY) {
				const auto out = fmt::format_to_n(jsonBuf, sizeof(jsonBuf) - 1,
					R"({{"anomaly_score":{:.6f},"status":"{}"}})", anomalyScore, statusStr);
				jsonLen = out.size;
			}
			else if (controlPointData->inferenceType == InferenceType::CLASSIFICATION) {
				// statusStr carries the class id as text; on error publish -1
				const auto out = inferenceError
					? fmt::format_to_n(jsonBuf, sizeof(jsonBuf) - 1,
						R"({{"class_id":"-1","confidence":0.0}})")
					: fmt::format_to_n(jsonBuf, sizeof(jsonBuf) - 1,
						R"({{"class_id":"{}","confidence":{:.6f}}})", statusStr, anomalyScore);
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

			// Throttled telemetry: even this enqueue is allocation-free, the
			// AsyncLogger consumer does the console I/O on core 0
			frameCounter++;
			/*
			if (frameCounter % 100 == 0) {
				Log::Info("Worker {} | infer {:.1f} ms | {}", controlPointData->idPunto,
					inferMs, fmt::string_view(jsonBuf, jsonLen));
			}
			*/
		}
	}
	Log::Info(">> Worker thread stopped for control point: {}", controlPointData->idPunto);
}

void WorkerONNX::InitializeLocalPC()
{
	// Keep local handles 
	hLocalMutex = OpenMutex(MUTEX_ALL_ACCESS, FALSE, controlPointData->mutexName);
	hEventReady = OpenEvent(EVENT_ALL_ACCESS, FALSE, controlPointData->eventReadyName);
	hEventResults = OpenEvent(EVENT_ALL_ACCESS, FALSE, controlPointData->resultsEventName);

	// Map image buffer for this point
	int nPayload = controlPointData->sizeX * controlPointData->sizeY * controlPointData->bpp / 8;
	std::wstring mmfName = fmt::format(L"MMF_{}_IMAGE", controlPointData->idPunto);

	hMMFImage = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, nPayload, mmfName.c_str());
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


	std::wstring mmfResName = fmt::format(L"MMF_{}_RESIMAGE", controlPointData->idPunto);

	hMMFResImage = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, nPayload, mmfResName.c_str());
	if (hMMFResImage == NULL) {
		throw std::runtime_error("Failed to create or open the image MMF!");
	}

	pResImageBuffer = MapViewOfFile(hMMFResImage, FILE_MAP_WRITE, 0, 0, nPayload);
	if (pResImageBuffer == NULL) {
		DWORD dwErr = GetLastError();
		CloseHandle(hMMFResImage);
		hMMFResImage = NULL;
		Log::Error("### ERROR: MapViewOfFile failed with code: {}", dwErr);
		throw std::runtime_error("MapViewOfFile failed!");
	}



	switch (controlPointData->inferenceType) {
	case InferenceType::ANOMALY:
		Log::Info("Initialize ONNX engine for ANOMALY!");
		aiEngine = std::make_unique<AnomalyEngine>();
		aiEngine->Initialize(controlPointData->pathModello, cpuPartition);
		break;
	case InferenceType::CLASSIFICATION:
		Log::Info("Initialize ONNX engine for CLASSIFICATION!");
		aiEngine = std::make_unique<ClassificationEngine>();
		aiEngine->Initialize(controlPointData->pathModello, cpuPartition);
		break;
	case InferenceType::OBJECT_DETECTION:
		Log::Info("Initialize ONNX engine for OBJECT DETECTION!");
		// engine = std::make_unique<ObjectDetectionEngine>();
		break;
	default:
		throw std::runtime_error("Unsupported Inference Type!");
	}
}