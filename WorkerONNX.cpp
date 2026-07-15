#include "WorkerONNX.h"
#include "ClassificationEngine.h"
#include <iostream>
#include <chrono>
#include <fmt/core.h>
#include <fmt/xchar.h>
#include <string>
#include <vector>
#include <cstdint>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

WorkerONNX::WorkerONNX(PTcontrolPoint pPoint) :
	controlPointData(pPoint),
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
		fmt::print(">> Worker {} shutdown pending...\n", controlPointData->idPunto);
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
	fmt::print(">> Worker {} stopped!\n", controlPointData->idPunto);
}

void WorkerONNX::MarkAsConfigured()
{
	DWORD waitRes = WaitForSingleObject(hLocalMutex, INFINITE);
	if (waitRes == WAIT_OBJECT_0 || waitRes == WAIT_ABANDONED)
	{
		controlPointData->status = PointState::CONFIGURED;
		fmt::print("Worker {} CONFIGURED\n", controlPointData->idPunto);
		ReleaseMutex(hLocalMutex);
	}
}

void WorkerONNX::MarkAsError()
{
	DWORD waitRes = WaitForSingleObject(hLocalMutex, INFINITE);
	if (waitRes == WAIT_OBJECT_0 || waitRes == WAIT_ABANDONED)
	{
		controlPointData->status = PointState::ERROR_DETECTED;
		fmt::print("Worker {} ERROR_DETECTED\n", controlPointData->idPunto);
		ReleaseMutex(hLocalMutex);
	}
}

void WorkerONNX::InferenceLoop()
{
	// Boost thread priority to minimize wake-up latency from WaitForSingleObject
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);

	fmt::print(">> Worker thread started for control point: {}\n", controlPointData->idPunto);
	bool shouldQuit = false;
	uint32_t frameCounter = 0; // Local counter for throttling logs

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
			std::string statusStr = "ERROR";

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
			}
			catch (const std::exception& e) {
				fmt::print(stderr, "### AI INFERENCE EXCEPTION ON CP {}: {}\n", controlPointData->idPunto, e.what());
				inferenceError = true;
			}

			const double inferMs = std::chrono::duration<double, std::milli>(
				std::chrono::steady_clock::now() - inferStart).count();

			// Variables for logging outside the mutex
			std::string serializedJson;

			// SYNCHRONIZE STATE AND JSON
			DWORD resWait = WaitForSingleObject(hLocalMutex, INFINITE);
			if (resWait == WAIT_OBJECT_0 || resWait == WAIT_ABANDONED) {
				if (!inferenceError) {
					controlPointData->results.state = InferenceState::RESULT_READY;
				}
				else {
					controlPointData->results.state = InferenceState::ERROR_DETECTED;
				}

				json responseJson;

				if (controlPointData->inferenceType == InferenceType::ANOMALY) {
					responseJson["anomaly_score"] = anomalyScore;
					responseJson["status"] = statusStr;
				}
				else if (controlPointData->inferenceType == InferenceType::CLASSIFICATION) {
					if (!inferenceError) {
						responseJson["class_id"] = stoi(statusStr);
						responseJson["confidence"] = anomalyScore;
					}
					else {
						responseJson["class_id"] = -1;
						responseJson["confidence"] = 0.0f;
					}
				}

				serializedJson = responseJson.dump();

				// Fast ASCII conversion
				std::wstring wJsonPayload(serializedJson.begin(), serializedJson.end());
				wcscpy_s(controlPointData->results.json, 1024, wJsonPayload.c_str());

				// Release Mutex BEFORE doing any console I/O
				ReleaseMutex(hLocalMutex);

				// Signal the producer immediately that data is ready
				SetEvent(hEventResults);
			}

			// Log only once every 100 frames to keep the hot path clean
			frameCounter++;
			if (frameCounter % 100 == 0) {
				fmt::print("Worker {} | infer {:.1f} ms | {}\n", controlPointData->idPunto, inferMs, serializedJson);
			}
		}
	}
	fmt::print(">> Worker thread stopped for control point: {}\n", controlPointData->idPunto);
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
		fmt::print(stderr, fg(fmt::color::red) | fmt::emphasis::bold, "### ERROR: MapViewOfFile failed with code: {}\n", dwErr);
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
		fmt::print(stderr, fg(fmt::color::red) | fmt::emphasis::bold, "### ERROR: MapViewOfFile failed with code: {}\n", dwErr);
		throw std::runtime_error("MapViewOfFile failed!");
	}



	switch (controlPointData->inferenceType) {
	case InferenceType::ANOMALY:
		fmt::print("Initialize ONNX engine for ANOMALY!\n");
		aiEngine = std::make_unique<AnomalyEngine>();
		aiEngine->Initialize(controlPointData->pathModello);
		break;
	case InferenceType::CLASSIFICATION:
		fmt::print("Initialize ONNX engine for CLASSIFICATION!\n");
		aiEngine = std::make_unique<ClassificationEngine>();
		aiEngine->Initialize(controlPointData->pathModello);
		break;
	case InferenceType::OBJECT_DETECTION:
		fmt::print("Initialize ONNX engine for OBJECT DETECTION!\n");
		// engine = std::make_unique<ObjectDetectionEngine>();
		break;
	default:
		throw std::runtime_error("Unsupported Inference Type!");
	}
}