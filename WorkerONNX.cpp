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
	if (waitRes == WAIT_OBJECT_0)
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
	if (waitRes == WAIT_OBJECT_0)
	{
		controlPointData->status = PointState::CONFIGURED;
		fmt::print("Worker {} CONFIGURED\n", controlPointData->idPunto);
		ReleaseMutex(hLocalMutex);
	}
}

void WorkerONNX::MarkAsError()
{
	DWORD waitRes = WaitForSingleObject(hLocalMutex, INFINITE);
	if (waitRes == WAIT_OBJECT_0)
	{
		controlPointData->status = PointState::ERROR_DETECTED;
		fmt::print("Worker {} ERROR_DETECTED\n", controlPointData->idPunto);
		ReleaseMutex(hLocalMutex);
	}
}

void WorkerONNX::InferenceLoop()
{
	fmt::print(">> Worker thread started for control point: {}\n", controlPointData->idPunto);
	bool shouldQuit = false;
	while (!shouldQuit) {
		bool inferenceError = false;
		// Thread waiting for the signal eventReady
		WaitForSingleObject(hEventReady, INFINITE);

		// Lock acquire 
		DWORD waitRes = WaitForSingleObject(hLocalMutex, INFINITE);
		if (waitRes == WAIT_OBJECT_0) {
			// Check if the state is QUIT or UPDATE_PENDING
			if (controlPointData->status == PointState::QUIT || controlPointData->status == PointState::UPDATE_PENDING) {
				shouldQuit = true;
			}

			ReleaseMutex(hLocalMutex);

			if (shouldQuit) break;

			fmt::print("Control point: {} - Inference running...\n", controlPointData->idPunto);

			float anomalyScore = 0.0f;
			std::string statusStr = "ERROR";

			try {
				// Call the ONNX Engine if successfully initialized and memory is mapped
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

			fmt::print("Control point: {} - Inference finished!\n", controlPointData->idPunto);


			// SYNCHRONIZE STATE AND JSON
			DWORD resWait = WaitForSingleObject(hLocalMutex, INFINITE);
			if (resWait == WAIT_OBJECT_0) {
				if (!inferenceError) {
					controlPointData->results.state = InferenceState::RESULT_READY;
				}
				else {
					controlPointData->results.state = InferenceState::ERROR_DETECTED;
				}

				// --- Build JSON payload safely using nlohmann/json ---
				json responseJson;

				if (controlPointData->inferenceType == InferenceType::ANOMALY) {
					responseJson["anomaly_score"] = anomalyScore;
					responseJson["status"] = statusStr; // Stored as a string
				}
				else if (controlPointData->inferenceType == InferenceType::CLASSIFICATION) {
					// For classification, statusStr holds the class ID.
					// On inference error statusStr is "ERROR", not a number: stoi would throw
					// with the mutex held and kill the worker thread.
					if (!inferenceError) {
						responseJson["class_id"] = stoi(statusStr);
						responseJson["confidence"] = anomalyScore;
					}
					else {
						responseJson["class_id"] = -1;
						responseJson["confidence"] = 0.0f;
					}
				}

				// Dump the JSON object to a standard string (compact, no indent)
				std::string serializedJson = responseJson.dump();

				// Convert to wide string to match your IPC Shared Memory TCHAR definition
				std::wstring wJsonPayload(serializedJson.begin(), serializedJson.end());

				// Copy the safely generated JSON directly into shared memory
				wcscpy_s(controlPointData->results.json, 1024, wJsonPayload.c_str());

				fmt::print("Worker {} results copied: {}\n", controlPointData->idPunto, serializedJson);

				ReleaseMutex(hLocalMutex);
				SetEvent(hEventResults);
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