#include "WorkerONNX.h"
#include <iostream>
#include <chrono>
#include <fmt/core.h>
#include <fmt/xchar.h>
#include <string>
#include <vector>
#include <cstdint>

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

void WorkerONNX::MarkAsCOnfigured()
{
	DWORD waitRes = WaitForSingleObject(hLocalMutex, INFINITE);
	if (waitRes == WAIT_OBJECT_0)
	{
		controlPointData->status = PointState::CONFIGURED;
		fmt::print("Worker {} CONFIGURED\n", controlPointData->idPunto);
		ReleaseMutex(hLocalMutex);
	}
}

void WorkerONNX::InferenceLoop()
{
	fmt::print(">> Worker thread started for control point: {}\n", controlPointData->idPunto);
	while (true) {
		// Thread waiting for the signal eventReady
		WaitForSingleObject(hEventReady, INFINITE);

		// Lock acquire 
		DWORD waitRes = WaitForSingleObject(hLocalMutex, INFINITE);
		if (waitRes == WAIT_OBJECT_0) {
			bool shouldQuit = false;

			// Check if the state is QUIT or UPDATE_PENDING
			if (controlPointData->status == PointState::QUIT || controlPointData->status == PointState::UPDATE_PENDING) {
				shouldQuit = true;
			}
			else if (controlPointData->status == PointState::CONFIGURED) {
				if (pImageBuffer != NULL) {
					int nPayload = controlPointData->sizeX * controlPointData->sizeY * controlPointData->bpp / 8;

					std::vector<uint8_t> expectedInput(nPayload);
					for (int p = 0; p < nPayload; p++) {
						expectedInput[p] = static_cast<uint8_t>(p % 2);
					}

					if (memcmp(pImageBuffer, expectedInput.data(), nPayload) == 0) {
						fmt::print(">> CP {}: _IMAGE memcmp PASSED!\n", controlPointData->idPunto);
					}
					else {
						fmt::print(stderr, "## CP {}: _IMAGE memcmp FAILED!\n", controlPointData->idPunto);
					}
				}
			}

			ReleaseMutex(hLocalMutex);

			if (shouldQuit) break;

			fmt::print("Control point: {} - Inference running...\n", controlPointData->idPunto);

			// SIMULATE INFERENCE COMPUTE TIME
			std::this_thread::sleep_for(std::chrono::milliseconds(15));

			if (pResImageBuffer != NULL) {
				int nResPayload = controlPointData->sizeX * controlPointData->sizeY;
				uint8_t* pPixels = static_cast<uint8_t*>(pResImageBuffer);

				for (int p = 0; p < nResPayload; p++) {
					pPixels[p] = static_cast<uint8_t>(p % 2);
				}
			}

			fmt::print("Control point: {} - Inference finished!\n", controlPointData->idPunto);

			// SYNCHRONIZE STATE AND JSON
			// At the end of the inference, it acquires the lock 
			DWORD resWait = WaitForSingleObject(hLocalMutex, INFINITE);
			if (resWait == WAIT_OBJECT_0) {
				controlPointData->results.state = InferenceState::RESULT_READY;

				// Test results
				std::wstring jsonPayload = fmt::format(L"{{\"anomaly_score\": {:.2f}, \"status\": \"OK\"}}", (rand() % 100) / 100.0);
				wcscpy_s(controlPointData->results.json, 1024, jsonPayload.c_str());
				fmt::print("Worker {} results copied!\n", controlPointData->idPunto);

				ReleaseMutex(hLocalMutex);

				// Set the event to notify the external process that the results are ready
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
		// engine = std::make_unique<AnomalyDetectionEngine>();
		break;
	case InferenceType::CLASSIFICATION:
		fmt::print("Initialize ONNX engine for CLASSIFICATION!\n");
		// engine = std::make_unique<ClassificationEngine>();
		break;
	case InferenceType::OBJECT_DETECTION:
		fmt::print("Initialize ONNX engine for OBJECT DETECTION!\n");
		// engine = std::make_unique<ObjectDetectionEngine>();
		break;
	default:
		throw std::runtime_error("Unsupported Inference Type!");
	}
}