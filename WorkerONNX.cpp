#include "WorkerONNX.h"
#include <iostream>
#include <chrono>
#include <fmt/core.h>
#include <fmt/xchar.h>
#include <string>

WorkerONNX::WorkerONNX(PTcontrolPoint pPoint) : controlPointData(pPoint), hLocalMutex(NULL), hEventReady(NULL), hEventResults(NULL), hMMFImage(NULL), pImageBuffer(NULL)
{
	InitializeLocalPC();
}

WorkerONNX::~WorkerONNX()
{
	Stop();
	if (pImageBuffer) UnmapViewOfFile(pImageBuffer);
	if (hMMFImage) CloseHandle(hMMFImage);
	if (hLocalMutex) CloseHandle(hLocalMutex);
	if (hEventReady) CloseHandle(hEventReady);
	if (hEventResults) CloseHandle(hEventResults);
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
				// TODO: Data extraction
				fmt::print("Worker {} critical section: copying image!\n", controlPointData->idPunto);
			}

			ReleaseMutex(hLocalMutex);

			if (shouldQuit) break;

			fmt::print("Control point: {} - Inference running...\n", controlPointData->idPunto);
			// TODO: call the inference
			std::this_thread::sleep_for(std::chrono::milliseconds(15));
			fmt::print("Control point: {} - Inference finished!\n", controlPointData->idPunto);

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
}