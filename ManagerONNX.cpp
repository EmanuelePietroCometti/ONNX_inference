#include "ManagerONNX.h"
#include <iostream>
#include <stdexcept>
#include <fmt/core.h>
#include <fmt/color.h>

ManagerONNX::ManagerONNX() : hMapFile(NULL), pSharedList(NULL), hGlobalMutex(NULL), hEventTrigger(NULL), hEventAck(NULL)
{
	if (!InitializeIPC())
	{
		fmt::print(stderr, fg(fmt::color::red) | fmt::emphasis::bold, "### ERROR: Global IPC initialization failed!\n");
		throw std::runtime_error("### ERROR: Global IPC initialization failed!");
	}
}

ManagerONNX::~ManagerONNX()
{
	if (pSharedList) UnmapViewOfFile(pSharedList);
	if (hMapFile) CloseHandle(hMapFile);
	if (hGlobalMutex) CloseHandle(hGlobalMutex);
	if (hEventTrigger) CloseHandle(hEventTrigger);
	if (hEventAck) CloseHandle(hEventAck);
}

void ManagerONNX::Run()
{
	fmt::print("ONNX manager started. Waiting for signals...\n");

	bool isRunning = true;
	while (isRunning)
	{
		// ONNX manager waits for a signal from the external program to start
		WaitForSingleObject(hEventTrigger, INFINITE);

		DWORD waitMutex = WaitForSingleObject(hGlobalMutex, INFINITE);
		if (waitMutex == WAIT_OBJECT_0) {
			__try {
				if (pSharedList->state == ListState::QUIT) {
					HandleTermination();
					isRunning = false;
				}
				else if (pSharedList->state == ListState::UPDATE_PENDING) {
					HandleConfiguration();
				}
			}
			__finally {
				ReleaseMutex(hGlobalMutex);
			}
		}
	}
	fmt::print("ONNX manager correctly terminated.\n");
}

bool ManagerONNX::InitializeIPC()
{
	// Mapping global shared memory
	hMapFile = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, sizeof(controlPointsList), L"CONTROLPOINTLIST");

	if (hMapFile == NULL)
	{
		fmt::print(stderr, fg(fmt::color::red) | fmt::emphasis::bold, "### ERROR: CreateFileMapping failed with code: {}\n", GetLastError());
		return false;
	}

	pSharedList = (PTcontrolPointsList)MapViewOfFile(hMapFile, FILE_MAP_WRITE, 0, 0, sizeof(controlPointsList));

	if (pSharedList == NULL)
	{
		DWORD dwErr = GetLastError();
		CloseHandle(hMapFile);
		hMapFile = NULL;
		fmt::print(stderr, fg(fmt::color::red) | fmt::emphasis::bold, "### ERROR: MapViewOfFile failed with code: {}\n", dwErr);
		return false;
	}

	// Global mutex and events initialization
	hGlobalMutex = CreateMutex(NULL, FALSE, TEXT("LISTMUTEX"));
	hEventTrigger = CreateEvent(NULL, FALSE, FALSE, TEXT("LISTEVENTTRIGGER"));
	hEventAck = CreateEvent(NULL, FALSE, FALSE, TEXT("LISTEVENTACK"));

	return (hGlobalMutex && hEventTrigger && hEventAck);
}

void ManagerONNX::HandleConfiguration()
{
	fmt::print(">> Configuration request received (UPDATE_PENDING)!\n");

	activeWorkers.clear();
	bool configSuccess = true;

	try {
		// Iteration on the control points list to configure each one
		for (DWORD i = 0; i < pSharedList->numPunti; i++) {
			DWORD idPunto = pSharedList->points[i].idPunto;

			// Start a new worker for this point 
			activeWorkers[idPunto] = std::make_unique<WorkerONNX>(&pSharedList->points[i]);
			activeWorkers[idPunto]->Start();

			// Set the state to CONFIGURED after the thread creation
			activeWorkers[idPunto]->MarkAsCOnfigured();
		}
	}
	catch (const std::exception& e) {
		fmt::print(stderr, "### ERROR DURING CONFIGURATION: {}\n", e.what());
		configSuccess = false;
	}

	if (!configSuccess) {
		pSharedList->state = ListState::ERROR_DETECTED;
	}

	// Notify the external program that all threads started and configured correctly
	SetEvent(hEventAck);

	fmt::print(">> Configuration completed. Ack sent!\n");
}

void ManagerONNX::HandleTermination()
{
	fmt::print(">> Shutdown request received (QUIT)!\n");

	// Iterate on each worker and call the Stop method
	activeWorkers.clear();

	// Send ack event 
	SetEvent(hEventAck);
	fmt::print(">> Shutdown completed. Ack sent!\n");
}