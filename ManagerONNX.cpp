#include "ManagerONNX.h"
#include "AsyncLogger.h"
#include <stdexcept>

ManagerONNX::ManagerONNX() : hMapFile(NULL), pSharedList(NULL), hGlobalMutex(NULL), hEventTrigger(NULL), hEventAck(NULL)
{
	if (!InitializeIPC())
	{
		Log::Error("### ERROR: Global IPC initialization failed!");
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
	Log::Info("ONNX manager started. Waiting for signals...");

	bool isRunning = true;
	while (isRunning)
	{
		// ONNX manager waits for a signal from the external program to start
		WaitForSingleObject(hEventTrigger, INFINITE);

		// WAIT_ABANDONED still grants ownership (previous owner crashed while
		// holding the mutex): it MUST be handled like WAIT_OBJECT_0, otherwise
		// the mutex is acquired here and never released, deadlocking both processes
		ListState requestedState = ListState::IDLE;
		DWORD waitMutex = WaitForSingleObject(hGlobalMutex, INFINITE);
		if (waitMutex == WAIT_OBJECT_0 || waitMutex == WAIT_ABANDONED) {
			requestedState = pSharedList->state;
			ReleaseMutex(hGlobalMutex);
		}
		if (pSharedList->state == ListState::QUIT) {
			HandleTermination();
			isRunning = false;
		}
		else if (pSharedList->state == ListState::UPDATE_PENDING) {
			HandleConfiguration();
		}
	}
	Log::Info("ONNX manager correctly terminated.");
}

bool ManagerONNX::InitializeIPC()
{
	// Global mutex and events initialization
	hGlobalMutex = CreateMutex(NULL, FALSE, TEXT("LISTMUTEX"));
	hEventTrigger = CreateEvent(NULL, FALSE, FALSE, TEXT("LISTEVENTTRIGGER"));
	hEventAck = CreateEvent(NULL, FALSE, FALSE, TEXT("LISTEVENTACK"));

	// Mapping global shared memory
	hMapFile = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, sizeof(controlPointsList), L"CONTROLPOINTLIST");

	if (hMapFile == NULL)
	{
		Log::Error("### ERROR: CreateFileMapping failed with code: {}", GetLastError());
		return false;
	}

	pSharedList = (PTcontrolPointsList)MapViewOfFile(hMapFile, FILE_MAP_WRITE, 0, 0, sizeof(controlPointsList));

	if (pSharedList == NULL)
	{
		DWORD dwErr = GetLastError();
		CloseHandle(hMapFile);
		hMapFile = NULL;
		Log::Error("### ERROR: MapViewOfFile failed with code: {}", dwErr);
		return false;
	}

	return (hGlobalMutex && hEventTrigger && hEventAck);
}

void ManagerONNX::HandleConfiguration()
{
	Log::Info(">> Configuration request received (UPDATE_PENDING)!");

	activeWorkers.clear();
	bool configSuccess = true;

	DWORD i = 0;
	try {
		// Iteration on the control points list to configure each one
		for (i = 0; i < pSharedList->numPunti; i++) {
			DWORD idPunto = pSharedList->points[i].idPunto;

			// Start a new worker for this point. Creation index + total count
			// determine the slice of physical cores owned by this worker's
			// session, so concurrent sessions never compete for a core
			activeWorkers[idPunto] = std::make_unique<WorkerONNX>(
				&pSharedList->points[i], i, pSharedList->numPunti);
			activeWorkers[idPunto]->Start();

			// Set the state to CONFIGURED after the thread creation
			activeWorkers[idPunto]->MarkAsConfigured();
		}
	}
	catch (const std::exception& e) {
		DWORD idPunto = pSharedList->points[i].idPunto;
		Log::Error("### ERROR DURING CONFIGURATION: {}", e.what());

		// If the WorkerONNX constructor threw, operator[] already inserted a NULL
		// unique_ptr for this id: calling MarkAsError() through it would dereference
		// a null pointer. In that case the worker never opened its local mutex either,
		// so write the error state directly into shared memory (the global mutex is
		// already held by Run()).
		auto it = activeWorkers.find(idPunto);
		if (it != activeWorkers.end() && it->second) {
			it->second->MarkAsError();
		}
		else {
			pSharedList->points[i].status = PointState::ERROR_DETECTED;
			Log::Error("Worker {} ERROR_DETECTED (construction failed)", idPunto);
			activeWorkers.erase(idPunto);
		}
		configSuccess = false;
	}

	DWORD waitMutex = WaitForSingleObject(hGlobalMutex, INFINITE);
	if (waitMutex == WAIT_OBJECT_0 || waitMutex == WAIT_ABANDONED) {
		pSharedList->state = configSuccess ? ListState::CONFIGURED : ListState::ERROR_DETECTED;
		ReleaseMutex(hGlobalMutex);
	}

	// Notify the external program that all threads started and configured correctly
	SetEvent(hEventAck);

	Log::Info(">> Configuration completed. Ack sent!");
}

void ManagerONNX::HandleTermination()
{
	Log::Info(">> Shutdown request received (QUIT)!");

	// Iterate on each worker and call the Stop method
	activeWorkers.clear();

	// Send ack event
	SetEvent(hEventAck);
	Log::Info(">> Shutdown completed. Ack sent!");
}