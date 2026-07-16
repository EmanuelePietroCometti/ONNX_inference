#pragma once


#include "IPC_Shared.h"
#include "IEngine.h"
#include "AnomalyEngine.h"
#include <thread>
#include <atomic>
#include <string>

class WorkerONNX {
public:
	// workerIndex is the 0-based creation index and workerCount the number of
	// concurrent control points: together they determine the slice of physical
	// cores this worker's session owns (see RT::ComputeCpuPartition)
	WorkerONNX(PTcontrolPoint pPoint, unsigned workerIndex, unsigned workerCount);
	~WorkerONNX();

	// Helper function for the worker class that allow to start, stop and mark configured 
	// the control point
	void Start();
	void Stop();
	void MarkAsConfigured();
	void MarkAsError();
private:
	std::unique_ptr<IEngine> aiEngine;
	// HANDLE for the local mutex and event
	PTcontrolPoint controlPointData;
	unsigned workerIndex;
	// Slice of physical cores owned by this worker + its ORT session
	RT::CpuPartition cpuPartition;
	std::thread workerThread;

	// Handle di sincronizzazione
	HANDLE hLocalMutex;
	HANDLE hEventReady;
	HANDLE hEventResults;

	// MMF Input
	HANDLE hMMFImage;
	LPVOID pImageBuffer;

	// MMF Output
	HANDLE hMMFResImage;
	LPVOID pResImageBuffer;

	void InferenceLoop();
	void InitializeLocalPC();
};