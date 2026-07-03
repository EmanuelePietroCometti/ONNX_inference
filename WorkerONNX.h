#pragma once


#include "IPC_Shared.h"
#include <thread>
#include <atomic>
#include <string>

class WorkerONNX {
public:
	WorkerONNX(PTcontrolPoint pPoint);
	~WorkerONNX();

	// Helper function for the worker class that allow to start, stop and mark configured 
	// the control point
	void Start();
	void Stop();
	void MarkAsCOnfigured();
private:
	// HANDLE for the local mutex and event
	PTcontrolPoint controlPointData;
	HANDLE hLocalMutex;
	HANDLE hEventReady;
	HANDLE hEventResults;
	HANDLE hMMFImage;
	void* pImageBuffer;

	std::thread workerThread;

	void InferenceLoop();
	void InitializeLocalPC();
};