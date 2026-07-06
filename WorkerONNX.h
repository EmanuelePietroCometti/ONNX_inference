#pragma once


#include "IPC_Shared.h"
#include "IEngine.h"
#include "AnomalyEngine.h"
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
	std::unique_ptr<IEngine> aiEngine;
	// HANDLE for the local mutex and event
	PTcontrolPoint controlPointData;
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