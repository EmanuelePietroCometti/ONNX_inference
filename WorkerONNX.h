#pragma once


#include "IPC_Shared.h"
#include "OrtEngine.h"
#include "AnomalyEngine.h"
#include "PackageRelease.h"
#include <thread>
#include <atomic>
#include <string>

namespace fs = std::filesystem;

class WorkerONNX {
public:
	// workerIndex is the 0-based creation index and workerCount the number of
	// concurrent control points: together they determine the slice of physical
	// cores this worker's session owns (see RT::ComputeCpuPartition)
	WorkerONNX(OrtRuntime& runtime, PTcontrolPoint pPoint, unsigned workerIndex, unsigned workerCount, ExecutionProvider executionProvider);
	~WorkerONNX();

	// Helper function for the worker class that allow to start, stop and mark configured 
	// the control point
	void Start();
	void Stop();
	void MarkAsConfigured();
	void MarkAsError();

	bool verifyONNXModelPathAndEngine(const fs::path& filePath);
private:
	void ReleaseResources();
	OrtRuntime& ortRuntime;
	std::unique_ptr<OrtEngine> aiEngine;
	// HANDLE for the local mutex and event
	PTcontrolPoint controlPointData;
	unsigned workerIndex;
	ExecutionProvider executionProvider;
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