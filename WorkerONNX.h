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
	std::thread workerThread;

	// Handle di sincronizzazione
	HANDLE hLocalMutex;
	HANDLE hEventReady;
	HANDLE hEventResults;

	// MMF Input (Immagine inviata dalla Selezionatrice)
	HANDLE hMMFImage;
	LPVOID pImageBuffer;

	// MMF Output (Anomaly Map scritta dall'AI) -> AGGIUNGI QUESTI
	HANDLE hMMFResImage;
	LPVOID pResImageBuffer;

	void InferenceLoop();
	void InitializeLocalPC();
};