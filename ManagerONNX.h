#pragma once

#include "IPC_Shared.h"
#include "WorkerONNX.h"

#include <Windows.h>
#include <map>
#include <memory>

class ManagerONNX {
public:
    ManagerONNX();
    ~ManagerONNX();

    // Main loop that waits for trigger
    void Run();
private:
    // IPC global handles
    HANDLE hMapFile;
    PTcontrolPointsList pSharedList;
    HANDLE hGlobalMutex;
    HANDLE hEventTrigger;
    HANDLE hEventAck;

    // Internal registry to track threads per control points 
    std::map<DWORD, std::unique_ptr<WorkerONNX>> activeWorkers;

    // Internal methods
    bool InitializeIPC();
    void HandleConfiguration();
    void HandleTermination();
};