#pragma once

#include "IPC_Shared.h"
#include "WorkerONNX.h"
#include "Configuration.h"

#include <Windows.h>
#include <map>
#include <memory>

// Named event that asks the whole process to stop (same prefix choice as the other IPC objects)
#define ONNX_SHUTDOWN_EVENT_NAME L"Global\\ONNX_INFERENCE_SHUTDOWN"

class ManagerONNX {
public:
    ManagerONNX(const IniConfiguration& cfg);
    ~ManagerONNX();

    // Main loop that waits for trigger
    void Run();

    // to be called once at startup, before costructing the manager
    static bool CreateShutdownEvent();
    // handle of the shutdown event
    static HANDLE ShutdownEventHandle();
    // asks the running manager to leave Run(): console crtl handler, later the service handler
    static void RequestShutdown();
    // from another process
    static bool RequestShutdownOfRunningInstance();
private:
    // IPC global handles
    HANDLE hMapFile;
    PTcontrolPointsList pSharedList;
    HANDLE hGlobalMutex;
    HANDLE hEventTrigger;
    HANDLE hEventAck;

    static HANDLE s_hShutdown;

    const IniConfiguration& config;

    // Internal registry to track threads per control points 
    std::map<DWORD, std::unique_ptr<WorkerONNX>> activeWorkers;

    // Internal methods
    bool InitializeIPC();
    void HandleConfiguration();
    void HandleTermination();
    void WriteSlotError(PTcontrolPoint pPoint, const char* what);
};