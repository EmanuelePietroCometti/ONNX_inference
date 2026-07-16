// ONNX_inference.cpp : Questo file contiene la funzione 'main', in cui inizia e termina l'esecuzione del programma.
//

#include "ManagerONNX.h"
#include "AsyncLogger.h"
#include "RealTimeConfig.h"

int main()
{
    // Logger first: from here on every subsystem logs through the async queue
    // and the console I/O happens on a low-priority thread pinned to core 0
    AsyncLogger::Instance().Start();

    // REALTIME priority class (HIGH fallback when not elevated) + 1 ms timer
    // resolution. Must happen before any worker thread is created.
    RT::EnableRealTimeProcess();

    // The main thread runs the manager loop: keep it on the reserved core,
    // away from the inference slices
    RT::ConfigureControlThread();

    try {
        ManagerONNX engine;
        engine.Run();
    }
    catch (const std::exception& e)
    {
        Log::Error("Exception occurred: {}", e.what());
    }

    RT::DisableRealTimeProcess();

    // Drains every queued message before exiting: nothing is lost
    AsyncLogger::Instance().Shutdown();
}
