#include "ManagerONNX.h"
#include "AsyncLogger.h"
#include "RealTimeConfig.h"
#include "Configuration.h"

#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <thread>

// Event setted at the end of the main: console manager return only after this command
static HANDLE g_hMainDone = NULL;

static constexpr const WCHAR* iniPath = L"ONNXInference_configuration.ini";

// INI next to the executable (a bare file name would be searched in C:\Windows)
static std::filesystem::path IniFilePath()
{
  WCHAR buf[MAX_PATH];
  const DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
  const std::filesystem::path dir = (n > 0 && n < MAX_PATH)
    ? std::filesystem::path(buf).parent_path()
    : std::filesystem::current_path();
  return dir / L"ONNXInference_configuration.ini";
}

// Keyboard thread: wait for 'q' key or for a shutdown request
static void KeyboardThread()
{
  SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);

  HANDLE hIn = GetStdHandle(STD_INPUT_HANDLE);
  DWORD mode = 0;
  if (hIn == INVALID_HANDLE_VALUE || !GetConsoleMode(hIn, &mode))
    return;

  HANDLE waitHandles[2] = { ManagerONNX::ShutdownEventHandle(), hIn };
  while(true)
  {
    DWORD w = WaitForMultipleObjects(2, waitHandles, FALSE, INFINITE);
    if (w != WAIT_OBJECT_0 + 1)
      return; // shutdown already requested

    INPUT_RECORD rec;
    DWORD n = 0;
    if (!ReadConsoleInput(hIn, &rec, 1, &n) || n == 0)
      continue;
    if (rec.EventType == KEY_EVENT && rec.Event.KeyEvent.bKeyDown)
    {
      char c = rec.Event.KeyEvent.uChar.AsciiChar;
      if (c == 'q' || c == 'Q')
      {
        Log::Info("Key 'q' pressed: stopping the inference program");
        ManagerONNX::RequestShutdown();
        return;
      }
    }
  }
}

// Ctrl+C, window closing, logoff, PC shutdown
static BOOL WINAPI ConsoleCtrlHandler(DWORD ctrlType)
{
  switch (ctrlType)
  {
  case CTRL_C_EVENT:
  case CTRL_BREAK_EVENT:
  case CTRL_CLOSE_EVENT:
  case CTRL_LOGOFF_EVENT:
  case CTRL_SHUTDOWN_EVENT:
    ManagerONNX::RequestShutdown();
    WaitForSingleObject(g_hMainDone, 5000);
    return TRUE;
  default:
    return FALSE;
  }
}

int main(int argc, char** argv)
{
  if (argc > 1 && strcmp(argv[1], "--stop") == 0)
  {
    if (ManagerONNX::RequestShutdownOfRunningInstance())
    {
      Log::Info("Shutdown requested.\n");
      return 0;
    }
    Log::Info("No running instance found.\n");
    return 1;
  }

  IniConfiguration& config = IniConfiguration::Instance();
  config.Load(IniFilePath().c_str());

  // Logger first: from here on every subsystem logs through the async queue
  // and the console I/O happens on a low-priority thread pinned to core 0
  AsyncLogger::Instance().Start(config);

  // Shutdown event and console manager
  g_hMainDone = CreateEvent(NULL, TRUE, FALSE, NULL);
  if (!ManagerONNX::CreateShutdownEvent())
  {
    AsyncLogger::Instance().Shutdown();
    return 2;
  }
  SetConsoleCtrlHandler(ConsoleCtrlHandler, TRUE);

  // REALTIME priority class (HIGH fallback when not elevated) + 1 ms timer
  // resolution. Must happen before any worker thread is created.
  RT::EnableRealTimeProcess();

  // The main thread runs the manager loop: keep it on the reserved core,
  // away from the inference slices
  RT::ConfigureControlThread();

  Log::Info("ONNX_inference running. Press 'q' to stop the program.\n");
  std::thread keyboard(KeyboardThread);

  try {
    ManagerONNX engine(config);
    engine.Run();         // return only on shutdown request (q, Ctrl+C, finestra, --stop)
  }
  catch (const std::exception& e)
  {
    Log::Error("Exception occurred: {}", e.what());
    ManagerONNX::RequestShutdown();     // Notify the exit to the keyboard thread
  }

  if (keyboard.joinable())
    keyboard.join();

  RT::DisableRealTimeProcess();

  Log::Info("ONNX manager correctly terminated.");

  // Drains every queued message before exiting: nothing is lost
  AsyncLogger::Instance().Shutdown();

  SetEvent(g_hMainDone);  // Free the console manager
  return 0;
}