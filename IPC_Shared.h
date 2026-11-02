#pragma once
#include <windows.h>
#include <cstddef>

#define IPC_NAME_PREFIX      L"Global\\"
#define IPC_NAME(n)          IPC_NAME_PREFIX n

#define IPC_LIST_MUTEX_NAME    IPC_NAME(L"LISTMUTEX")
#define IPC_LIST_TRIGGER_NAME  IPC_NAME(L"LISTEVENTTRIGGER")
#define IPC_LIST_ACK_NAME      IPC_NAME(L"LISTEVENTACK")
#define IPC_LIST_MMF_NAME      IPC_NAME(L"CONTROLPOINTLIST")

#pragma pack(push, 4)
// ENUMS
// Manages the state of the inference result
enum class InferenceState : WORD {
  IDLE = 0,
  RESULT_READY = 1,
  PENDING = 2,
  ERROR_DETECTED = 4
};

// Manages the operational status of the control point
enum class PointState : WORD {
  IDLE = 0,
  CONFIGURED = 1,
  UPDATE_PENDING = 2,
  QUIT = 3,
  ERROR_DETECTED = 4
};

// Manages the global synchronization state
enum class ListState : WORD {
  IDLE = 0,
  CONFIGURED = 1,
  UPDATE_PENDING = 2,
  QUIT = 3,
  ERROR_DETECTED = 4
};

// Defines the type of AI task
enum class InferenceType : WORD {
  NOT_DEFINED = 0,
  ANOMALY = 1,
  CLASSIFICATION = 2,
  OBJECT_DETECTION = 3
};

// STRUCTS
// Contains the output produced by the AI model
typedef struct resultInference {
  InferenceState state;
  DWORD sizeX;
  DWORD sizeY;
  WCHAR json[1024];
} resultInference, * PTresultInference;

// Defines the configuration parameters for a single control point
typedef struct controlPoint {
  DWORD idPunto;
  DWORD sizeX;
  DWORD sizeY;
  DWORD bpp;
  WCHAR pathModello[512];
  WCHAR mutexName[128];
  WCHAR resultsEventName[128];
  WCHAR eventReadyName[128];
  PointState status;
  resultInference results;
  InferenceType inferenceType;
} controlPoint, * PTcontrolPoint;

// Manages the global list of control points for IPC
typedef struct controlPointsList {
  DWORD numPunti;
  ListState state;
  WCHAR listMutexName[128];
  WCHAR listEventTriggerName[128];
  WCHAR listEventAckName[128];
  controlPoint points[1024];
} controlPointsList, * PTcontrolPointsList;

#pragma pack(pop)


static_assert(sizeof(WCHAR) == 2, "WCHAR must be 16 bit");
static_assert(sizeof(resultInference) == 2060, "resultInference layout changed");
static_assert(offsetof(resultInference, json) == 12, "resultInference::json offset");
static_assert(sizeof(controlPoint) == 3876, "controlPoint layout changed");
static_assert(offsetof(controlPoint, pathModello) == 16, "controlPoint::pathModello offset");
static_assert(offsetof(controlPoint, status) == 1808, "controlPoint::status offset");
static_assert(offsetof(controlPoint, results) == 1812, "controlPoint::results offset");
static_assert(offsetof(controlPoint, inferenceType) == 3872, "controlPoint::inferenceType offset");
static_assert(sizeof(controlPointsList) == 3969800, "controlPointsList layout changed");
static_assert(offsetof(controlPointsList, points) == 776, "controlPointsList::points offset");