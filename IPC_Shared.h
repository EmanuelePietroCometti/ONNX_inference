#pragma once
#include <windows.h>
#include <tchar.h>

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
    QUIT = 3
};

// Defines the type of AI task
enum class InferenceType : WORD {
    ANOMALY = 0,
    CLASSIFICATION = 1,
    OBJECT_DETECTION = 2
};

// STRUCTS
// Contains the output produced by the AI model
typedef struct resultInference {
    volatile InferenceState state;
    DWORD sizeX;
    DWORD sizeY;
    TCHAR json[1024];
} resultInference, * PTresultInference;

// Defines the configuration parameters for a single control point
typedef struct controlPoint {
    DWORD idPunto;
    DWORD sizeX;
    DWORD sizeY;
    DWORD bpp;
    TCHAR pathModello[512];
    TCHAR mutexName[128];
    TCHAR resultsEventName[128];
    TCHAR eventReadyName[128];
    PointState status;
    resultInference results;
    InferenceType inferenceType;
} controlPoint, * PTcontrolPoint;

// Manages the global list of control points for IPC
typedef struct controlPointsList {
    DWORD numPunti;
    ListState state;
    TCHAR listMutexName[128];
    TCHAR listEventTriggerName[128];
    TCHAR listEventAckName[128];
    controlPoint points[1024];
} controlPointsList, * PTcontrolPointsList;