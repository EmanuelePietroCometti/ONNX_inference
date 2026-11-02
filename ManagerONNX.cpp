#include "ManagerONNX.h"
#include "AsyncLogger.h"
#include <stdexcept>
#include <string>

// Writes the reason of a configuration failure into the result JSON of the slot, so the
// external program can show why the point was not configured. The text is reduced to
// printable ASCII without quotes/backslashes to keep the JSON valid.
void ManagerONNX::WriteSlotError(PTcontrolPoint pPoint, const char* what)
{
	std::string json = "{\"error\":\"";
	size_t n = 0;
	for (const char* c = what; c != nullptr && *c != 0 && n < config.GetGeneralConfigurationMaxText(); ++c, ++n)
	{
		const unsigned char ch = static_cast<unsigned char>(*c);
		json.push_back((ch < 0x20 || ch > 0x7E || ch == '"' || ch == '\\') ? ' ' : static_cast<char>(ch));
	}
	json += "\"}";

	constexpr size_t kJsonCapacity = sizeof(pPoint->results.json) / sizeof(WCHAR);
	const size_t len = json.size() < kJsonCapacity - 1 ? json.size() : kJsonCapacity - 1;
	for (size_t k = 0; k < len; ++k)
		pPoint->results.json[k] = static_cast<WCHAR>(static_cast<unsigned char>(json[k]));
	pPoint->results.json[len] = 0;
	pPoint->results.state = InferenceState::ERROR_DETECTED;
}

// Process shutdown

// Process shutdown
// One manual-reset event, created at startup before the manager. It can be set from any
// thread: console ctrl handler, "--stop" from another process, later the service handler.
HANDLE ManagerONNX::s_hShutdown = NULL;

ManagerONNX::ManagerONNX(const IniConfiguration& cfg) : config(cfg), hMapFile(NULL), pSharedList(NULL), hGlobalMutex(NULL), hEventTrigger(NULL), hEventAck(NULL)
{
	if (!InitializeIPC())
	{
		const DWORD err = GetLastError();   // da leggere prima della pulizia
		throw std::runtime_error(
			"Global IPC initialization failed (GetLastError=" + std::to_string(err) + ")");
	}
}

ManagerONNX::~ManagerONNX()
{
	if (pSharedList) UnmapViewOfFile(pSharedList);
	if (hMapFile) CloseHandle(hMapFile);
	if (hGlobalMutex) CloseHandle(hGlobalMutex);
	if (hEventTrigger) CloseHandle(hEventTrigger);
	if (hEventAck) CloseHandle(hEventAck);
}
void ManagerONNX::Run()
{
	if (s_hShutdown == NULL)
	{
		Log::Error("### ERROR: CreateShutdownEvent() must be called before Run()");
		return;
	}

	Log::Info("ONNX manager started. Waiting for signals...");

	// The manager runs until a shutdown is requested. 
	// QUIT from the external program only stops the workers: the process stays
	// up and waits for the next configuration.
	// The shutdown event is first in the array: if both are signaled it wins
	HANDLE waitHandles[2] = { s_hShutdown, hEventTrigger };

	bool isRunning = true;
	while(isRunning)
	{
		DWORD waitResult = WaitForMultipleObjects(2, waitHandles, FALSE, INFINITE);
		if (waitResult == WAIT_OBJECT_0)
		{
			isRunning = false;
			continue;
		}
		if (waitResult != WAIT_OBJECT_0 + 1)
		{
			// Never spin on a failing wait: this thread runs at real-time priority
			Log::Error("### ERROR: wait on shutdown/trigger failed (result {}, error {})", waitResult, GetLastError());
			Sleep(100);
			continue;
		}

		// WAIT_ABANDONED still grants ownership (previous owner crashed while
		// holding the mutex): it MUST be handled like WAIT_OBJECT_0, otherwise
		// the mutex is acquired here and never released, deadlocking both processes
		ListState requestedState = ListState::IDLE;
		DWORD waitMutex = WaitForSingleObject(hGlobalMutex, INFINITE);
		if (waitMutex == WAIT_OBJECT_0 || waitMutex == WAIT_ABANDONED) {
			requestedState = pSharedList->state;
			ReleaseMutex(hGlobalMutex);
		}
		else {
			Log::Error("### ERROR: wait on the global mutex failed (error {})", GetLastError());
			continue;
		}

		// Decide on the state read under the mutex, not on a second unprotected read
		switch (requestedState) {
		case ListState::UPDATE_PENDING:
			HandleConfiguration();
			break;
		case ListState::QUIT:
			HandleTermination();
			break;
		default:
			break;
		}
	}

	// Shutdown of the process: the external program is not waiting for an ack
	Log::Info(">> Process shutdown requested. Stopping workers...");
	activeWorkers.clear();
	Log::Info(">> Workers stopped. Leaving the main loop.");
}

bool ManagerONNX::CreateShutdownEvent()
{
	if (s_hShutdown != NULL)
		return true;

	s_hShutdown = CreateEventW(NULL, TRUE, FALSE, ONNX_SHUTDOWN_EVENT_NAME);
	if (s_hShutdown == NULL)
	{
		Log::Error("### ERROR: CreateEvent (shutdown) failed with code: {}", GetLastError());
		return false;
	}
	return true;
}

HANDLE ManagerONNX::ShutdownEventHandle()
{
	return s_hShutdown;
}

void ManagerONNX::RequestShutdown()
{
	if (s_hShutdown != NULL) SetEvent(s_hShutdown);
}

bool ManagerONNX::RequestShutdownOfRunningInstance()
{
	HANDLE h = OpenEventW(EVENT_MODIFY_STATE, FALSE, ONNX_SHUTDOWN_EVENT_NAME);
	if (h == NULL) return false;
	BOOL ok = SetEvent(h);
	CloseHandle(h);
	return ok != FALSE;
}

bool ManagerONNX::InitializeIPC()
{
	// Logs which step failed, releases whatever was opened so far (the destructor
	// does not run when the constructor throws) and leaves the real error code in
	// GetLastError() for the constructor.
	auto fail = [this](const char* step) {
		const DWORD err = GetLastError();   // read IMMEDIATELY after the failing call
		Log::Error("### ERROR: {} failed with code: {}", step, err);
		if (pSharedList) { UnmapViewOfFile(pSharedList); pSharedList = NULL; }
		if (hMapFile) { CloseHandle(hMapFile);        hMapFile = NULL; }
		if (hGlobalMutex) { CloseHandle(hGlobalMutex);    hGlobalMutex = NULL; }
		if (hEventTrigger) { CloseHandle(hEventTrigger);   hEventTrigger = NULL; }
		if (hEventAck) { CloseHandle(hEventAck);       hEventAck = NULL; }
		SetLastError(err);
		return false;
		};

	hGlobalMutex = CreateMutex(NULL, FALSE, IPC_LIST_MUTEX_NAME);
	if (!hGlobalMutex) return fail("CreateMutex");

	hEventTrigger = CreateEvent(NULL, FALSE, FALSE, IPC_LIST_TRIGGER_NAME);
	if (!hEventTrigger) return fail("CreateEvent (trigger)");

	hEventAck = CreateEvent(NULL, FALSE, FALSE, IPC_LIST_ACK_NAME);
	if (!hEventAck) return fail("CreateEvent (ack)");

	// Mapping global shared memory
	hMapFile = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, sizeof(controlPointsList), IPC_LIST_MMF_NAME);
	if (!hMapFile) return fail("CreateFileMapping");

	pSharedList = (PTcontrolPointsList)MapViewOfFile(hMapFile, FILE_MAP_WRITE, 0, 0, sizeof(controlPointsList));
	if (!pSharedList) return fail("MapViewOfFile");

	return true;
}

void ManagerONNX::HandleConfiguration()
{
	Log::Info(">> Configuration request received (UPDATE_PENDING)!");

	// The external program stopped sending frames before asking for the update and
	// does not touch the list until it receives the ack (it released LISTMUTEX before
	// signaling the trigger). Old workers go first: they release sessions, threads and
	// handles before the new ones are built
	activeWorkers.clear();

	bool configSuccess = true;

	DWORD count = pSharedList->numPunti;
	if (count > config.GetGeneralConfigurationMaxControlPoints()) {
		Log::Error("### ERROR: numPunti = {} exceeds the {} available slots", count, config.GetGeneralConfigurationMaxControlPoints());
		count = 0;
		configSuccess = false;
	}

	// A failing control point does not prevent the others from being configured:
	// the list state reports the failure, every point reports its own status
	for (DWORD i = 0; i < count; i++) {
		PTcontrolPoint pPoint = &pSharedList->points[i];
		const DWORD idPunto = pPoint->idPunto;

		try {
			if (activeWorkers.find(idPunto) != activeWorkers.end()) {
				throw std::runtime_error("duplicate idPunto in the list");
			}

			// Start a new worker for this point. Creation index + total count
			// determine the slice of physical cores owned by this worker's
			// session, so concurrent sessions never compete for a core
			auto worker = std::make_unique<WorkerONNX>(pPoint, i, count);
			worker->Start();

			// Set the state to CONFIGURED after the thread creation
			worker->MarkAsConfigured();
			activeWorkers[idPunto] = std::move(worker);
		}
		catch (const std::exception& e) {
			// The worker constructor released everything it had opened before throwing,
			// so there is no worker to mark: write the error state straight into the slot
			Log::Error("### ERROR DURING CONFIGURATION of control point {}: {}", idPunto, e.what());
			pPoint->status = PointState::ERROR_DETECTED;
			WriteSlotError(pPoint, e.what());
			configSuccess = false;
		}
	}

	DWORD waitMutex = WaitForSingleObject(hGlobalMutex, INFINITE);
	if (waitMutex == WAIT_OBJECT_0 || waitMutex == WAIT_ABANDONED) {
		pSharedList->state = configSuccess ? ListState::CONFIGURED : ListState::ERROR_DETECTED;
		ReleaseMutex(hGlobalMutex);
	}

	// Notify the external program that all threads started and configured correctly
	SetEvent(hEventAck);

	Log::Info(">> Configuration completed. Ack sent!");
}

void ManagerONNX::HandleTermination()
{
	Log::Info(">> Shutdown request received (QUIT)!");

	// Stop every worker. The process itself stays up, waiting for the next configuration
	activeWorkers.clear();

	// Send ack event
	SetEvent(hEventAck);
	Log::Info(">> Workers stopped. Ack sent, manager still running!");
}