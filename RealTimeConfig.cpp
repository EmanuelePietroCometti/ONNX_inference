#include "RealTimeConfig.h"
#include "AsyncLogger.h"

#include <windows.h>
#include <timeapi.h>
#include <algorithm>
#include <thread>

#pragma comment(lib, "winmm.lib")

namespace {

struct CoreInfo {
    unsigned lp;
    BYTE efficiencyClass;
};

struct CoreLayout {
    std::vector<unsigned> compute;
    unsigned reserved;
};

// Enumerates the PHYSICAL cores this process is actually allowed to use and
// returns one 0-based logical processor id per core.
//
// Two filters are applied:
//  - Physical topology (GetLogicalProcessorInformationEx): one logical
//    processor per core. On the production target (Codesys machine) SMT/
//    hyper-threading is disabled in the BIOS, so core == logical processor
//    and this is a no-op; the code stays correct on HT-enabled dev machines.
//  - Process affinity mask (GetProcessAffinityMask): the Codesys runtime
//    reserves whole cores for the PLC and removes them from the mask. Any
//    core outside the mask MUST be skipped: SetThreadAffinityMask onto it
//    would fail, and contending it would disturb the PLC cycle.
//
// NOTE: limited to processor group 0, i.e. at most 64 logical processors;
// beyond that SetThreadGroupAffinity and group-aware masks are required.
std::vector<CoreInfo> AvailableComputeCores()
{
    std::vector<CoreInfo> cores;

    DWORD_PTR processMask = 0;
    DWORD_PTR systemMask = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &processMask, &systemMask) || processMask == 0) {
        processMask = ~static_cast<DWORD_PTR>(0); // query failed: assume all allowed
    }

    DWORD length = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length);
    if (GetLastError() == ERROR_INSUFFICIENT_BUFFER && length > 0) {
        std::vector<char> buffer(length);
        auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data());
        if (GetLogicalProcessorInformationEx(RelationProcessorCore, info, &length)) {
            for (char* p = buffer.data(); p < buffer.data() + length;) {
                auto* rec = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(p);
                if (rec->Relationship == RelationProcessorCore && rec->Processor.GroupCount >= 1
                    && rec->Processor.GroupMask[0].Group == 0) {
                    // Only the part of the core usable by THIS process: a core
                    // fully reserved (e.g. by the Codesys runtime) drops out here
                    KAFFINITY mask = rec->Processor.GroupMask[0].Mask & processMask;
                    unsigned bit = 0;
                    while (mask != 0 && (mask & 1) == 0) { mask >>= 1; ++bit; }
                    if (mask != 0 && bit < 64) {
                        cores.push_back({bit, rec->Processor.EfficiencyClass});
                    }
                }
                p += rec->Size;
            }
        }
    }

    if (cores.empty()) {
        // Fallback: no topology info, treat every allowed logical processor as
        // a core (exact on the HT-disabled production target)
        const unsigned logical = (std::max)(1u, std::thread::hardware_concurrency());
        for (unsigned i = 0; i < logical && i < 64; ++i) {
            if ((processMask >> i) & 1) {
                cores.push_back({i, 0});
            }
        }
    }

    std::sort(cores.begin(), cores.end(), [](const CoreInfo& a, const CoreInfo& b) {return a.lp < b.lp; });
    return cores;
}

CoreLayout ClassifyCores() {
    const std::vector<CoreInfo> cores = AvailableComputeCores();
    CoreLayout layout{ {}, 0 };

    if (cores.empty()) { return layout; }

    BYTE maxClass = 0;

    for (const CoreInfo& c : cores) maxClass = (std::max)(maxClass, c.efficiencyClass);

    std::vector<unsigned> eCores;
    for (const CoreInfo& c : cores) {
        if (c.efficiencyClass == maxClass) layout.compute.push_back(c.lp);
        else eCores.push_back(c.lp);
    }

    if (!eCores.empty()) {
        layout.reserved = eCores.front();
    }
    else {
        layout.reserved = layout.reserved = layout.compute.front();
        if (layout.compute.size() > 1) layout.compute.erase(layout.compute.begin());
    }
}

} // namespace

namespace RT {

bool EnableRealTimeProcess()
{
    // 1 ms scheduler/timer granularity: without this, event wake-ups and any
    // timed wait quantize to ~15.6 ms, which alone breaks a 4-5 ms budget
    timeBeginPeriod(1);

    HANDLE hProcess = GetCurrentProcess();

    // REALTIME_PRIORITY_CLASS puts the process above every normal system
    // activity. Windows only grants it to elevated processes: when the
    // privilege is missing the call "succeeds" but the class is silently
    // downgraded, so the actual class must be read back and verified.
    SetPriorityClass(hProcess, REALTIME_PRIORITY_CLASS);
    const DWORD granted = GetPriorityClass(hProcess);

    if (granted == REALTIME_PRIORITY_CLASS) {
        Log::Info("[RT] Process priority class: REALTIME");
        return true;
    }

    if (!SetPriorityClass(hProcess, HIGH_PRIORITY_CLASS)) {
        Log::Error("[RT] SetPriorityClass(HIGH) failed with code {}", GetLastError());
        return false;
    }
    Log::Warning("[RT] REALTIME class not granted (run elevated to enable it). "
        "Falling back to HIGH priority class.");
    return false;
}

void DisableRealTimeProcess()
{
    timeEndPeriod(1);
}

unsigned PhysicalCoreCount()
{
    return static_cast<unsigned>(AvailableComputeCores().size());
}

CpuPartition ComputeCpuPartition(unsigned workerIndex, unsigned workerCount)
{
    CpuPartition part;
    if (workerCount == 0) workerCount = 1;

    const CoreLayout layout = ClassifyCores();
    const std::vector<unsigned>& cores = layout.compute;
    const unsigned available = static_cast<unsigned>(cores.size());

    if (workerCount >= available) {
        // More workers than compute cores: give each worker one core, wrapping
        // around. Slices overlap, so downstream tuning must stay conservative.
        part.logicalProcessors = cores;
        part.shared = true;
        if (part.shared) {
            Log::Warning("[RT] {} workers on {} P-core(s): worker {} run single-thread {} "
                "on the SHARED P-core set (first lp {})",
                workerCount, available, workerIndex, cores.front());
        }
        return part;
    }

    // Contiguous disjoint slices; the remainder cores go to the first workers
    const unsigned base = available / workerCount;
    const unsigned remainder = available % workerCount;
    const unsigned begin = workerIndex * base + (std::min)(workerIndex, remainder);
    const unsigned count = base + (workerIndex < remainder ? 1u : 0u);

    for (unsigned k = 0; k < count; ++k) {
        part.logicalProcessors.push_back(cores[begin + k]);
    }

    Log::Info("[RT] Worker {} owns {} P-core(s): first logical processor {}",
        workerIndex, count, part.logicalProcessors.front());
    return part;
}

void ConfigureInferenceThread(unsigned workerIndex, const CpuPartition& partition)
{
    HANDLE hThread = GetCurrentThread();

    // TIME_CRITICAL: base priority 15 in HIGH class, 31 in REALTIME class.
    // The worker must preempt everything else the moment the frame event fires.
    if (!SetThreadPriority(hThread, THREAD_PRIORITY_TIME_CRITICAL)) {
        Log::Error("[RT] Worker {}: SetThreadPriority(TIME_CRITICAL) failed with code {}",
            workerIndex, GetLastError());
    }

    if (partition.logicalProcessors.empty()) {
        Log::Warning("[RT] Worker {}: empty CPU partition, affinity pinning skipped", workerIndex);
        return;
    }

    // The worker runs on the FIRST core of its slice; the ORT pool threads of
    // its session are pinned to the remaining slice cores (OrtsessionConfig)
    const unsigned core = partition.logicalProcessors.front();
    DWORD_PTR mask = 0;
    if (partition.shared) {
        for (unsigned c : partition.logicalProcessors) mask |= static_cast<DWORD_PTR>(1);
    }
    if (SetThreadAffinityMask(hThread, mask) == 0) {
        Log::Error("[RT] Worker {}: SetThreadAffinityMask(core {}) failed with code {}",
            workerIndex, core, GetLastError());
    }
    else {
        Log::Info("[RT] Worker {}: TIME_CRITICAL, {} ({} core(s) {})",
            workerIndex, 
            partition.shared ? "flaoting on shared P set" : "pinned to slice core",
            partition.logicalProcessors.size(),
            partition.shared ? ", SHARED" : ""
        );
    }
}

void ConfigureBackgroundThread()
{
    HANDLE hThread = GetCurrentThread();

    // Background service work (console I/O) runs at the lowest priority on the
    // reserved core: it only executes when the compute cores have nothing
    // pending. The reserved core is the first AVAILABLE one, not hardcoded 0
    // (which may belong to the Codesys runtime and be outside our affinity mask).
    SetThreadPriority(hThread, THREAD_PRIORITY_LOWEST);

    const CoreLayout layout = ClassifyCores();
    if (layout.compute.size() > 2) {
        if (SetThreadAffinityMask(GetCurrentThread(), static_cast<DWORD_PTR>(1) << layout.reserved) != 0) {
            Log::Info("[RT] Control thread pinned to reserved core {}", layout.reserved);
        }
    }
}

void ConfigureControlThread()
{
    // Default priority is kept: the manager only waits on IPC events. Pinning
    // it to the reserved core guarantees it is never scheduled onto a compute
    // core in the middle of a frame.
    const CoreLayout layout = ClassifyCores();
    if (layout.compute.size() > 2) {
        if (SetThreadAffinityMask(GetCurrentThread(), static_cast<DWORD_PTR>(1) << layout.reserved) != 0) {
            Log::Info("[RT] Control thread pinned to reserved core {}", layout.reserved);
        }
    }
}

} // namespace RT
