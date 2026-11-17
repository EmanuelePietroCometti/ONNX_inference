#pragma once

#include <windows.h>
#include <fmt/format.h>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <utility>
#include "Configuration.h"

//
// AsyncLogger
// Dedicated logging subsystem for the real-time inference pipeline.
//
// Design goals (hard real-time producers):
//  - The hot path (Log call) NEVER blocks on I/O and NEVER allocates on the
//    heap: the message is formatted into a stack buffer, then memcpy'd into a
//    pre-allocated fixed-size slot of a bounded ring buffer under a mutex held
//    for a few tens of nanoseconds.
//  - If the queue is full the message is DROPPED (and counted), it never
//    stalls an inference thread: losing a log line is acceptable, missing a
//    frame deadline is not.
//  - A single background consumer thread (lowest priority, pinned to core 0,
//    away from the inference cores) drains the queue periodically and performs
//    the actual console I/O.
//
// Usage:
//   AsyncLogger::Instance().Start();          // once, at process startup
//   Log::Info("Worker {} ready", id);         // from any thread
//   Log::Error("failure: {}", what);
//   AsyncLogger::Instance().Shutdown();       // once, at exit (drains queue)
//

enum class LogLevel : uint8_t { Info = 0, Warning = 1, Error = 2 };

class AsyncLogger {
public:
	static AsyncLogger& Instance();

	void Start(const IniConfiguration& cfg);
	void Shutdown();

	template <typename... Args>
	void Log(LogLevel level, fmt::format_string<Args...> fmtStr, Args&&... args)
	{
		// Below the configured verbosity: bail out before any formatting or locking
		if (static_cast<WORD>(level) < config.GetLoggerVerbosity())
			return;
		const size_t maxChars = max_message_chars_;
		if (maxChars == 0)
			return;

		// Allocate runtime-sized buffer directly on the caller's stack frame (zero heap overhead)
		char* local = static_cast<char*>(_alloca(maxChars));

		const auto out = fmt::format_to_n(local, maxChars, fmtStr, std::forward<Args>(args)...);
		const size_t written = (std::min)(static_cast<size_t>(out.size), maxChars);

		Enqueue(level, local, static_cast<uint16_t>(written));
	}

	AsyncLogger(const AsyncLogger&) = delete;
	AsyncLogger& operator=(const AsyncLogger&) = delete;

private:
	explicit AsyncLogger(const IniConfiguration& cfg) : config(cfg) {};
	~AsyncLogger();

	struct Slot {
		SYSTEMTIME ts;
		LogLevel   level;
		uint16_t   length;
		char* text; // Points into the pre-allocated contiguous text_pool_
	};

	void Enqueue(LogLevel level, const char* msg, uint16_t length);
	void ConsumerLoop();
	// Drains and prints every queued slot. Returns the number of slots printed.
	size_t DrainQueue();

	// Runtime configuration dimensions loaded from .ini in Start()
	size_t max_message_chars_ = 0;
	size_t queue_capacity_ = 0;

	// Pre-allocated once in Start() - no heap allocations during Log()
	std::unique_ptr<char[]> text_pool_;

	// Bounded ring buffer, allocated once with the singleton (no heap churn)
	std::unique_ptr<Slot[]> ring_;
	size_t head_ = 0;   // next write position
	size_t tail_ = 0;   // next read position
	size_t count_ = 0;  // slots currently queued

	std::mutex mutex_;
	std::condition_variable cv_;
	std::thread consumer_;
	std::atomic<bool> running_{ false };
	std::atomic<uint64_t> dropped_{ 0 };

	const IniConfiguration& config;
};

// Convenience free functions: every subsystem logs through these instead of
// printing to the console directly
namespace Log {

	template <typename... Args>
	inline void Info(fmt::format_string<Args...> fmtStr, Args&&... args)
	{
		AsyncLogger::Instance().Log(LogLevel::Info, fmtStr, std::forward<Args>(args)...);
	}

	template <typename... Args>
	inline void Warning(fmt::format_string<Args...> fmtStr, Args&&... args)
	{
		AsyncLogger::Instance().Log(LogLevel::Warning, fmtStr, std::forward<Args>(args)...);
	}

	template <typename... Args>
	inline void Error(fmt::format_string<Args...> fmtStr, Args&&... args)
	{
		AsyncLogger::Instance().Log(LogLevel::Error, fmtStr, std::forward<Args>(args)...);
	}

} // namespace Log
