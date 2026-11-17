#include "AsyncLogger.h"
#include "RealTimeConfig.h"

#include <fmt/core.h>
#include <fmt/color.h>
#include <chrono>
#include <cstdio>
#include <cstring>

AsyncLogger& AsyncLogger::Instance()
{
	static AsyncLogger instance(IniConfiguration::Instance());
	return instance;
}

AsyncLogger::~AsyncLogger()
{
	Shutdown();
}

// fmt emette i colori come sequenze ANSI (\x1b[38;2;...m): il conhost classico
// non le interpreta finche' non si attiva ENABLE_VIRTUAL_TERMINAL_PROCESSING,
// e le stampa letteralmente come "<-[38;2;...m". Se l'attivazione fallisce
// (console troppo vecchia) i colori vengono semplicemente disabilitati.
static bool s_ansiColorSupported = false;

static void EnableVTProcessing()
{
	s_ansiColorSupported = true;
	for (DWORD stdHandle : { STD_OUTPUT_HANDLE, STD_ERROR_HANDLE }) {
		HANDLE h = GetStdHandle(stdHandle);
		DWORD mode = 0;
		if (h == NULL || h == INVALID_HANDLE_VALUE || !GetConsoleMode(h, &mode)) {
			continue;   // stream redirezionato su file/pipe: non e' una console
		}
		if (!SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING)) {
			s_ansiColorSupported = false;
		}
	}
}

void AsyncLogger::Start(const IniConfiguration& cfg)
{
	bool expected = false;
	if (!running_.compare_exchange_strong(expected, true))
		return;

	// Read runtime dimensions from the .ini configuration singleton
	queue_capacity_ = std::clamp<size_t>(config.GetLoggerQueueCapacity(), 1, 1'000'000);
	max_message_chars_ = std::clamp<size_t>(config.GetLoggerMaxMessagesChars(), 16, 4096);

	// Allocate ring slots and contiguous character pool ONCE at startup
	ring_ = std::make_unique<Slot[]>(queue_capacity_);
	text_pool_ = std::make_unique<char[]>(queue_capacity_ * max_message_chars_);

	for (size_t i = 0; i < queue_capacity_; ++i)
	{
		ring_[i].text = &text_pool_[i * max_message_chars_];
		ring_[i].length = 0;
	}

	head_ = tail_ = count_ = 0;
	consumer_ = std::thread(&AsyncLogger::ConsumerLoop, this);
}

void AsyncLogger::Enqueue(LogLevel level, const char* msg, uint16_t length)
{
	SYSTEMTIME now;
	GetLocalTime(&now);

	{
		std::lock_guard<std::mutex> lock(mutex_);
		if (count_ == queue_capacity_)
		{
			dropped_.fetch_add(1, std::memory_order_relaxed);
			return;
		}

		Slot& slot = ring_[head_];
		slot.ts = now;
		slot.level = level;
		slot.length = length;
		std::memcpy(slot.text, msg, length);

		head_ = (head_ + 1) % queue_capacity_;
		++count_;
	}
}

void AsyncLogger::Shutdown()
{
	bool expected = true;
	if (!running_.compare_exchange_strong(expected, false)) {
		return; // never started or already stopped
	}
	cv_.notify_one();
	if (consumer_.joinable()) {
		consumer_.join();
	}
	// Final drain: nothing queued after Shutdown returns is lost
	DrainQueue();
	std::fflush(stdout);
}

size_t AsyncLogger::DrainQueue()
{
	size_t printed = 0;
	const WORD verbosity = config.GetLoggerVerbosity();
	for (;;) {
		Slot local;
		{
			std::lock_guard<std::mutex> lock(mutex_);
			if (count_ == 0) break;
			// Copy ONE slot out under the lock (~half a microsecond), then
			// print with the lock released so producers are never stalled by I/O
			local = ring_[tail_];
			tail_ = (tail_ + 1) % config.GetLoggerQueueCapacity();
			--count_;
		}

		if (static_cast<WORD>(local.level) < verbosity)
			continue;

		const fmt::string_view text(local.text, local.length);
		switch (local.level) {
		case LogLevel::Error:
			fmt::print(stderr, s_ansiColorSupported
				? fg(fmt::color::red) | fmt::emphasis::bold : fmt::text_style{},
				"{:02}:{:02}:{:02}.{:03} [ERR ] {}\n",
				local.ts.wHour, local.ts.wMinute, local.ts.wSecond, local.ts.wMilliseconds, text);
			break;
		case LogLevel::Warning:
			fmt::print(s_ansiColorSupported
				? fg(fmt::color::yellow) : fmt::text_style{},
				"{:02}:{:02}:{:02}.{:03} [WARN] {}\n",
				local.ts.wHour, local.ts.wMinute, local.ts.wSecond, local.ts.wMilliseconds, text);
			break;
		default:
			fmt::print("{:02}:{:02}:{:02}.{:03} [INFO] {}\n",
				local.ts.wHour, local.ts.wMinute, local.ts.wSecond, local.ts.wMilliseconds, text);
			break;
		}
		++printed;
	}

	const uint64_t dropped = dropped_.exchange(0, std::memory_order_relaxed);
	if (dropped > 0) {
		fmt::print(stderr, s_ansiColorSupported
			? fg(fmt::color::red) | fmt::emphasis::bold : fmt::text_style{},
			"[LOGGER] {} messages dropped (queue full)\n", dropped);
	}
	return printed;
}

void AsyncLogger::ConsumerLoop()
{
	// Lowest priority, pinned to core 0: console I/O must never steal CPU time
	// from the TIME_CRITICAL inference threads on the dedicated cores
	RT::ConfigureBackgroundThread();

	while (running_.load(std::memory_order_relaxed)) {
		{
			std::unique_lock<std::mutex> lock(mutex_);
			cv_.wait_for(lock, std::chrono::milliseconds(config.GetLoggerFlushIntervalMs()),
				[this] { return count_ >= config.GetLoggerNotifyThreshold() || !running_.load(std::memory_order_relaxed); });
		}
		DrainQueue();
	}
	// Shutdown drains once more after the join, nothing is lost here
}
