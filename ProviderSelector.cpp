#include "ProviderSelector.h"
#include "OrtRuntime.h"
#include "Configuration.h"
#include "AsyncLogger.h"
#include <onnxruntime_session_options_config_keys.h>
#include <windows.h>
#include <process.h>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(ORT_EP_GPU)
#include <cuda_runtime.h>
#endif

namespace
{
	// Custom thread factory for the ORT intra-op pool.
	//
	// On the CPU EP the convolutions run on the intra-op pool threads, NOT on the
	// worker thread that calls Run(). Boosting only the worker would leave the
	// threads doing the actual compute at NORMAL priority. This factory creates
	// the pool threads and raises each one to TIME_CRITICAL before it starts.
	struct OrtThreadCtx {
		OrtThreadWorkerFn fn;
		void* param;
	};

	unsigned __stdcall OrtRtThreadEntry(void* arg)
	{
		OrtThreadCtx* ctx = static_cast<OrtThreadCtx*>(arg);
		const OrtThreadCtx local = *ctx;
		delete ctx;

		SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
		local.fn(local.param);
		return 0;
	}

	OrtCustomThreadHandle OrtRtCreateThread(void* /*options*/, OrtThreadWorkerFn fn, void* param)
	{
		OrtThreadCtx* ctx = new OrtThreadCtx;
		ctx->fn = fn;
		ctx->param = param;

		const uintptr_t handle = _beginthreadex(nullptr, 0, &OrtRtThreadEntry, ctx, 0, nullptr);
		if (handle == 0) {
			delete ctx;
			return nullptr;
		}
		return reinterpret_cast<OrtCustomThreadHandle>(handle);
	}

	void OrtRtJoinThread(OrtCustomThreadHandle handle)
	{
		HANDLE h = (HANDLE)handle;
		WaitForSingleObject(h, INFINITE);
		CloseHandle(h);
	}

#if defined(ORT_EP_OPENVINO)
	// Reads a text file and strips any UTF-8 BOM (the JSON parser does not accept it).
	// Returns an empty string if the file cannot be opened.
	std::string ReadTextFile(const std::filesystem::path& path)
	{
		std::ifstream file(path, std::ios::binary);
		if (!file)
			return std::string();

		std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
		if (text.size() >= 3 &&
			static_cast<unsigned char>(text[0]) == 0xEF &&
			static_cast<unsigned char>(text[1]) == 0xBB &&
			static_cast<unsigned char>(text[2]) == 0xBF)
			text.erase(0, 3);
		return text;
	}
#endif

#if defined(ORT_EP_GPU)
	// IniConfiguration exposes WCHAR*, ORT option values are narrow UTF-8 strings.
	std::string WideToUtf8(const WCHAR* wide)
	{
		if (wide == nullptr || *wide == L'\0')
			return std::string();

		const int length = WideCharToMultiByte(CP_UTF8, 0, wide, -1, nullptr, 0, nullptr, nullptr);
		if (length <= 1)
			return std::string();

		std::string text(static_cast<size_t>(length - 1), '\0');
		WideCharToMultiByte(CP_UTF8, 0, wide, -1, &text[0], length, nullptr, nullptr);
		return text;
	}

	// Validates the ini 'device' against the CUDA devices actually present.
	// Returns a valid CUDA ordinal; throws if there is no CUDA device at all.
	int ResolveCudaDeviceIndex(const IniConfiguration& ini, const std::string& tag)
	{
		int count = 0;
		const cudaError_t err = cudaGetDeviceCount(&count);
		if (err != cudaSuccess || count <= 0)
			throw std::runtime_error(std::string("no CUDA device available: ") + cudaGetErrorString(err));

		const int wanted = static_cast<int>(ini.GetGeneralConfigurationGpuDevice());
		if (wanted < count)
			return wanted;

		Log::Warning("[{}] ini device={} is not valid ({} CUDA device(s) found): using device 0.", tag, wanted, count);
		return 0;
	}

	// Options shared by the classic and the plugin path (everything except the device).
	std::unordered_map<std::string, std::string> CudaProviderOptions(const IniConfiguration& ini)
	{
		std::unordered_map<std::string, std::string> opts;
		opts["cudnn_conv_use_max_workspace"] = std::to_string(ini.GetCudaConvUseMaxWorkspace());
		opts["do_copy_in_default_stream"] = ini.GetCudaDoCopyInDefaultStream() ? "1" : "0";
		opts["cudnn_conv1d_pad_to_nc1d"] = ini.GetCudaCudnnConv1dPadToNc1d() ? "1" : "0";

		if (ini.GetCudaGpuMemLimitMB() > 0) {   // 0 = no limit (ORT default)
			const unsigned long long bytes = static_cast<unsigned long long>(ini.GetCudaGpuMemLimitMB()) * 1024ULL * 1024ULL;
			opts["gpu_mem_limit"] = std::to_string(bytes);
		}

		const std::string arena = WideToUtf8(ini.GetCudaArenaExtendStrategy());
		const std::string algo = WideToUtf8(ini.GetCudaCudnnConvAlgoSearch());
		if (!arena.empty())
			opts["arena_extend_strategy"] = arena;
		if (!algo.empty())
			opts["cudnn_conv_algo_search"] = algo;
		return opts;
	}
	bool ParseEpSelectionPolicy(const std::string& name, OrtExecutionProviderDevicePolicy& policy)
	{
		if (name == "PREFER_GPU")
			policy = OrtExecutionProviderDevicePolicy_PREFER_GPU;
		else if (name == "PREFER_CPU")
			policy = OrtExecutionProviderDevicePolicy_PREFER_CPU;
		else if (name == "PREFER_NPU")
			policy = OrtExecutionProviderDevicePolicy_PREFER_NPU;
		else if (name == "MAX_PERFORMANCE")
			policy = OrtExecutionProviderDevicePolicy_MAX_PERFORMANCE;
		else if (name == "MAX_EFFICIENCY")
			policy = OrtExecutionProviderDevicePolicy_MAX_EFFICIENCY;
		else if (name == "MIN_OVERALL_POWER")
			policy = OrtExecutionProviderDevicePolicy_MIN_OVERALL_POWER;
		else
			return false;
		return true;
	}
#endif
}

ProviderSelector::ProviderSelector(OrtRuntime& runtime, const std::string& tag) :
	m_runtime(runtime),
	m_tag(tag),
	m_usedEpDevice(false)
{
}

ExecutionProvider ProviderSelector::Configure(Ort::SessionOptions& so, const RT::CpuPartition& cpuPartition, ExecutionProvider requested)
{
	ExecutionProvider active = ExecutionProvider::CPU;
	m_usedEpDevice = false;

#if defined(ORT_EP_GPU)
	so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

	if (requested == ExecutionProvider::TensorRT)
	{
		active = AppendTensorRT(so);
	}
	else if (requested == ExecutionProvider::CUDA)
	{
		active = AppendCuda(so);
	}
	else if (requested == ExecutionProvider::CPU)
	{
		Log::Info("[{}] CPU execution provider requested: native ONNX Runtime CPU.", m_tag);
	}
	else
	{
		Log::Error("[{}] Invalid execution provider requested: {}. With this release you can use CPU, CUDA or TensorRT.", m_tag, static_cast<int>(requested));
		throw std::runtime_error("Invalid execution provider requested: " + std::to_string(static_cast<int>(requested)));
	}
#elif defined(ORT_EP_OPENVINO)
	// OpenVINO recompiles the graph internally: ORT-side fusions are wasted work.
	so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
	active = AppendOpenVINO(so, requested);
#elif defined(ORT_EP_CPU)
	so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	if (requested != ExecutionProvider::CPU)
	{
		Log::Error("[{}] ### CPU-only build: requested execution provider {} ignored, the session runs on CPU.", m_tag, static_cast<int>(requested));
	}
#endif

	ConfigureThreading(so, cpuPartition, active);
	return active;
}

void ProviderSelector::ConfigureThreading(Ort::SessionOptions& so, const RT::CpuPartition& cpuPartition,
	ExecutionProvider active)
{
	// The accelerator runs the heavy compute on its own threads/streams;
	// a single ORT intra-op thread just feeds it and avoids context switching.
	if (active == ExecutionProvider::CUDA || active == ExecutionProvider::TensorRT) {
		so.SetIntraOpNumThreads(1);
		return;
	}

	// CPU path (native CPU and OpenVINO): the ORT intra-op pool runs the convolutions.

	// Threads available to THIS session: its core slice when partitioned, otherwise
	// all usable physical cores minus the reserved one. The topology comes from RT:
	// hardware_concurrency()/2 would be wrong on the production target, where
	// hyper-threading is disabled.
	unsigned sliceThreads;
	if (cpuPartition.logicalProcessors.empty()) {
		const unsigned computeCores = RT::PhysicalCoreCount();
		sliceThreads = computeCores > 1 ? computeCores - 1 : 1;
	}
	else {
		sliceThreads = static_cast<unsigned>(cpuPartition.logicalProcessors.size());
	}

	so.EnableCpuMemArena();                                     // avoid per-run OS allocations
	so.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);         // no inter-op locking overhead
	so.SetIntraOpNumThreads(static_cast<int>(sliceThreads));    // this session's core slice

	// Flush denormals to zero ON THE ORT WORKER THREADS. Setting MXCSR by hand only
	// affects the calling thread, not the intra-op pool that runs the conv kernels.
	so.AddConfigEntry(kOrtSessionOptionsConfigSetDenormalAsZero, "1");

	// Pin the pool threads inside this session's core slice. ORT expects
	// intra_op_num_threads - 1 entries: the calling thread (our worker, pinned to
	// slice core 0) is the pool's implicit first member. ORT processor ids are 1-BASED.
	if (cpuPartition.logicalProcessors.size() > 1) {
		std::string affinity;
		for (size_t t = 1; t < cpuPartition.logicalProcessors.size(); ++t) {
			if (!affinity.empty())
				affinity += ';';
			affinity += std::to_string(cpuPartition.logicalProcessors[t] + 1);
		}
		so.AddConfigEntry(kOrtSessionOptionsConfigIntraOpThreadAffinities, affinity.c_str());
		Log::Info("[{}] Intra-op pool pinned inside the slice (1-based ids: '{}')", m_tag, affinity);
	}

	// Spinning: on a DEDICATED slice, busy-waiting burns only cores this session
	// owns and removes wake-up latency. Without a partition, or when slices
	// overlap, spinning fights the other sessions.
	const bool dedicatedSlice = !cpuPartition.logicalProcessors.empty() && !cpuPartition.shared;
	so.AddConfigEntry(kOrtSessionOptionsConfigAllowIntraOpSpinning, dedicatedSlice ? "1" : "0");

	// Pool threads created at TIME_CRITICAL priority, see the factory above
	so.SetCustomCreateThreadFn(OrtRtCreateThread);
	so.SetCustomJoinThreadFn(OrtRtJoinThread);
	Log::Info("[{}] Intra-op pool: {} TIME_CRITICAL threads, spinning {}.",
		m_tag, sliceThreads, dedicatedSlice ? "ON" : "OFF");
}

std::string ProviderSelector::ResolveFromExeDir(const std::string& path, const std::string& fallback) const
{
	const std::filesystem::path p(path.empty() ? std::string(fallback) : path);
	if (p.is_absolute())
		return p.string();
	return (m_runtime.ExecutableDirectory() / p).string();
}
#if defined(ORT_EP_GPU)
ExecutionProvider ProviderSelector::AppendTensorRT(Ort::SessionOptions& so)
{
	const IniConfiguration& ini = IniConfiguration::Instance();
	const OrtApi& api = Ort::GetApi();
	OrtTensorRTProviderOptionsV2* trt = nullptr;
	bool tensorRT = false;

	try {
		Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&trt));

		const std::string deviceId = std::to_string(ResolveCudaDeviceIndex(ini, m_tag));
		const std::string fp16 = ini.GetTensorRTFP16() ? "1" : "0";																												// FP16 kernels: big speedup on modern GPUs
		const std::string cacheOn = ini.GetTensorRTCacheEnable() ? "1" : "0";																							// reuse engine + kernel timings across runs
		const std::string optLevel = std::to_string(ini.GetTensorRTOptimizationLevel());																	// 5 = most aggressive builder search
		const std::string engineCache = ResolveFromExeDir(WideToUtf8(ini.GetTensorRTEngineCache()), "trt_engine_cache");
		const std::string timingCache = ResolveFromExeDir(WideToUtf8(ini.GetTensorRTTimingCache()), "trt_timing_cache");
		const std::string maxWorkspace = std::to_string(
			static_cast<unsigned long long>(ini.GetTensorRTMaxWorkspaceSizeMB()) * 1024ULL * 1024ULL);											// MB -> byte
		const std::string maxPartIter = std::to_string(ini.GetTensorRTMaxPartitionIterations());
		const std::string minSubgraph = std::to_string(ini.GetTensorRTMinSubgraphSize());
		const std::string int8On = ini.GetTensorRTInt8Enable() ? "1" : "0";
		const std::string int8Native = ini.GetTensorRTInt8UseNativeCalibrationTable() ? "1" : "0";
		const std::string dumpSub = ini.GetTensorRTDumpSubgraphs() ? "1" : "0";

		// Same order in the two arrays
		const char* keys[] = {
			"device_id",
			"trt_max_workspace_size",
			"trt_max_partition_iterations",
			"trt_min_subgraph_size",
			"trt_fp16_enable",
			"trt_int8_enable",
			"trt_int8_use_native_calibration_table",
			"trt_dump_subgraphs",
			"trt_engine_cache_enable",
			"trt_engine_cache_path",
			"trt_timing_cache_enable",
			"trt_timing_cache_path",
			"trt_builder_optimization_level"
		};
		const char* values[] = {
			deviceId.c_str(),
			maxWorkspace.c_str(),
			maxPartIter.c_str(),
			minSubgraph.c_str(),
			fp16.c_str(),
			int8On.c_str(),
			int8Native.c_str(),
			dumpSub.c_str(),
			cacheOn.c_str(),
			engineCache.c_str(),
			cacheOn.c_str(),
			timingCache.c_str(),
			optLevel.c_str()
		};
		const size_t count = sizeof(keys) / sizeof(keys[0]);

		Ort::ThrowOnError(api.UpdateTensorRTProviderOptions(trt, keys, values, count));
		so.AppendExecutionProvider_TensorRT_V2(*trt);
		tensorRT = true;
		Log::Info("[{}] TensorRT EP appended (FP16 + engine/timing cache).", m_tag);
	}
	catch (const std::exception& e) {
		Log::Warning("[{}] TensorRT unavailable, trying CUDA: {}", m_tag, e.what());
	}
	catch (...) {
		Log::Warning("[{}] TensorRT unavailable, trying CUDA: unknown error", m_tag);
	}

	if (trt != nullptr)
		api.ReleaseTensorRTProviderOptions(trt);

	// CUDA also covers the subgraphs TensorRT cannot handle
	try {
		AppendCudaClassic(so);
		if (tensorRT) {
			Log::Info("[{}] CUDA EP appended (fallback for subgraphs TensorRT cannot handle).", m_tag);
			return ExecutionProvider::TensorRT;
		}
		Log::Info("[{}] CUDA EP appended.", m_tag);
		return ExecutionProvider::CUDA;
	}
	catch (const std::exception& e) {
		Log::Error("[{}] GPU providers unavailable, falling back to native ONNX Runtime CPU: {}", m_tag, e.what());
		return ExecutionProvider::CPU;
	}
}
ExecutionProvider ProviderSelector::AppendCuda(Ort::SessionOptions& so)
{
	ExecutionProvider expected = ExecutionProvider::CPU;
	if (ApplySelectionPolicy(so, expected))
	{
		return expected;
	}

	try
	{
		AppendCudaPlugin(so);
		return ExecutionProvider::CUDA;
	}
	catch (const std::exception& e)
	{
		Log::Warning("[{}] CUDA plugin-EP selection failed ({}), trying the classic provider options.", m_tag, e.what());
	}

	try
	{
		AppendCudaClassic(so);
		Log::Info("[{}] CUDA EP appended.", m_tag);
		return ExecutionProvider::CUDA;
	}
	catch (const std::exception& e) 
	{
		Log::Error("[{}] CUDA unavailable, falling back to native ONNX Runtime CPU: {}", m_tag, e.what());
		return ExecutionProvider::CPU;
	}
}

void ProviderSelector::AppendCudaClassic(Ort::SessionOptions& so)
{
	const IniConfiguration& ini = IniConfiguration::Instance();
	const OrtApi& api = Ort::GetApi();

	OrtCUDAProviderOptionsV2* cuda = nullptr;
	Ort::ThrowOnError(api.CreateCUDAProviderOptions(&cuda));

	std::unordered_map<std::string, std::string> opts = CudaProviderOptions(ini);
	std::vector<const char*> keys;
	std::vector<const char*> values;

	try
	{
		opts["device_id"] = std::to_string(ResolveCudaDeviceIndex(ini, m_tag));
		for (std::unordered_map<std::string, std::string>::const_iterator it = opts.begin(); it != opts.end(); ++it)
		{
			keys.push_back(it->first.c_str());
			values.push_back(it->second.c_str());
		}
		Ort::ThrowOnError(api.UpdateCUDAProviderOptions(cuda, keys.data(), values.data(), keys.size()));
		so.AppendExecutionProvider_CUDA_V2(*cuda);
	}
	catch (...)
	{
		api.ReleaseCUDAProviderOptions(cuda);
		throw;
	}
	api.ReleaseCUDAProviderOptions(cuda);
}

void ProviderSelector::AppendCudaPlugin(Ort::SessionOptions& so)
{
	const IniConfiguration& ini = IniConfiguration::Instance();
	const std::vector<Ort::ConstEpDevice> gpus = m_runtime.CudaLibrary().Devices(OrtHardwareDeviceType_GPU);
	if (gpus.empty())
		throw std::runtime_error("ONNX Runtime reports no GPU device for the CUDA EP");

	for (size_t i = 0; i < gpus.size();++i)
	{
		const Ort::ConstHardwareDevice hw = gpus[i].Device();
		const char* luid = hw.Metadata().GetValue("LUID");
		Log::Info("[{}] EP device {}: ep='{}' hwVendor='{}' vendorId=0x{:x} deviceId=0x{:x} LUID={}", m_tag, i, gpus[i].EpName(), hw.Vendor(), hw.VendorId(), hw.DeviceId(), luid ? luid : "<n/a>");
	}

	size_t index = ini.GetGeneralConfigurationGpuDevice();
	if (index >= gpus.size())
	{
		Log::Warning("[{}] ini device={} is not valid ({} CUDA EP device(s) found): using the first one.", m_tag, index, gpus.size());
		index = 0;
	}

	std::vector<Ort::ConstEpDevice> selected;
	selected.push_back(gpus[index]);
	so.AppendExecutionProvider_V2(m_runtime.Env(), selected, CudaProviderOptions(ini));
	m_usedEpDevice = true;
	Log::Info("[{}] CUDA EP appended on device {} of {} (plugin EP selection).", m_tag, index, gpus.size());
}

bool ProviderSelector::ApplySelectionPolicy(Ort::SessionOptions& so, ExecutionProvider& expected)
{
	const std::string name = WideToUtf8(IniConfiguration::Instance().GetGeneralConfigurationEpSelectionPolicy());
	if (name.empty())
		return false;

	OrtExecutionProviderDevicePolicy policy = OrtExecutionProviderDevicePolicy_DEFAULT;
	if (!ParseEpSelectionPolicy(name, policy))
	{
		Log::Warning("[{}] Unknown epSelectionPolicy '{}': using the explicit device selection.", m_tag, name);
		return false;
	}

	try
	{
		const bool gpuListed = !m_runtime.CudaLibrary().Devices(OrtHardwareDeviceType_GPU).empty();
		so.SetEpSelectionPolicy(policy);   // no EP appended: ORT chooses when the session is created

		// Only an expectation: OrtEngine confirms it after the session exists
		if (gpuListed && policy != OrtExecutionProviderDevicePolicy_PREFER_CPU)
			expected = ExecutionProvider::CUDA;
		else
			expected = ExecutionProvider::CPU;

		m_usedEpDevice = true;
		Log::Info("[{}] EP selection policy '{}' set: provider chosen by ONNX Runtime.", m_tag, name);
		return true;
	}
	catch (const std::exception& e) {
		Log::Warning("[{}] EP selection policy failed ({}), using the explicit device selection.", m_tag, e.what());
		return false;
	}
}
#elif defined(ORT_EP_OPENVINO)
ExecutionProvider ProviderSelector::AppendOpenVINO(Ort::SessionOptions& so, ExecutionProvider requested)
{
	if (requested != ExecutionProvider::OpenVINO_CPU && requested != ExecutionProvider::OpenVINO_GPU)
		return ExecutionProvider::CPU;

	// Precision, streams, priority and cache are defined in openvino_config.json,
	// next to the executable. load_config expects the JSON CONTENT, not the file path.
	const std::filesystem::path configFile = m_runtime.ExecutableDirectory() / "openvino_config.json";
	const bool gpu = (requested == ExecutionProvider::OpenVINO_GPU);
	const char* device = gpu ? "GPU" : "CPU";

	try {
		const std::string loadConfig = ReadTextFile(configFile);
		if (loadConfig.empty())
			throw std::runtime_error("OpenVINO config file missing or empty: " + configFile.string());

		std::unordered_map<std::string, std::string> options;
		options["device_type"] = device;
		options["load_config"] = loadConfig;
		options["disable_dynamic_shapes"] = gpu ? "True" : "False";
		so.AppendExecutionProvider_OpenVINO_V2(options);

		Log::Info("[{}] OpenVINO EP appended. Device: {}, config: {}.", m_tag, device, configFile.string());
		return requested;
	}
	catch (const std::exception& e) {
		Log::Error("[{}] OpenVINO unavailable. Falling back to native ONNX Runtime CPU: {}", m_tag, e.what());
	}
	catch (...) {
		Log::Error("[{}] OpenVINO unavailable. Falling back to native ONNX Runtime CPU: unknown error", m_tag);
	}
	return ExecutionProvider::CPU;
}
#endif


