#pragma once
#include <Windows.h>

// Utilizzo del pattern Singleton per la creazione di una classe
// di configurazione che viene instanziata una sola volta a inzio del programma
// e che non viene ricreata. Chiunque voglia avere accesso a questa classe
// basta che includa Configuration.h e che invochi il metodo Instance().
class IniConfiguration {
public:
	static IniConfiguration& Instance()
	{
		static IniConfiguration instance;
		return instance;
	}

	IniConfiguration(const IniConfiguration&) = delete;
	IniConfiguration& operator=(const IniConfiguration&) = delete;
	IniConfiguration(IniConfiguration&&) = delete;
	IniConfiguration& operator=(IniConfiguration&&) = delete;
	void Load(const WCHAR* path);

	WORD GetLoggerVerbosity() const;
	bool GetLoggerANSIColorSupported() const;
	WORD GetLoggerMaxMessagesChars() const;
	WORD GetLoggerFlushIntervalMs() const;
	WORD GetLoggerNotifyThreshold() const;
	WORD GetLoggerQueueCapacity() const;

	WORD GetONNXWarmUps() const;

	WORD GetTensorRTFP16() const;
	WORD GetTensorRTCacheEnable() const;
	const WCHAR* GetTensorRTEngineCache() const;
	const WCHAR* GetTensorRTTimingCache() const;
	WORD GetTensorRTOptimizationLevel() const;
	WORD GetTensorRTMaxWorkspaceSizeMB() const;
	WORD GetTensorRTMaxPartitionIterations() const;
	WORD GetTensorRTMinSubgraphSize() const;
	WORD GetTensorRTInt8Enable() const;
	WORD GetTensorRTInt8UseNativeCalibrationTable() const;
	WORD GetTensorRTDumpSubgraphs() const;

	WORD GetGeneralConfigurationMaxControlPoints() const;
	WORD GetGeneralConfigurationMaxText() const;
	WORD GetGeneralConfigurationGpuDevice() const;
	const WCHAR* GetGeneralConfigurationEpSelectionPolicy() const;
	WORD GetCudaConvUseMaxWorkspace() const;
	WORD GetCudaGpuMemLimitMB() const;
	const WCHAR* GetCudaArenaExtendStrategy() const;
	const WCHAR* GetCudaCudnnConvAlgoSearch() const;
	WORD GetCudaDoCopyInDefaultStream() const;
	WORD GetCudaCudnnConv1dPadToNc1d() const;

private:
	IniConfiguration() = default;
	~IniConfiguration();

	WORD logger_verbosity = 0;
	bool logger_ansiColorSupported = false;
	WORD logger_maxMessagesChars = 256;
	WORD logger_flushIntervalMs = 1000;
	WORD logger_notifyThreshold = 0;
	WORD logger_queueCapacity = 1024;

	WORD onnx_warmUps = 0;

	WORD tensort_fp16 = 0;
	WORD tensort_cacheEnable = 0;
	WCHAR tensort_engineCache[512] = { 0 };
	WCHAR tensort_timingCache[512] = { 0 };
	WORD tensort_optimizationLevel = 3;
	WORD tensort_maxWorkspaceSizeMB = 0;
	WORD tensort_maxPartitionIterations = 1000;
	WORD tensort_minSubgraphSize = 1;
	WORD tensort_int8Enable = 0;
	WORD tensort_int8UseNativeCalibrationTable = 0;
	WORD tensort_dumpSubgraphs = 0;

	WORD cuda_cudnnConvUseMaxWorkspace = 1;
	WORD cuda_gpuMemLimitMB = 2048;
	WCHAR cuda_arenaExtendStrategy[64] = L"kSameAsRequested";
	WCHAR cuda_cudnnConvAlgoSearch[64] = L"DEFAULT";
	WORD cuda_doCopyInDefaultStream = 1;
	WORD cuda_cudnnConv1dPadToNc1d = 1;

	WORD generalConfiguration_maxControlPoints = 0;
	WORD generalConfiguration_maxText = 0;
	WORD generalConfiguration_gpuDevice = 0;
	WCHAR generalConfiguration_epSelectionPolicy[64] = L"";
};