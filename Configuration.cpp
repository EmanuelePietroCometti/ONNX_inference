#include "Configuration.h"

IniConfiguration::~IniConfiguration() = default;

void IniConfiguration::Load(const WCHAR* path_iniConfigurationFile)
{
	if (path_iniConfigurationFile == nullptr)
	{
		return;
	}

	logger_verbosity = static_cast<WORD>(GetPrivateProfileIntW(L"Logger", L"verbosity", 0, path_iniConfigurationFile));
	logger_ansiColorSupported = GetPrivateProfileIntW(L"Logger", L"ansiColorSupported", 0, path_iniConfigurationFile) != 0;
	logger_maxMessagesChars = static_cast<WORD>(GetPrivateProfileIntW(L"Logger", L"maxMessagesChars", 256, path_iniConfigurationFile));
	logger_flushIntervalMs = static_cast<WORD>(GetPrivateProfileIntW(L"Logger", L"flushIntervalMs", 1000, path_iniConfigurationFile));
	logger_notifyThreshold = static_cast<WORD>(GetPrivateProfileIntW(L"Logger", L"notifyThreshold", 0, path_iniConfigurationFile));
	logger_queueCapacity = static_cast<WORD>(GetPrivateProfileIntW(L"Logger", L"queueCapacity", 1024, path_iniConfigurationFile));

	onnx_warmUps = static_cast<WORD>(GetPrivateProfileIntW(L"ONNX", L"warmUps", 200, path_iniConfigurationFile));

	tensort_fp16 = static_cast<WORD>(GetPrivateProfileIntW(L"TensorRT", L"fp16", 0, path_iniConfigurationFile));
	tensort_cacheEnable = static_cast<WORD>(GetPrivateProfileIntW(L"TensorRT", L"cacheEnable", 0, path_iniConfigurationFile));
	tensort_optimizationLevel = static_cast<WORD>(GetPrivateProfileIntW(L"TensorRT", L"optimizationLevel", 3, path_iniConfigurationFile));

	GetPrivateProfileStringW(L"TensorRT", L"engineCache", L"", tensort_engineCache, 512, path_iniConfigurationFile);
	GetPrivateProfileStringW(L"TensorRT", L"timingCache", L"", tensort_timingCache, 512, path_iniConfigurationFile);
	tensort_maxWorkspaceSizeMB = static_cast<WORD>(GetPrivateProfileIntW(L"TensorRT", L"maxWorkspaceSizeMB", 0, path_iniConfigurationFile));
	tensort_maxPartitionIterations = static_cast<WORD>(GetPrivateProfileIntW(L"TensorRT", L"maxPartitionIterations", 1000, path_iniConfigurationFile));
	tensort_minSubgraphSize = static_cast<WORD>(GetPrivateProfileIntW(L"TensorRT", L"minSubgraphSize", 1, path_iniConfigurationFile));
	tensort_int8Enable = static_cast<WORD>(GetPrivateProfileIntW(L"TensorRT", L"int8Enable", 0, path_iniConfigurationFile));
	tensort_int8UseNativeCalibrationTable = static_cast<WORD>(GetPrivateProfileIntW(L"TensorRT", L"int8UseNativeCalibrationTable", 0, path_iniConfigurationFile));
	tensort_dumpSubgraphs = static_cast<WORD>(GetPrivateProfileIntW(L"TensorRT", L"dumpSubgraphs", 0, path_iniConfigurationFile));

	cuda_cudnnConvUseMaxWorkspace = static_cast<WORD>(GetPrivateProfileIntW(L"CUDA", L"cudaConvUseMaxWorkspace", 1, path_iniConfigurationFile));
	cuda_gpuMemLimitMB = static_cast<WORD>(GetPrivateProfileIntW(L"CUDA", L"gpuMemLimitMB", 2048, path_iniConfigurationFile));
	GetPrivateProfileStringW(L"CUDA", L"arenaExtendStrategy", L"kSameAsRequested", cuda_arenaExtendStrategy, 64, path_iniConfigurationFile);
	GetPrivateProfileStringW(L"CUDA", L"cudnnConvAlgoSearch", L"DEFAULT", cuda_cudnnConvAlgoSearch, 64, path_iniConfigurationFile);
	cuda_doCopyInDefaultStream = static_cast<WORD>(GetPrivateProfileIntW(L"CUDA", L"doCopyInDefaultStream", 1, path_iniConfigurationFile));
	cuda_cudnnConv1dPadToNc1d = static_cast<WORD>(GetPrivateProfileIntW(L"CUDA", L"cudnnConv1dPadToNc1d", 1, path_iniConfigurationFile));

	generalConfiguration_maxControlPoints = static_cast<WORD>(GetPrivateProfileIntW(L"GeneralConfiguration", L"maxControlPoints", 0, path_iniConfigurationFile));
	generalConfiguration_maxText = static_cast<WORD>(GetPrivateProfileIntW(L"GeneralConfiguration", L"maxText", 0, path_iniConfigurationFile));
	generalConfiguration_gpuDevice = static_cast<WORD>(GetPrivateProfileIntW(L"GeneralConfiguration", L"device", 0, path_iniConfigurationFile));
	GetPrivateProfileStringW(L"GeneralConfiguration", L"epSelectionPolicy", L"", generalConfiguration_epSelectionPolicy, 64, path_iniConfigurationFile);
}

WORD IniConfiguration::GetLoggerVerbosity() const { return logger_verbosity; }
bool IniConfiguration::GetLoggerANSIColorSupported() const { return logger_ansiColorSupported; }
WORD IniConfiguration::GetLoggerMaxMessagesChars() const { return logger_maxMessagesChars; }
WORD IniConfiguration::GetLoggerFlushIntervalMs() const { return logger_flushIntervalMs; }
WORD IniConfiguration::GetLoggerNotifyThreshold() const { return logger_notifyThreshold; }
WORD IniConfiguration::GetLoggerQueueCapacity() const { return logger_queueCapacity; }

WORD IniConfiguration::GetONNXWarmUps() const { return onnx_warmUps; }

WORD IniConfiguration::GetTensorRTFP16() const { return tensort_fp16; }
WORD IniConfiguration::GetTensorRTCacheEnable() const { return tensort_cacheEnable; }

const WCHAR* IniConfiguration::GetTensorRTEngineCache() const { return tensort_engineCache; }
const WCHAR* IniConfiguration::GetTensorRTTimingCache() const { return tensort_timingCache; }

WORD IniConfiguration::GetTensorRTOptimizationLevel() const { return tensort_optimizationLevel; }

WORD IniConfiguration::GetTensorRTMaxWorkspaceSizeMB() const { return tensort_maxWorkspaceSizeMB; }
WORD IniConfiguration::GetTensorRTMaxPartitionIterations() const { return tensort_maxPartitionIterations; }
WORD IniConfiguration::GetTensorRTMinSubgraphSize() const { return tensort_minSubgraphSize; }
WORD IniConfiguration::GetTensorRTInt8Enable() const { return tensort_int8Enable; }
WORD IniConfiguration::GetTensorRTInt8UseNativeCalibrationTable() const { return tensort_int8UseNativeCalibrationTable; }
WORD IniConfiguration::GetTensorRTDumpSubgraphs() const { return tensort_dumpSubgraphs; }

WORD IniConfiguration::GetGeneralConfigurationMaxControlPoints() const { return generalConfiguration_maxControlPoints; }
WORD IniConfiguration::GetGeneralConfigurationMaxText() const { return generalConfiguration_maxText; }
WORD IniConfiguration::GetGeneralConfigurationGpuDevice() const { return generalConfiguration_gpuDevice; }
const WCHAR* IniConfiguration::GetGeneralConfigurationEpSelectionPolicy() const { return generalConfiguration_epSelectionPolicy; }

WORD IniConfiguration::GetCudaConvUseMaxWorkspace() const { return cuda_cudnnConvUseMaxWorkspace; }
WORD IniConfiguration::GetCudaGpuMemLimitMB() const { return cuda_gpuMemLimitMB; }
const WCHAR* IniConfiguration::GetCudaArenaExtendStrategy() const { return cuda_arenaExtendStrategy; }
const WCHAR* IniConfiguration::GetCudaCudnnConvAlgoSearch() const { return cuda_cudnnConvAlgoSearch; }
WORD IniConfiguration::GetCudaDoCopyInDefaultStream() const { return cuda_doCopyInDefaultStream; }
WORD IniConfiguration::GetCudaCudnnConv1dPadToNc1d() const { return cuda_cudnnConv1dPadToNc1d; }
