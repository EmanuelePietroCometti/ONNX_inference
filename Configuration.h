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

	const WCHAR* GetOpenVINOTargetDevice() const;
	const WCHAR* GetOpenVINOTargetPrecision() const;
	WORD GetOpenVINONumStreams() const;

	WORD GetGeneralConfigurationMaxControlPoints() const;
	WORD GetGeneralConfigurationMaxText() const;
	WORD GetGeneralConfigurationGpuDevice() const;
	WORD GetCudaConvUseMaxWorkspace() const;

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

	WORD cuda_cudnnConvUseMaxWorkspace = 0;

	WCHAR openVINO_targetDevice[64] = { 0 };
	WCHAR openVINO_targetPrecision[64] = { 0 };
	WORD openVINO_numStreams = 1;

	WORD generalConfiguration_maxControlPoints = 0;
	WORD generalConfiguration_maxText = 0;
	WORD generalConfiguration_gpuDevice = 0;
};