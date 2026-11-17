#pragma once
#include "IPC_Shared.h"
#include "RealTimeConfig.h"
#include <onnxruntime_cxx_api.h>
#include <string>

class OrtRuntime;

class ProviderSelector
{
public:
	ProviderSelector(OrtRuntime& runtime, const std::string& tag);

	ExecutionProvider Configure(Ort::SessionOptions& so, const RT::CpuPartition& cpuPartition, ExecutionProvider requested);

	bool UsedEpDevice() const { return m_usedEpDevice; }

private:
	void ConfigureThreading(Ort::SessionOptions& so, const RT::CpuPartition& cpuPartition, ExecutionProvider active);
	std::string ResolveFromExeDir(const std::string& path, const std::string& fallback) const;

#if defined(ORT_EP_GPU)
	ExecutionProvider AppendTensorRT(Ort::SessionOptions& so);	// TensorRT -> CUDA -> CPU
	ExecutionProvider AppendCuda(Ort::SessionOptions& so);			// policy -> plugin -> classic -> CPU

	void AppendCudaClassic(Ort::SessionOptions& so);
	void AppendCudaPlugin(Ort::SessionOptions& so);
	bool ApplySelectionPolicy(Ort::SessionOptions& so, ExecutionProvider& expected);
#elif defined(ORT_EP_OPENVINO)
	ExecutionProvider AppendOpenVINO(Ort::SessionOptions& so, ExecutionProvider requested);
#endif

	OrtRuntime& m_runtime;
	std::string m_tag;
	bool m_usedEpDevice;
};