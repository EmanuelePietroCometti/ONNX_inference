#pragma once
#include <onnxruntime_cxx_api.h>
#include <windows.h>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include "IPC_Shared.h"
#include "RealTimeConfig.h"

class OrtRuntime;
class IoBuffers;

// Base of every engine: owns the session and its I/O buffers.
// Derived classes do pre-processing, post-processing and say which tensors they use.
class OrtEngine {
public:
	virtual ~OrtEngine();

	// cpuPartition is the slice of physical cores owned by this session.
	virtual void Initialize(const std::filesystem::path& modelPath, const RT::CpuPartition& cpuPartition,
		ExecutionProvider executionProvider) = 0;

	virtual void Infer(const void* pInputImage, uint32_t width, uint32_t height, uint32_t channels,
		void* pOutputHeatmap, float& outAnomalyScore, std::string& outStatus, DWORD idPunto) = 0;

protected:
	OrtEngine(OrtRuntime& runtime, const std::string& tag);

	// Step 1: provider selection + session.
	void CreateSession(const std::filesystem::path& modelPath, const RT::CpuPartition& cpuPartition,
		ExecutionProvider requested);

	// Step 2, once shapes and host vectors are ready: binds the tensors named by DescribeIo.
	void CreateIoBuffers();
	virtual void DescribeIo(IoBuffers& io) = 0;

	IoBuffers& Io() { return *m_io; }
	ExecutionProvider ActiveProvider() const { return m_ep; }

	// Members are destroyed bottom-up: I/O buffers first, then the session.
	// The runtime belongs to the manager and outlives both.
	OrtRuntime& m_runtime;
	std::unique_ptr<Ort::Session> session;

private:
	bool UsesCudaMemory() const;

	std::string m_tag;
	ExecutionProvider m_ep;
	bool m_usesEpDevice;
	std::unique_ptr<IoBuffers> m_io;
};