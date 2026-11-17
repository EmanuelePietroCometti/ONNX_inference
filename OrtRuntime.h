#pragma once
#include "PluginEPLibrary.h"
#include <onnxruntime_cxx_api.h>
#include <filesystem>

class OrtRuntime
{
public:
	OrtRuntime();
	~OrtRuntime();
	OrtRuntime(const OrtRuntime&) = delete;
	OrtRuntime& operator=(const OrtRuntime&) = delete;

	Ort::Env& Env() { return m_env; }
	const std::filesystem::path& ExecutableDirectory() const { return m_executableDirectory; }

	CudaEpLibrary& CudaLibrary();

private:
	Ort::Env m_env;
	std::filesystem::path m_executableDirectory;
	std::unique_ptr<CudaEpLibrary> m_cuda;
};