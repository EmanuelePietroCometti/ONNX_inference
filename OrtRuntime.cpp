#include "OrtRuntime.h"
#include <Windows.h>

OrtRuntime::OrtRuntime() :
	m_env(ORT_LOGGING_LEVEL_WARNING, "OrtRuntimeEnv")
{
	wchar_t buffer[MAX_PATH];
	const DWORD length = GetModuleFileNameW(nullptr, buffer, MAX_PATH);
	if (length > 0 && length < MAX_PATH)
		m_executableDirectory = std::filesystem::path(buffer).parent_path();
	else
		m_executableDirectory = std::filesystem::current_path();
}

OrtRuntime::~OrtRuntime()
{
}

CudaEpLibrary& OrtRuntime::CudaLibrary()
{
	if (m_cuda == nullptr)
		m_cuda = std::make_unique<CudaEpLibrary>(m_env, m_executableDirectory);
	return *m_cuda;
}
