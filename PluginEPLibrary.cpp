#include "PluginEPLibrary.h"
#include "AsyncLogger.h"
#include <filesystem>


PluginEPLibrary::PluginEPLibrary(Ort::Env& env, const std::string& epName, const std::string& registrationName, const std::filesystem::path& libraryPath) :
	m_env(env), 
	m_epName(epName), 
	m_registered(false)
{
	const std::vector<Ort::ConstEpDevice> devices = m_env.GetEpDevices();
	for (size_t i = 0; i < devices.size(); ++i) {
		if (m_epName == devices[i].EpName())
			return;

		if(!std::filesystem::is_regular_file(libraryPath))
			throw std::runtime_error("Plugin EP library not found: " + libraryPath.string());	

		m_env.RegisterExecutionProviderLibrary(m_registrationName.c_str(), libraryPath.native());
		m_registered = true;
	}
}

PluginEPLibrary::~PluginEPLibrary()
{
	if (!m_registered)
		return;

	try
	{
		m_env.UnregisterExecutionProviderLibrary(m_registrationName.c_str());
	}
	catch (std::exception& e)
	{
		Log::Warning("EP library '{0}' unregistration failed: {1}", m_epName, e.what());
	}
}

std::vector<Ort::ConstEpDevice> PluginEPLibrary::Devices(OrtHardwareDeviceType type) const
{
	std::vector<Ort::ConstEpDevice> result;
	const std::vector<Ort::ConstEpDevice> devices = m_env.GetEpDevices();
	for (size_t i = 0; i < devices.size(); ++i) {
		if (m_epName == devices[i].EpName() && type == devices[i].Device().Type())
			result.push_back(devices[i]);
	}
	return result;
}

CudaEpLibrary::CudaEpLibrary(Ort::Env& env, const std::filesystem::path& directory) :
	PluginEPLibrary(env, "cuda", "CUDAExecutionProvider", directory / "onnxruntime_providers_cuda.dll")
{
}
