#pragma once
#include <onnxruntime_cxx_api.h>
#include <filesystem>
#include <string>
#include <vector>
#include <filesystem>


class PluginEPLibrary
{
public:
	virtual ~PluginEPLibrary();
	PluginEPLibrary(const PluginEPLibrary&) = delete;
	PluginEPLibrary& operator=(const PluginEPLibrary&) = delete;

	std::vector<Ort::ConstEpDevice> Devices(OrtHardwareDeviceType type) const;
	const std::string& EpName() const { return m_epName; }

protected:
	PluginEPLibrary(Ort::Env& env, const std::string& epName, const std::string& registrationName, const std::filesystem::path& libraryPath);

private:
	Ort::Env& m_env;
	std::string m_epName;
	std::string m_registrationName;
	bool m_registered;
};


class CudaEpLibrary : public PluginEPLibrary {
public:
	CudaEpLibrary(Ort::Env& env, const std::filesystem::path& directory);
};