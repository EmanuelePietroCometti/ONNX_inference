#include "OrtEngine.h"
#include "OrtRuntime.h"
#include "ProviderSelector.h"
#include "IoBuffers.h"
#include "AsyncLogger.h"
#include <stdexcept>
#include <vector>

OrtEngine::OrtEngine(OrtRuntime& runtime, const std::string& tag)
	: m_runtime(runtime), m_tag(tag), m_ep(ExecutionProvider::CPU), m_usesEpDevice(false)
{
}

OrtEngine::~OrtEngine()
{
}

// VRAM buffers only for the CUDA-based providers: OpenVINO and CPU work on host memory
bool OrtEngine::UsesCudaMemory() const
{
	return m_ep == ExecutionProvider::TensorRT || m_ep == ExecutionProvider::CUDA;
}

void OrtEngine::CreateSession(const std::filesystem::path& modelPath, const RT::CpuPartition& cpuPartition,
	ExecutionProvider requested)
{
	if (modelPath.empty())
		throw std::runtime_error(m_tag + " model path is empty.");

	Ort::SessionOptions options;
	ProviderSelector selector(m_runtime, m_tag);
	m_ep = selector.Configure(options, cpuPartition, requested);
	m_usesEpDevice = selector.UsedEpDevice();

	m_io.reset();   // buffers of a previous session go before it
	session.reset(new Ort::Session(m_runtime.Env(), modelPath.c_str(), options));

#if defined(ORT_EP_GPU)
	// With the plugin path or a policy, ask the session which EP it really gave to the input
	if (m_usesEpDevice) {
		const std::vector<Ort::ConstEpDevice> devices = session->GetEpDeviceForInputs();
		if (!devices.empty() && static_cast<const OrtEpDevice*>(devices[0]) != nullptr) {
			ExecutionProvider actual = ExecutionProvider::CPU;
			if (m_runtime.CudaLibrary().EpName() == devices[0].EpName())
				actual = ExecutionProvider::CUDA;

			if (actual != m_ep)
				Log::Warning("[{}] ONNX Runtime selected '{}', not the expected provider.", m_tag, devices[0].EpName());
			m_ep = actual;
		}
	}
#endif
}

void OrtEngine::CreateIoBuffers()
{
	m_io.reset();

	if (!UsesCudaMemory()) {
		m_io.reset(new HostIoBuffers(*session));
		DescribeIo(*m_io);
	}
#if defined(ORT_EP_GPU)
	else {
		if (m_usesEpDevice) {
			try {
				m_io.reset(new OrtDeviceIoBuffers(*session, m_runtime.Env()));
				DescribeIo(*m_io);

				// One full round trip, so a problem shows up here and not on the first frame
				m_io->Upload();
				m_io->Run();
				m_io->Download();
			}
			catch (const std::exception& e) {
				Log::Warning("[{}] ORT device tensors not usable ({}), using CUDA buffers.", m_tag, e.what());
				m_io.reset();
			}
		}

		if (m_io == nullptr) {
			m_io.reset(new CudaIoBuffers(*session));
			DescribeIo(*m_io);
		}
	}
#endif

	Log::Info("[{}] I/O binding: {}.", m_tag, m_io->Description());
}