#include "IoBuffers.h"
#include <stdexcept>
#include <utility>

#if defined(ORT_EP_GPU)
#include <cuda_runtime.h>
#endif

// ---------------- IoBuffers ----------------

IoBuffers::IoBuffers(Ort::Session& session) : m_session(session), m_binding(session)
{
}

IoBuffers::~IoBuffers()
{
}

void IoBuffers::AddInput(const std::string& name, void* hostData, size_t byteCount, const std::vector<int64_t>& shape, ONNXTensorElementDataType type)
{
	Ort::Value& tensor = CreateTensor(true, hostData, byteCount, shape, type);
	m_binding.BindInput(name.c_str(), tensor);
}

void IoBuffers::AddOutput(const std::string& name, void* hostData, size_t byteCount, const std::vector<int64_t>& shape, ONNXTensorElementDataType type)
{
	Ort::Value& tensor = CreateTensor(false, hostData, byteCount, shape, type);
	m_binding.BindOutput(name.c_str(), tensor);
}

void IoBuffers::Run()
{
	m_session.Run(Ort::RunOptions{ nullptr }, m_binding);
}

// ---------------- HostIoBuffers ----------------

HostIoBuffers::HostIoBuffers(Ort::Session& session)
	: IoBuffers(session), m_cpuMemory(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
}

Ort::Value& HostIoBuffers::CreateTensor(bool /*isInput*/, void* hostData, size_t byteCount, const std::vector<int64_t>& shape, ONNXTensorElementDataType type)
{
	m_tensors.push_back(Ort::Value::CreateTensor(m_cpuMemory, hostData, byteCount, shape.data(), shape.size(), type));
	return m_tensors.back();
}

#if defined(ORT_EP_GPU)

// ---------------- CudaIoBuffers ----------------

CudaIoBuffers::CudaIoBuffers(Ort::Session& session)
	: IoBuffers(session), m_cudaMemory("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault)
{
}

CudaIoBuffers::~CudaIoBuffers()
{
	for (size_t i = 0; i < m_inputs.size(); ++i)
		cudaFree(m_inputs[i].device);
	for (size_t i = 0; i < m_outputs.size(); ++i)
		cudaFree(m_outputs[i].device);
}

Ort::Value& CudaIoBuffers::CreateTensor(bool isInput, void* hostData, size_t byteCount,
	const std::vector<int64_t>& shape, ONNXTensorElementDataType type)
{
	Block block;
	block.host = hostData;
	block.device = nullptr;
	block.byteCount = byteCount;

	if (cudaMalloc(&block.device, byteCount) != cudaSuccess)
		throw std::runtime_error("cudaMalloc failed for an I/O tensor");

	if (isInput)
		m_inputs.push_back(block);
	else
		m_outputs.push_back(block);

	m_tensors.push_back(Ort::Value::CreateTensor(m_cudaMemory, block.device, byteCount, shape.data(), shape.size(), type));
	return m_tensors.back();
}

void CudaIoBuffers::Upload()
{
	for (size_t i = 0; i < m_inputs.size(); ++i)
		cudaMemcpy(m_inputs[i].device, m_inputs[i].host, m_inputs[i].byteCount, cudaMemcpyHostToDevice);
}

void CudaIoBuffers::Download()
{
	for (size_t i = 0; i < m_outputs.size(); ++i)
		cudaMemcpy(m_outputs[i].host, m_outputs[i].device, m_outputs[i].byteCount, cudaMemcpyDeviceToHost);
}

// ---------------- OrtDeviceIoBuffers ----------------

OrtDeviceIoBuffers::OrtDeviceIoBuffers(Ort::Session& session, Ort::Env& env) : 
	IoBuffers(session), 
	m_env(env),
	m_cpuMemory(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
	m_deviceAllocator(nullptr)
{
	const std::vector<Ort::ConstEpDevice> devices = session.GetEpDeviceForInputs();
	if (devices.empty() || static_cast<const OrtEpDevice*>(devices[0]) == nullptr)
		throw std::runtime_error("the session reports no EP device for its input");

	// Memory of the device the session really runs on: no hard-coded GPU index
	const Ort::ConstMemoryInfo deviceMemory = devices[0].GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);
	Ort::UnownedAllocator allocator = m_env.GetSharedAllocator(deviceMemory);
	m_deviceAllocator = allocator;
	if (m_deviceAllocator == nullptr)
		throw std::runtime_error("ONNX Runtime has no shared allocator for the session's device");
}

Ort::Value& OrtDeviceIoBuffers::CreateTensor(bool isInput, void* hostData, size_t byteCount,
	const std::vector<int64_t>& shape, ONNXTensorElementDataType type)
{
	Ort::Value host = Ort::Value::CreateTensor(m_cpuMemory, hostData, byteCount, shape.data(), shape.size(), type);
	Ort::Value device = Ort::Value::CreateTensor(m_deviceAllocator, shape.data(), shape.size(), type);

	if (isInput) {
		m_hostInputs.push_back(std::move(host));
		m_deviceInputs.push_back(std::move(device));
		return m_deviceInputs.back();
	}
	m_hostOutputs.push_back(std::move(host));
	m_deviceOutputs.push_back(std::move(device));
	return m_deviceOutputs.back();
}

void OrtDeviceIoBuffers::Upload()
{
	// No stream: the copy is finished when the call returns
	Ort::Status status = m_env.CopyTensors(m_hostInputs, m_deviceInputs, nullptr);
	if (!status.IsOK())
		throw std::runtime_error("host -> device copy failed: " + status.GetErrorMessage());
}

void OrtDeviceIoBuffers::Download()
{
	Ort::Status status = m_env.CopyTensors(m_deviceOutputs, m_hostOutputs, nullptr);
	if (!status.IsOK())
		throw std::runtime_error("device -> host copy failed: " + status.GetErrorMessage());
}

#endif