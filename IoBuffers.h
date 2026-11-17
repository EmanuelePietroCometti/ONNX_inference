#pragma once
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

// Input/output tensors of one session, bound once and reused for every frame.
// The engine owns the host memory (its vectors). This class decides where ORT
// reads and writes, and copies between the two places when they differ.
class IoBuffers {
public:
	explicit IoBuffers(Ort::Session& session);
	virtual ~IoBuffers();
	IoBuffers(const IoBuffers&) = delete;
	IoBuffers& operator=(const IoBuffers&) = delete;

	// hostData must stay valid, at the same address, as long as this object lives.
	void AddInput(const std::string& name, void* hostData, size_t byteCount,
		const std::vector<int64_t>& shape, ONNXTensorElementDataType type);
	void AddOutput(const std::string& name, void* hostData, size_t byteCount,
		const std::vector<int64_t>& shape, ONNXTensorElementDataType type);

	virtual void Upload() = 0;     // host -> device, before Run
	void Run();
	virtual void Download() = 0;   // device -> host, after Run

	virtual const char* Description() const = 0;

protected:
	// Creates and keeps the tensor ORT must use for this block of host memory.
	virtual Ort::Value& CreateTensor(bool isInput, void* hostData, size_t byteCount, const std::vector<int64_t>& shape, ONNXTensorElementDataType type) = 0;

private:
	Ort::Session& m_session;
	Ort::IoBinding m_binding;
};

// CPU and OpenVINO: ORT works directly on the engine's host memory. No copies.
class HostIoBuffers : public IoBuffers {
public:
	explicit HostIoBuffers(Ort::Session& session);
	void Upload() override {}
	void Download() override {}
	const char* Description() const override { return "host memory, no copies"; }

protected:
	Ort::Value& CreateTensor(bool isInput, void* hostData, size_t byteCount,
		const std::vector<int64_t>& shape, ONNXTensorElementDataType type) override;

private:
	Ort::MemoryInfo m_cpuMemory;
	std::vector<Ort::Value> m_tensors;
};

#if defined(ORT_EP_GPU)
class CudaIoBuffers : public IoBuffers {
public:
	explicit CudaIoBuffers(Ort::Session& session);
	~CudaIoBuffers() override;
	void Upload() override;
	void Download() override;
	const char* Description() const override { return "CUDA buffers (cudaMalloc + cudaMemcpy)"; }

protected:
	Ort::Value& CreateTensor(bool isInput, void* hostData, size_t byteCount,
		const std::vector<int64_t>& shape, ONNXTensorElementDataType type) override;

private:
	struct Block {
		void* host;
		void* device;
		size_t byteCount;
	};

	Ort::MemoryInfo m_cudaMemory;
	std::vector<Block> m_inputs;
	std::vector<Block> m_outputs;
	std::vector<Ort::Value> m_tensors;
};

// Sample 20: device tensors owned by ORT, copies with CopyTensors.
// Works only when the session knows its EP device (plugin path or policy).
class OrtDeviceIoBuffers : public IoBuffers {
public:
	OrtDeviceIoBuffers(Ort::Session& session, Ort::Env& env);
	void Upload() override;
	void Download() override;
	const char* Description() const override { return "ORT device tensors (CopyTensors)"; }

protected:
	Ort::Value& CreateTensor(bool isInput, void* hostData, size_t byteCount, const std::vector<int64_t>& shape, ONNXTensorElementDataType type) override;

private:
	Ort::Env& m_env;
	Ort::MemoryInfo m_cpuMemory;
	OrtAllocator* m_deviceAllocator;   // shared allocator of the session's device, owned by ORT
	std::vector<Ort::Value> m_hostInputs;
	std::vector<Ort::Value> m_deviceInputs;
	std::vector<Ort::Value> m_deviceOutputs;
	std::vector<Ort::Value> m_hostOutputs;
};

#endif