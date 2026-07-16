#pragma once

#include "RealTimeConfig.h"
#include <string>
#include <vector>
#include <cstdint>

class IEngine {
public:
	virtual ~IEngine() = default;

	// Initializes the ONNX session (execution providers, warmup). cpuPartition
	// is the slice of physical cores owned by this session: on the CPU path the
	// intra-op pool is sized on it and pinned inside it, so concurrent control
	// points never compete for the same cores.
	virtual void Initialize(const std::wstring& modelPath, const RT::CpuPartition& cpuPartition) = 0;

	// Method that perform inference
	virtual void Infer(const void* pInputImage,uint32_t width, uint32_t height, uint32_t channels, void* pOutputHeatmap, float& outAnomalyScore, std::string& outStatus) = 0;
};
