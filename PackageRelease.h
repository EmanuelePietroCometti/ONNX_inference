#pragma once
#include <windows.h>

namespace PackageInfo
{
#if defined(ORT_EP_GPU)
	inline constexpr WORD kRelease = 1;   // NVIDIA (TensorRT/CUDA)
#elif defined(ORT_EP_OPENVINO)
	inline constexpr WORD kRelease = 2;   // OpenVINO
#elif defined(ORT_EP_CPU)
	inline constexpr WORD kRelease = 3;   // CPU-only
#else
#error "No execution provider selected: define one of ORT_EP_GPU, ORT_EP_OPENVINO or ORT_EP_CPU."
#endif
}