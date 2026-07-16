#include "OrtsessionConfig.h"
#include <onnxruntime_session_options_config_keys.h>
#include <fmt/core.h>
#include <algorithm>
#include <filesystem>
#include <memory>
#include <thread>
#include <unordered_map>

//
// OrtSessionConfig.h
// Single source of truth for ONNX Runtime execution-provider selection and
// session-option tuning. Every engine (Anomaly, Classification, ...) calls
// ConfigureOrtSessionOptions() so the configuration is guaranteed identical
// across all of them and maintained in one place.
//
// Build is selected at compile time via exactly one of:
//   ORT_EP_GPU       -> TensorRT (FP16 + engine/timing cache) -> CUDA -> CPU
//   ORT_EP_OPENVINO  -> OpenVINO (single stream) -> CPU
//   ORT_EP_CPU       -> built-in ORT CPU EP
//


// Configures execution providers + session options for one session.
// Returns true if a hardware-accelerated EP (TensorRT / CUDA / OpenVINO) was
// appended, false if the run will happen on the built-in CPU EP.
// 'tag' is only used to prefix log lines (e.g. "Anomaly", "Classification").
bool ConfigureOrtSessionOptions(Ort::SessionOptions& so, const std::string& tag)
{
    bool hardwareAccelerated = false;

#if defined(ORT_EP_GPU)
    // GPU build: TensorRT -> CUDA -> CPU
    // ORT graph fusions still run before the EP partitions the graph, so keep
    // them enabled; TensorRT/CUDA then take the fused subgraphs.
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    const OrtApi& api = Ort::GetApi();

    // Absolute cache paths. Relative paths ("./trt_cache") break the moment the
    // working directory differs from the exe directory (service, launcher, or a
    // caller that changed CWD), silently disabling the cache.
    static const std::string engineCache =
        std::filesystem::absolute("trt_engine_cache").string();
    static const std::string timingCache =
        std::filesystem::absolute("trt_timing_cache").string();

    // TensorRT
    try {
        OrtTensorRTProviderOptionsV2* trt = nullptr;
        Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&trt));
        std::unique_ptr<OrtTensorRTProviderOptionsV2, void (*)(OrtTensorRTProviderOptionsV2*)>
            trtGuard(trt, [](OrtTensorRTProviderOptionsV2* p) {
            Ort::GetApi().ReleaseTensorRTProviderOptions(p);
                });

        const char* keys[] = {
            "device_id",
            "trt_fp16_enable",                // FP16 kernels: big speedup on modern GPUs
            "trt_engine_cache_enable",        // reuse the serialized engine across process runs
            "trt_engine_cache_path",
            "trt_timing_cache_enable",        // reuse kernel-timing results: much faster (re)builds
            "trt_timing_cache_path",
            "trt_builder_optimization_level"  // 5 = most aggressive builder search (slower build, faster engine)
        };
        const char* values[] = {
            "0", "1", "1", engineCache.c_str(), "1", timingCache.c_str(), "5"
        };
        Ort::ThrowOnError(api.UpdateTensorRTProviderOptions(
            trt, keys, values, sizeof(keys) / sizeof(keys[0])));

        so.AppendExecutionProvider_TensorRT_V2(*trt);
        hardwareAccelerated = true;
        fmt::print("[{}] TensorRT EP appended (FP16 + engine/timing cache).\n", tag);
    }
    catch (const Ort::Exception& e) {
        fmt::print(stderr, "[{}] TensorRT unavailable, trying CUDA: {}\n", tag, e.what());
    }

    // CUDA fallback (also covers subgraphs TensorRT cannot handle)
    try {
        OrtCUDAProviderOptionsV2* cuda = nullptr;
        Ort::ThrowOnError(api.CreateCUDAProviderOptions(&cuda));
        std::unique_ptr<OrtCUDAProviderOptionsV2, void (*)(OrtCUDAProviderOptionsV2*)>
            cudaGuard(cuda, [](OrtCUDAProviderOptionsV2* p) {
            Ort::GetApi().ReleaseCUDAProviderOptions(p);
                });

        const char* keys[] = { "device_id", "cudnn_conv_use_max_workspace" };
        const char* values[] = { "0", "1" };  // let cuDNN pick the fastest conv algo
        Ort::ThrowOnError(api.UpdateCUDAProviderOptions(
            cuda, keys, values, sizeof(keys) / sizeof(keys[0])));

        so.AppendExecutionProvider_CUDA_V2(*cuda);
        hardwareAccelerated = true;
        fmt::print("[{}] CUDA EP appended.\n", tag);
    }
    catch (const Ort::Exception& e) {
        fmt::print(stderr, "[{}] CUDA unavailable, falling back to CPU: {}\n", tag, e.what());
    }

#elif defined(ORT_EP_OPENVINO)
    // OpenVINO build
    // OpenVINO recompiles the graph internally; ORT-side fusions are wasted work
    // (or interfere), so disable them and let OpenVINO own the optimization.
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);

    const unsigned int physicalCores = std::max(1u, std::thread::hardware_concurrency() / 2);
    try {
        std::unordered_map<std::string, std::string> ov;
        ov["device_type"] = "CPU";   // "GPU,CPU" or "AUTO:GPU,CPU" to prefer an Intel GPU
        ov["precision"] = "FP32";  // "FP16" if the device supports it and accuracy holds
        ov["num_streams"] = "1";     // single stream: minimize single-frame latency
        ov["num_of_threads"] = std::to_string(physicalCores);

        so.AppendExecutionProvider_OpenVINO_V2(ov);
        hardwareAccelerated = true;
        fmt::print("[{}] OpenVINO EP appended (CPU, FP32, 1 stream, {} threads).\n", tag, physicalCores);
    }
    catch (const Ort::Exception& e) {
        // Fallback runs on the built-in CPU EP: re-enable ORT fusions, which are
        // exactly what we want on CPU. hardwareAccelerated stays false so the
        // unified CPU tuning block below applies threads/arena/denormals.
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        fmt::print(stderr, "[{}] OpenVINO unavailable, falling back to CPU: {}\n", tag, e.what());
    }

#elif defined(ORT_EP_CPU)
    // CPU build: built-in ORT CPU EP
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    fmt::print("[{}] CPU build: using the default ONNX Runtime CPU EP.\n", tag);

#else
#error "No execution provider selected: define one of ORT_EP_GPU, ORT_EP_OPENVINO or ORT_EP_CPU."
#endif

    // ---- Threading + numeric policy, driven by whether an accelerator loaded ----
    if (hardwareAccelerated) {
        // The accelerator runs the heavy compute on its own threads/streams;
        // a single ORT intra-op thread just feeds it and avoids context switching.
        so.SetIntraOpNumThreads(1);
    }
    else {
        // CPU path: the ORT intra-op pool actually runs the convolutions.
        const unsigned int physicalCores = std::max(1u, std::thread::hardware_concurrency() / 2);

        so.EnableCpuMemArena();                                     // avoid per-run OS allocations
        so.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);         // no inter-op locking overhead
        so.SetIntraOpNumThreads(static_cast<int>(physicalCores));   // physical cores, not logical

        // Flush denormals to zero ON THE ORT WORKER THREADS. Setting MXCSR by hand
        // (_MM_SET_FLUSH_ZERO_MODE / _MM_SET_DENORMALS_ZERO_MODE) only affects the
        // calling thread, not the intra-op pool that runs the conv kernels, so it
        // is effectively a no-op there. This config entry is applied by ORT to
        // every intra-op thread, which is what actually matters for CNNs.
        so.AddConfigEntry(kOrtSessionOptionsConfigSetDenormalAsZero, "1");

        // Optional real-time tuning: by default ORT busy-waits (spins) between ops
        // to shave latency. If this worker shares cores with the high-priority
        // frame producer, that spinning can add jitter. Uncomment to trade a bit
        // of latency for lower CPU contention (measure before keeping it):
        so.AddConfigEntry(kOrtSessionOptionsConfigAllowIntraOpSpinning, "0");
    }

    return hardwareAccelerated;
}