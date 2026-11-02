#include "AnomalyEngine.h"
#include "AsyncLogger.h"
#include <xmmintrin.h> 
#include <pmmintrin.h> 
#include <algorithm>
#include <cmath>
#include <cstring>
#include <chrono>
#include <string>
#include <thread>
#include <unordered_map>
#include "OrtsessionConfig.h"
#include <array>
#include <stdexcept>

#if defined(ORT_EP_GPU)
#include <cuda_runtime.h>
#endif

AnomalyEngine::AnomalyEngine() : config(IniConfiguration::Instance())
{
    env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "AnomalyEngineEnv");
    memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
}

AnomalyEngine::~AnomalyEngine() {
#if defined(ORT_EP_GPU)
    // Safely free VRAM blocks allocated via CUDA to prevent memory leaks
    if (d_input) cudaFree(d_input);
    if (d_score) cudaFree(d_score);
    if (d_map) cudaFree(d_map);
#endif
}

void AnomalyEngine::Initialize(const std::wstring& modelPath, const RT::CpuPartition& cpuPartition)
{
    Ort::SessionOptions sessionOptions;

    // Check runtime hardware acceleration status to enable/disable CUDA bindings dynamically
    m_useGpuIoBinding = ConfigureOrtSessionOptions(sessionOptions, "Anomaly", cpuPartition);

    session = std::make_unique<Ort::Session>(*env, modelPath.c_str(), sessionOptions);
    Log::Info("ONNX Session created successfully for Anomaly Detection.");

    LoadContractMetadata();

    try {
        Ort::AllocatorWithDefaultOptions allocator;
        inputName = session->GetInputNameAllocated(0, allocator).get();

        outputNames.clear();
        for (size_t idx = 0; idx < session->GetOutputCount(); ++idx) {
            outputNames.emplace_back(session->GetOutputNameAllocated(idx, allocator).get());
        }

        // Extrapolate output indices BEFORE allocation
        for (size_t idx = 0; idx < session->GetOutputCount(); ++idx) {
            auto shape = session->GetOutputTypeInfo(idx).GetTensorTypeAndShapeInfo().GetShape();
            if (shape.size() == 1 || (shape.size() == 2 && shape[1] == 1)) {
                if (scoreIdx < 0) scoreIdx = static_cast<int>(idx);
            }
            else if (shape.size() >= 3) {
                if (mapIdx < 0) mapIdx = static_cast<int>(idx);
            }
        }
        if (scoreIdx < 0) throw std::runtime_error("Anomaly model missing scalar score output.");

        // Geometry extrapolation & sanitization
        Ort::TypeInfo typeInfo = session->GetInputTypeInfo(0);
        inputShape = typeInfo.GetTensorTypeAndShapeInfo().GetShape();
        if (inputShape[0] == -1) inputShape[0] = 1;

        modelChannels = inputShape[1];
        modelHeight = inputShape[2];
        modelWidth = inputShape[3];

        m_scoreShape = session->GetOutputTypeInfo(scoreIdx).GetTensorTypeAndShapeInfo().GetShape();
        for (auto& d : m_scoreShape) if (d < 0) d = 1;

        if (mapIdx >= 0) {
            m_mapShape = session->GetOutputTypeInfo(mapIdx).GetTensorTypeAndShapeInfo().GetShape();
            for (auto& d : m_mapShape) if (d < 0) d = 1;
        }

        // Calculate buffer sizes
        size_t inputElements = 1;
        for (int64_t d : inputShape) inputElements *= d;

        size_t mapElements = 1;
        if (mapIdx >= 0) {
            for (int64_t d : m_mapShape) mapElements *= d;
        }

        // Host allocations (required for pre-processing / fallback)
        inputTensorValues.assign(inputElements, 0.0f);
        m_splitPlanes.resize(modelChannels);
        m_hostScore.assign(1, 0.0f);
        if (mapIdx >= 0) m_hostMap.assign(mapElements, 0.0f);

        // Setup I/O binding
        io_binding = std::make_unique<Ort::IoBinding>(*session);

#if defined(ORT_EP_GPU)
        if (m_useGpuIoBinding) {
            // Pre-allocate GPU VRAM strictly for Zero-Copy tensor operation
            if (cudaMalloc(&d_input, inputElements * sizeof(float)) != cudaSuccess) throw std::runtime_error("CUDA Malloc input");
            if (cudaMalloc(&d_score, sizeof(float)) != cudaSuccess) throw std::runtime_error("CUDA Malloc score");
            if (mapIdx >= 0 && cudaMalloc(&d_map, mapElements * sizeof(float)) != cudaSuccess) throw std::runtime_error("CUDA Malloc map");

            Ort::MemoryInfo cudaMemInfo("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);

            m_inputTensor = Ort::Value::CreateTensor<float>(cudaMemInfo, d_input, inputElements, inputShape.data(), inputShape.size());
            io_binding->BindInput(inputName.c_str(), m_inputTensor);

            m_scoreTensor = Ort::Value::CreateTensor<float>(cudaMemInfo, d_score, 1, m_scoreShape.data(), m_scoreShape.size());
            io_binding->BindOutput(outputNames[scoreIdx].c_str(), m_scoreTensor);

            if (mapIdx >= 0) {
                m_mapTensor = Ort::Value::CreateTensor<float>(cudaMemInfo, d_map, mapElements, m_mapShape.data(), m_mapShape.size());
                io_binding->BindOutput(outputNames[mapIdx].c_str(), m_mapTensor);
            }
            Log::Info("GPU I/O Binding established with Zero-Copy VRAM tensors.");
        }
        else
#endif
        {
            // Fallback: CPU Direct Memory Mapping
            m_inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(), inputElements, inputShape.data(), inputShape.size());
            io_binding->BindInput(inputName.c_str(), m_inputTensor);

            m_scoreTensor = Ort::Value::CreateTensor<float>(memoryInfo, m_hostScore.data(), 1, m_scoreShape.data(), m_scoreShape.size());
            io_binding->BindOutput(outputNames[scoreIdx].c_str(), m_scoreTensor);

            if (mapIdx >= 0) {
                m_mapTensor = Ort::Value::CreateTensor<float>(memoryInfo, m_hostMap.data(), mapElements, m_mapShape.data(), m_mapShape.size());
                io_binding->BindOutput(outputNames[mapIdx].c_str(), m_mapTensor);
            }
            Log::Info("CPU I/O Binding established with System RAM mapping.");
        }

        // Warmup runs
        double firstMs = 0.0, lastMs = 0.0;
        for (int r = 0; r < config.GetONNXWarmUps(); ++r) {
            auto startTime = std::chrono::steady_clock::now();
            session->Run(Ort::RunOptions{ nullptr }, *io_binding);
            lastMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - startTime).count();
            if (r == 0) firstMs = lastMs;
        }

        Log::Info("Warmup completed: {} runs, first {:.1f} ms, last {:.1f} ms. Engine locked.", config.GetONNXWarmUps(), firstMs, lastMs);
    }
    catch (const Ort::Exception& e) {
        Log::Error("### ERROR DURING INITIALIZATION: {}", e.what());
        throw;
    }
}

void AnomalyEngine::LoadContractMetadata()
{
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::ModelMetadata metadata = session->GetModelMetadata();

    auto lookupString = [&](const char* key) -> std::string {
        auto value = metadata.LookupCustomMetadataMapAllocated(key, allocator);
        return value ? std::string(value.get()) : std::string();
        };
    auto lookupFloat = [&](const char* key, float& target) -> bool {
        auto value = metadata.LookupCustomMetadataMapAllocated(key, allocator);
        if (value) {
            target = std::stof(value.get());
            Log::Info("Metadata '{}' = {}", key, target);
            return true;
        }
        return false;
        };

    // Verify contract version presence
    const std::string contractVersion = lookupString("contract_version");
    if (contractVersion.empty()) {
        Log::Warning("Metadata 'contract_version' missing. Attempting to proceed assuming backwards compatibility.");
    }
    else {
        Log::Info("ONNX Contract Version validated: {}", contractVersion);
    }

    // Normalization check: updated key to match new Python exporter
    const std::string normalization = lookupString("normalize_inside_graph");
    normalizationInGraph = (normalization == "true");
    if (!normalizationInGraph) {
        Log::Warning("### WARNING: normalize_inside_graph='{}' (expected 'true'). Host will apply ImageNet normalization as fallback.", normalization);
    }

    // Channel order: the graph is trained on RGB (ImageNet). OpenCV decodes BMPs
    // as BGR, so honor preproc_color_conversion and swap channels before feeding
    // the model. Missing key defaults to bgr2rgb (all our exporter's models).
    const std::string colorConversion = lookupString("preproc_color_conversion");
    m_convertBgrToRgb = (colorConversion == "bgr2rgb");
    Log::Info("Metadata 'preproc_color_conversion' = '{}' -> BGR2RGB swap {}.", colorConversion.empty() ? "(missing, default)" : colorConversion, m_convertBgrToRgb ? "ENABLED" : "disabled");

    // Calibration Status Verification
    const std::string calibratedStatus = lookupString("calibrated");
    if (calibratedStatus != "true") {
        Log::Warning("Model 'calibrated' flag is not true. Using uncalibrated raw tensor bounds.");
    }

    // Load bounds and thresholds using the new standard keys
    const bool hasMin = lookupFloat("map_min_raw", mapMin);
    const bool hasMax = lookupFloat("map_max_raw", mapMax);
    const bool hasThreshold = lookupFloat("image_threshold_raw", scoreThreshold);
    hasPixelThreshold = lookupFloat("pixel_threshold_raw", pixelThreshold);

    if (!hasMin || !hasMax || !hasThreshold) {
        throw std::runtime_error(
            "Anomaly model rejected: calibration metadata is incomplete "
            "(map_min_raw/map_max_raw/image_threshold_raw). Re-run calibrate.py.");
    }

    if (mapMax - mapMin <= 0.0f) {
        throw std::runtime_error("Invalid calibration metadata: map_max_raw must be greater than map_min_raw.");
    }

    // Evaluate normalization formula to ensure C++ rendering matches Python behavior
    const std::string normFormula = lookupString("normalization_formula");
    if (!normFormula.empty() && normFormula != "minmax") {
        Log::Warning("Model exported with normalization_formula='{}'. C++ engine uses 'minmax' natively. Visual results may drift.", normFormula);
    }

    Log::Info("Contract limits verified. mapMin={}, mapMax={}, scoreThreshold={}, pixelThreshold={} (present={}), normalizationInGraph={}.",
        mapMin, mapMax, scoreThreshold, pixelThreshold, hasPixelThreshold, normalizationInGraph);
}

namespace {
    // Triangle-filter (bilinear) resample coefficients for ONE axis. Mirrors
    // inference_simulation/src/antialias_resize.py::precompute_coeffs bit-for-bit:
    // on a downscale the filter support widens by the scale factor (antialiasing).
    void PrecomputeCoeffs(int inSize, int outSize,
        std::vector<int>& bounds, std::vector<double>& weights, int& ksize)
    {
        const double scale = static_cast<double>(inSize) / static_cast<double>(outSize);
        const double filterscale = scale >= 1.0 ? scale : 1.0;   // widen on downscale
        const double support = 1.0 * filterscale;                // bilinear half-width
        const double ss = 1.0 / filterscale;

        ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
        bounds.assign(outSize, 0);
        weights.assign(static_cast<size_t>(outSize) * ksize, 0.0);

        for (int o = 0; o < outSize; ++o) {
            const double center = (o + 0.5) * scale;
            int xmin = static_cast<int>(center - support + 0.5);
            if (xmin < 0) xmin = 0;
            int xmax = static_cast<int>(center + support + 0.5);
            if (xmax > inSize) xmax = inSize;
            const int n = xmax - xmin;
            double total = 0.0;
            for (int t = 0; t < n; ++t) {
                double w = 1.0 - std::abs((xmin + t - center + 0.5) * ss);  // bilinear_filter
                if (w < 0.0) w = 0.0;
                weights[static_cast<size_t>(o) * ksize + t] = w;
                total += w;
            }
            if (total > 0.0)
                for (int t = 0; t < n; ++t) weights[static_cast<size_t>(o) * ksize + t] /= total;
            bounds[o] = xmin;
        }
    }

    inline uint8_t RoundClipU8(double v) {
        double r = std::floor(v + 0.5);   // round half up (pixels are non-negative)
        if (r < 0.0) r = 0.0;
        if (r > 255.0) r = 255.0;
        return static_cast<uint8_t>(r);
    }
}

void AnomalyEngine::BuildResizeCoeffs(int inW, int inH)
{
    PrecomputeCoeffs(inW, static_cast<int>(modelWidth), m_hBounds, m_hWeights, m_hK);
    PrecomputeCoeffs(inH, static_cast<int>(modelHeight), m_vBounds, m_vWeights, m_vK);
    m_rsInW = inW;
    m_rsInH = inH;
}

// Antialiased bilinear resize of a CV_8UC3 image to (modelWidth x modelHeight).
// Separable: horizontal pass -> uint8 round/clip -> vertical pass -> uint8. Double
// accumulation to match the Python reference exactly.
void AnomalyEngine::ResizeAntialias(const cv::Mat& src, cv::Mat& dst)
{
    const int inH = src.rows, inW = src.cols;
    const int outW = static_cast<int>(modelWidth), outH = static_cast<int>(modelHeight);
    if (inW != m_rsInW || inH != m_rsInH) BuildResizeCoeffs(inW, inH);

    // Horizontal: [inH x inW] -> [inH x outW]
    m_hpass.create(inH, outW, CV_8UC3);
    for (int y = 0; y < inH; ++y) {
        const uint8_t* srow = src.ptr<uint8_t>(y);
        uint8_t* hrow = m_hpass.ptr<uint8_t>(y);
        for (int o = 0; o < outW; ++o) {
            const int s = m_hBounds[o];
            const int avail = std::min(m_hK, inW - s);
            const double* w = &m_hWeights[static_cast<size_t>(o) * m_hK];
            double a0 = 0.0, a1 = 0.0, a2 = 0.0;
            for (int t = 0; t < avail; ++t) {
                const uint8_t* px = srow + static_cast<size_t>(s + t) * 3;
                const double wt = w[t];
                a0 += wt * px[0]; a1 += wt * px[1]; a2 += wt * px[2];
            }
            uint8_t* op = hrow + static_cast<size_t>(o) * 3;
            op[0] = RoundClipU8(a0); op[1] = RoundClipU8(a1); op[2] = RoundClipU8(a2);
        }
    }

    // Vertical: [inH x outW] -> [outH x outW]
    dst.create(outH, outW, CV_8UC3);
    for (int o = 0; o < outH; ++o) {
        const int s = m_vBounds[o];
        const int avail = std::min(m_vK, inH - s);
        const double* w = &m_vWeights[static_cast<size_t>(o) * m_vK];
        uint8_t* drow = dst.ptr<uint8_t>(o);
        for (int x = 0; x < outW; ++x) {
            double a0 = 0.0, a1 = 0.0, a2 = 0.0;
            for (int t = 0; t < avail; ++t) {
                const uint8_t* px = m_hpass.ptr<uint8_t>(s + t) + static_cast<size_t>(x) * 3;
                const double wt = w[t];
                a0 += wt * px[0]; a1 += wt * px[1]; a2 += wt * px[2];
            }
            uint8_t* op = drow + static_cast<size_t>(x) * 3;
            op[0] = RoundClipU8(a0); op[1] = RoundClipU8(a1); op[2] = RoundClipU8(a2);
        }
    }
}

void AnomalyEngine::Infer(const void* pInputImage, uint32_t width, uint32_t height,
    uint32_t channels, void* pOutputHeatmap, float& outAnomalyScore, std::string& outStatus, DWORD idPunto)
{
    int type = 0;
    switch (channels)
    {
    case 1:
      type = CV_8UC1;
      break;
    case 3:
      type = CV_8UC3;
      break;
    default:
      throw std::invalid_argument("OpenCV ERROR: Unsupported number of channels (" + std::to_string(channels) + "). Expected 1 or 3.");
    }
    cv::Mat rawImg(height, width, type, const_cast<void*>(pInputImage));
    // Antialiased bilinear downscale to the model size — matches training/reference
    // (inference_simulation) exactly. cv::INTER_AREA does NOT and inflated scores.
    const cv::Mat* pSrc = &rawImg;
    if (channels == 1) {
      cv::cvtColor(rawImg, m_bgrImage, cv::COLOR_GRAY2BGR);
      pSrc = &m_bgrImage;
    }
    ResizeAntialias(*pSrc, m_resizeImage);

    // The model expects RGB (ImageNet order); OpenCV holds BGR. Swap in place so
    // cv::split below yields planes [R, G, B] aligned with the graph and mean/std.
    if (m_convertBgrToRgb && modelChannels == 3) {
        cv::cvtColor(m_resizeImage, m_resizeImage, cv::COLOR_BGR2RGB);
    }

    const float mean[] = { 0.485f, 0.456f, 0.406f };
    const float stddev[] = { 0.229f, 0.224f, 0.225f };
    const size_t planeSize = static_cast<size_t>(modelHeight) * static_cast<size_t>(modelWidth);

    cv::split(m_resizeImage, m_splitPlanes);
    for (int c = 0; c < modelChannels; ++c) {
        const float scale = normalizationInGraph ? (1.0f / 255.0f) : (1.0f / (255.0f * stddev[c]));
        const float offset = normalizationInGraph ? 0.0f : (mean[c] / stddev[c]);

        // Write processed data directly into host buffer
        cv::Mat planeF32(static_cast<int>(modelHeight), static_cast<int>(modelWidth), CV_32FC1, inputTensorValues.data() + static_cast<size_t>(c) * planeSize);
        m_splitPlanes[c].convertTo(planeF32, CV_32FC1, scale, -offset);
    }

#if defined(ORT_EP_GPU)
    // Synchronous memory transfer: Only triggers if hardware acceleration is active
    if (m_useGpuIoBinding) {
        cudaMemcpy(d_input, inputTensorValues.data(), inputTensorValues.size() * sizeof(float), cudaMemcpyHostToDevice);
    }
#endif

    // Run inference matching exactly the bound VRAM/RAM buffers
    session->Run(Ort::RunOptions{ nullptr }, *io_binding);

#if defined(ORT_EP_GPU)
    if (m_useGpuIoBinding) {
        // Retrieve processed buffers from VRAM
        cudaMemcpy(m_hostScore.data(), d_score, sizeof(float), cudaMemcpyDeviceToHost);
        if (mapIdx >= 0) {
            cudaMemcpy(m_hostMap.data(), d_map, m_hostMap.size() * sizeof(float), cudaMemcpyDeviceToHost);
        }
    }
#endif

    // Score comparison evaluated natively against image_threshold_raw
    const float rawScore = m_hostScore[0];
    outAnomalyScore = rawScore;
    outStatus = (rawScore >= scoreThreshold) ? "REJECT" : "OK";

    if (writeHeatmap && pOutputHeatmap && mapIdx >= 0) {
      // raw anomaly map, casted to 8 bit with calibration values
      cv::Mat floatHeatmap(static_cast<int>(m_mapShape[2]), static_cast<int>(m_mapShape[3]), CV_32FC1, m_hostMap.data());

      const double alpha = 255.0 / (mapMax - mapMin);
      const double beta = -255.0 * mapMin / (mapMax - mapMin);
      floatHeatmap.convertTo(m_uint8Heatmap, CV_8UC1, alpha, beta);

      // output with the same dimension od the ROI
      cv::resize(m_uint8Heatmap, m_largeHeatmap, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);

      memcpy(pOutputHeatmap, m_largeHeatmap.data, static_cast<size_t>(width) * height);
    }
}