#include "ClassificationEngine.h"
#include "AsyncLogger.h"
#include <xmmintrin.h> 
#include <pmmintrin.h> 
#include <cstring>
#include <chrono>
#include <string>
#include <thread>
#include <unordered_map>
#include <algorithm>
#include <stdexcept>
#include "OrtsessionConfig.h"

#if defined(ORT_EP_GPU)
#include <cuda_runtime.h>
#endif

namespace {

    // Parser for the flat JSON string array. Avoids pulling a JSON dependency into the engine for a
    // single well-known, machine-generated field.
    std::vector<std::string> ParseJsonStringArray(const std::string& json)
    {
        std::vector<std::string> out;
        size_t i = json.find('[');
        if (i == std::string::npos) {
            throw std::runtime_error("names_ordered metadata is not a JSON array.");
        }

        for (++i; i < json.size(); ++i) {
            if (json[i] == ']') break;
            if (json[i] != '"') continue;

            std::string value;
            for (++i; i < json.size() && json[i] != '"'; ++i) {
                if (json[i] == '\\' && i + 1 < json.size()) ++i;
                value.push_back(json[i]);
            }
            out.push_back(std::move(value));
        }

        if (out.empty()) {
            throw std::runtime_error("names_ordered metadata contains no class names.");
        }
        return out;
    }

    std::string LookupMetadata(const Ort::ModelMetadata& meta,
        Ort::AllocatorWithDefaultOptions& allocator,
        const char* key,
        bool required)
    {
        Ort::AllocatedStringPtr value = meta.LookupCustomMetadataMapAllocated(key, allocator);
        if (value == nullptr) {
            if (required) {
                throw std::runtime_error(std::string("Missing required ONNX metadata key: ") + key);
            }
            return {};
        }
        return std::string(value.get());
    }

} // namespace

ClassificationEngine::ClassificationEngine()
{
    env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ClassificationEngineEnv");
    memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
}

ClassificationEngine::~ClassificationEngine() {
#if defined(ORT_EP_GPU)
    if (d_input) cudaFree(d_input);
    if (d_classId) cudaFree(d_classId);
    if (d_confidence) cudaFree(d_confidence);
#endif
}

void ClassificationEngine::LoadGraphContract()
{
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::ModelMetadata meta = session->GetModelMetadata();

    // --- Class names -------------------------------------------------------
    // The graph was exported without --embed-names, so it emits class_id only.
    // This vector is the sole index -> label mapping available at runtime.
    m_classNames = ParseJsonStringArray(LookupMetadata(meta, allocator, "names_ordered", true));

    // --- Normalization contract -------------------------------------------
    // 'none' means the graph performs /255 internally and expects raw [0,255]
    // floats. Feeding pre-normalized data would silently produce wrong
    // predictions, so refuse to run instead of guessing.
    const std::string normalization = LookupMetadata(meta, allocator, "normalization", true);
    if (normalization != "none") {
        throw std::runtime_error(
            "ONNX graph declares normalization='" + normalization +
            "' but this engine feeds raw [0,255] values. Re-export with --normalize none.");
    }

    // --- Geometry contract -------------------------------------------------
    const std::string imgszStr = LookupMetadata(meta, allocator, "imgsz", true);
    const int64_t graphImgsz = std::stoll(imgszStr);
    if (graphImgsz != modelHeight || graphImgsz != modelWidth) {
        throw std::runtime_error(
            "imgsz mismatch: metadata says " + imgszStr + " but the input tensor is " +
            std::to_string(modelWidth) + "x" + std::to_string(modelHeight) + ".");
    }

    // --- Colour order ------------------------------------------------------
    const std::string colorOrder = LookupMetadata(meta, allocator, "input_color_order", false);
    if (!colorOrder.empty() && colorOrder != "RGB") {
        throw std::runtime_error(
            "ONNX graph expects colour order '" + colorOrder + "'; this engine converts BGR->RGB.");
    }

    // --- Output resolution -------------------------------------------------
    // Resolve by name rather than by position: the export can emit class_id,
    // confidence, probs and class_name in an order that is not guaranteed.
    const bool hasClassId =
        std::find(outputNames.begin(), outputNames.end(), "class_id") != outputNames.end();
    const bool hasConfidence =
        std::find(outputNames.begin(), outputNames.end(), "confidence") != outputNames.end();

    if (!hasClassId || !hasConfidence) {
        throw std::runtime_error(
            "The ONNX graph does not expose both 'class_id' and 'confidence'. "
            "Export it with export_onnx.py, which folds argmax/max into the graph.");
    }

    m_outClassIdName = "class_id";
    m_outConfidenceName = "confidence";

    Log::Info("Classification contract: {} classes, imgsz {}, normalization {}, batch {}",
        m_classNames.size(), graphImgsz, normalization, inputShape[0]);
}


void ClassificationEngine::Initialize(const std::wstring& modelPath, const RT::CpuPartition& cpuPartition)
{
    Ort::SessionOptions sessionOptions;
    m_useGpuIoBinding = ConfigureOrtSessionOptions(sessionOptions, "Classification", cpuPartition);

    session = std::make_unique<Ort::Session>(*env, modelPath.c_str(), sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;
    inputName = session->GetInputNameAllocated(0, allocator).get();

    outputNames.clear();
    for (size_t idx = 0; idx < session->GetOutputCount(); ++idx) {
        outputNames.emplace_back(session->GetOutputNameAllocated(idx, allocator).get());
    }

    Ort::TypeInfo typeInfo = session->GetInputTypeInfo(0);
    auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
    inputShape = tensorInfo.GetShape();

    if (inputShape[0] <= 0) inputShape[0] = 1;
    if (inputShape[2] <= 0 || inputShape[3] <= 0) {
        Ort::ModelMetadata meta = session->GetModelMetadata();
        const std::string imgszStr = LookupMetadata(meta, allocator, "imgsz", true);
        const int64_t imgsz = std::stoll(imgszStr);
        if (inputShape[2] <= 0) inputShape[2] = imgsz;
        if (inputShape[3] <= 0) inputShape[3] = imgsz;
    }

    modelChannels = inputShape[1];
    modelHeight = inputShape[2];
    modelWidth = inputShape[3];

    LoadGraphContract();

    // Cache output shapes dynamically
    for (size_t i = 0; i < session->GetOutputCount(); i++) {
        std::string name = session->GetOutputNameAllocated(i, allocator).get();
        auto shape = session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        for (auto& d : shape) if (d < 0) d = 1;
        if (name == m_outClassIdName) m_classIdShape = shape;
        if (name == m_outConfidenceName) m_confShape = shape;
    }

    size_t totalElements = 1;
    for (int64_t dim : inputShape) totalElements *= dim;

    inputTensorValues.assign(totalElements, 0.0f);
    m_hostClassId.assign(1, 0);
    m_hostConfidence.assign(1, 0.0f);

    io_binding = std::make_unique<Ort::IoBinding>(*session);

#if defined(ORT_EP_GPU)
    if (m_useGpuIoBinding) {
        if (cudaMalloc(&d_input, totalElements * sizeof(float)) != cudaSuccess) throw std::runtime_error("CUDA Malloc Class input");
        if (cudaMalloc(&d_classId, sizeof(int64_t)) != cudaSuccess) throw std::runtime_error("CUDA Malloc Class ID");
        if (cudaMalloc(&d_confidence, sizeof(float)) != cudaSuccess) throw std::runtime_error("CUDA Malloc Class Conf");

        Ort::MemoryInfo cudaMemInfo("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);

        m_inputTensor = Ort::Value::CreateTensor<float>(cudaMemInfo, d_input, totalElements, inputShape.data(), inputShape.size());
        io_binding->BindInput(inputName.c_str(), m_inputTensor);

        m_classIdTensor = Ort::Value::CreateTensor<int64_t>(cudaMemInfo, d_classId, 1, m_classIdShape.data(), m_classIdShape.size());
        io_binding->BindOutput(m_outClassIdName.c_str(), m_classIdTensor);

        m_confidenceTensor = Ort::Value::CreateTensor<float>(cudaMemInfo, d_confidence, 1, m_confShape.data(), m_confShape.size());
        io_binding->BindOutput(m_outConfidenceName.c_str(), m_confidenceTensor);
    }
    else
#endif
    {
        m_inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(), totalElements, inputShape.data(), inputShape.size());
        io_binding->BindInput(inputName.c_str(), m_inputTensor);

        m_classIdTensor = Ort::Value::CreateTensor<int64_t>(memoryInfo, m_hostClassId.data(), 1, m_classIdShape.data(), m_classIdShape.size());
        io_binding->BindOutput(m_outClassIdName.c_str(), m_classIdTensor);

        m_confidenceTensor = Ort::Value::CreateTensor<float>(memoryInfo, m_hostConfidence.data(), 1, m_confShape.data(), m_confShape.size());
        io_binding->BindOutput(m_outConfidenceName.c_str(), m_confidenceTensor);
    }

    constexpr int kWarmupRuns = 50;
    double firstMs = 0.0, lastMs = 0.0;
    for (int r = 0; r < kWarmupRuns; ++r) {
        auto startTime = std::chrono::steady_clock::now();
        session->Run(Ort::RunOptions{ nullptr }, *io_binding);
        lastMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - startTime).count();
        if (r == 0) firstMs = lastMs;
    }
    Log::Info("Classification Warmup completed: {} runs, first {:.1f} ms, last {:.1f} ms.", kWarmupRuns, firstMs, lastMs);
}

void ClassificationEngine::Infer(const void* pInputImage, uint32_t width, uint32_t height, uint32_t channels,
    void* pOutputHeatmap, float& outAnomalyScore, std::string& outStatus)
{
    auto t0 = std::chrono::steady_clock::now();
    cv::Mat rawImg(height, width, CV_8UC3, const_cast<void*>(pInputImage));
    cv::resize(rawImg, m_resizeImage, cv::Size(static_cast<int>(modelWidth), static_cast<int>(modelHeight)), 0, 0, cv::INTER_LINEAR);
    cv::cvtColor(m_resizeImage, m_rgbImage, cv::COLOR_BGR2RGB);

    auto t1 = std::chrono::steady_clock::now();
    if (m_splitPlanes.empty()) m_splitPlanes.resize(modelChannels);

    const size_t planeSize = static_cast<size_t>(modelHeight) * static_cast<size_t>(modelWidth);
    cv::split(m_rgbImage, m_splitPlanes);

    for (int c = 0; c < modelChannels; ++c) {
        cv::Mat planeF32(static_cast<int>(modelHeight), static_cast<int>(modelWidth), CV_32FC1,
            inputTensorValues.data() + static_cast<size_t>(c) * planeSize);
        m_splitPlanes[c].convertTo(planeF32, CV_32FC1);
    }

    auto t2 = std::chrono::steady_clock::now();

#if defined(ORT_EP_GPU)
    if (m_useGpuIoBinding) {
        cudaMemcpy(d_input, inputTensorValues.data(), inputTensorValues.size() * sizeof(float), cudaMemcpyHostToDevice);
    }
#endif

    session->Run(Ort::RunOptions{ nullptr }, *io_binding);

#if defined(ORT_EP_GPU)
    if (m_useGpuIoBinding) {
        cudaMemcpy(m_hostClassId.data(), d_classId, sizeof(int64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(m_hostConfidence.data(), d_confidence, sizeof(float), cudaMemcpyDeviceToHost);
    }
#endif

    auto t3 = std::chrono::steady_clock::now();

    const int64_t classId = m_hostClassId[0];
    const float confidence = m_hostConfidence[0];

    if (classId < 0 || classId >= static_cast<int64_t>(m_classNames.size())) {
        throw std::runtime_error("class_id " + std::to_string(classId) + " out of range.");
    }

    auto t4 = std::chrono::steady_clock::now();

    outAnomalyScore = confidence;
    outStatus = m_classNames[static_cast<size_t>(classId)];

    using ms = std::chrono::duration<double, std::milli>;
    accResize += ms(t1 - t0).count();
    accNorm += ms(t2 - t1).count();
    accRun += ms(t3 - t2).count();
    accPost += ms(t4 - t3).count();
    maxRun = std::max(maxRun, ms(t3 - t2).count());

    if (++frameCount % 100 == 0) {
        Log::Info("avg over 100: resize {:.2f} | norm {:.2f} | RUN {:.2f} (max {:.2f}) | post {:.2f}",
            accResize / 100, accNorm / 100, accRun / 100, maxRun, accPost / 100);
        accResize = accNorm = accRun = accPost = maxRun = 0.0;
    }
}