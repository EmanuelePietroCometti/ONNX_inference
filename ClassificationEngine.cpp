#include "ClassificationEngine.h"
#include "AsyncLogger.h"
#include <xmmintrin.h> // For _MM_SET_FLUSH_ZERO_MODE
#include <pmmintrin.h> // For _MM_SET_DENORMALS_ZERO_MODE
#include <cstring>
#include <chrono>
#include <string>
#include <thread>
#include <unordered_map>
#include <algorithm>
#include <stdexcept>
#include "OrtsessionConfig.h"

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

    // Execution providers + session options: identical policy across all
    // engines. The CPU partition sizes and pins this session's intra-op pool.
    ConfigureOrtSessionOptions(sessionOptions, "Classification", cpuPartition);

    session = std::make_unique<Ort::Session>(*env, modelPath.c_str(), sessionOptions);

    // Extract I/O names dynamically from the model graph and store them for Infer
    // (no hardcoded names)
    Ort::AllocatorWithDefaultOptions allocator;
    inputName = session->GetInputNameAllocated(0, allocator).get();

    outputNames.clear();
    for (size_t idx = 0; idx < session->GetOutputCount(); ++idx) {
        outputNames.emplace_back(session->GetOutputNameAllocated(idx, allocator).get());
    }

    // Extract the input shape dynamically from the model graph and store it for Infer
    Ort::TypeInfo typeInfo = session->GetInputTypeInfo(0);
    auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
    inputShape = tensorInfo.GetShape();

    // Handle dynamic dimensions (exported as -1 in ONNX). Spatial dims must not
    // be guessed: a wrong fallback produces a running model with wrong results,
    // so the real value is read from the 'imgsz' metadata below and cross-checked.
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

    // Validate the deployment contract before allocating anything.
    LoadGraphContract();

    // Pre-allocate the input tensor buffer reused by every Infer call
    size_t totalElements = 1;
    for (int64_t dim : inputShape) {
        totalElements *= dim;
    }
    inputTensorValues.assign(totalElements, 0.0f);

    // Create the Ort::Value ONCE over the pre-allocated buffer: Infer refills
    // the buffer in place and reuses this same tensor on every frame
    m_inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorValues.size(), inputShape.data(), inputShape.size());

    const char* inputNames[] = { inputName.c_str() };
    const char* warmupOutputNames[] = { m_outClassIdName.c_str(), m_outConfidenceName.c_str() };

    // Multiple warmup runs: the first one pays graph compilation and lazy
    // allocations, the following ones stabilize the execution path. This runs at
    // Configure time, BEFORE the camera trigger starts, so the first real frame
    // does not hit a cold engine (the camera cannot be delayed in production).
    // Warm up exactly the output set Infer requests, otherwise a different graph
    // partition is compiled and the first real frame still pays for it.
    constexpr int kWarmupRuns = 50;
    double firstMs = 0.0;
    double lastMs = 0.0;
    for (int r = 0; r < kWarmupRuns; ++r) {
        auto startTime = std::chrono::steady_clock::now();
        session->Run(Ort::RunOptions{ nullptr }, inputNames, &m_inputTensor, 1, warmupOutputNames, 2);
        lastMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - startTime).count();
        if (r == 0) firstMs = lastMs;
    }

    Log::Info("Classification Warmup completed: {} runs, first {:.1f} ms, last {:.1f} ms.",
        kWarmupRuns, firstMs, lastMs);
}

void ClassificationEngine::Infer(const void* pInputImage, uint32_t width, uint32_t height, uint32_t channels,
    void* pOutputHeatmap, float& outAnomalyScore, std::string& outStatus)
{
    auto t0 = std::chrono::steady_clock::now();

    // Wrap the raw pointer into a cv::Mat without copying data
    cv::Mat rawImg(height, width, CV_8UC3, const_cast<void*>(pInputImage));

    // Bilinear, matching the training transform. INTER_NEAREST is marginally
    // faster but introduces aliasing the model never saw during training, which
    // matters for thin features such as a missing rivet.
    cv::resize(rawImg, m_resizeImage,
        cv::Size(static_cast<int>(modelWidth), static_cast<int>(modelHeight)),
        0, 0, cv::INTER_LINEAR);

    // The frame grabber delivers BGR; the graph declares RGB (input_color_order).
    cv::cvtColor(m_resizeImage, m_rgbImage, cv::COLOR_BGR2RGB);

    auto t1 = std::chrono::steady_clock::now();

    // NO normalization here. The exported graph already contains Div 255 and was
    // trained with mean=(0,0,0), std=(1,1,1) (Ultralytics classify_transforms
    // defaults). Applying ImageNet mean/std here would normalize twice and push
    // the input far outside the training distribution, silently.
    if (m_splitPlanes.empty()) {
        m_splitPlanes.resize(modelChannels);
    }

    // Vectorized HWC -> CHW conversion, plain uint8 -> float32 cast
    const size_t planeSize = static_cast<size_t>(modelHeight) * static_cast<size_t>(modelWidth);
    cv::split(m_rgbImage, m_splitPlanes);

    for (int c = 0; c < modelChannels; ++c) {
        // Wrapper Mat aliasing the contiguous tensor buffer: convertTo writes in-place
        cv::Mat planeF32(static_cast<int>(modelHeight), static_cast<int>(modelWidth), CV_32FC1,
            inputTensorValues.data() + static_cast<size_t>(c) * planeSize);
        m_splitPlanes[c].convertTo(planeF32, CV_32FC1);
    }

    // Zero-copy: m_inputTensor already wraps inputTensorValues, which the
    // conversion above refilled in place. Nothing to create per frame.
    const char* inputNamesC[] = { inputName.c_str() };

    // Request only the two outputs actually consumed: ORT prunes the rest of the
    // graph, so 'probs' is never materialized. Stack array, no shared state.
    const char* outputNamesC[] = { m_outClassIdName.c_str(), m_outConfidenceName.c_str() };

    auto t2 = std::chrono::steady_clock::now();

    // Execute inference
    auto outputTensors = session->Run(
        Ort::RunOptions{ nullptr }, inputNamesC, &m_inputTensor, 1, outputNamesC, 2);

    auto t3 = std::chrono::steady_clock::now();

    // Post-processing: argmax and max are already folded into the graph, so this
    // is a plain read of two scalars.
    const int64_t classId = outputTensors[0].GetTensorData<int64_t>()[0];
    const float confidence = outputTensors[1].GetTensorData<float>()[0];

    if (classId < 0 || classId >= static_cast<int64_t>(m_classNames.size())) {
        throw std::runtime_error(
            "class_id " + std::to_string(classId) + " out of range for " +
            std::to_string(m_classNames.size()) + " known classes.");
    }

    auto t4 = std::chrono::steady_clock::now();

    // Assign outputs for the WorkerONNX pipeline
    outAnomalyScore = confidence;
    outStatus = m_classNames[static_cast<size_t>(classId)];

    // Profiling and metrics
    using ms = std::chrono::duration<double, std::milli>;
    accResize += ms(t1 - t0).count();
    accNorm += ms(t2 - t1).count();
    accRun += ms(t3 - t2).count();
    accPost += ms(t4 - t3).count();
    maxRun = std::max(maxRun, ms(t3 - t2).count());

    if (++frameCount % 100 == 0) {
        Log::Info("avg over {}: resize {:.2f} | norm {:.2f} | RUN {:.2f} (max {:.2f}) | post {:.2f}",
            frameCount, accResize / frameCount, accNorm / frameCount,
            accRun / frameCount, maxRun, accPost / frameCount);
    }
}