#pragma once

// Export contract of the ONNX models produced by PyTorchTrainingTool.
//
// A single module reads and validates the model metadata for BOTH engines:
//   - contract version
//   - preprocessing: cv::resize interpolation, stretch, no crop, no antialias, colour order
//   - anomaly: calibration, normalisation formula, map -> uint8 conversion
//   - classification: class names, imgsz, input normalisation
// The engines only consume the structs returned here. Every violation throws
// std::runtime_error with an explicit message: a model that does not honour the contract
// is rejected at load time, never run with silently different preprocessing.

#include <cstdint>
#include <functional>
#include <string>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <onnxruntime_cxx_api.h>

namespace Contract
{
	// Major version supported by the engines (metadata 'contract_version', "<major>.<minor>").
	constexpr int kSupportedContractMajor = 1;

	// Read-only view of the model metadata (key -> string). Missing key -> empty string.
	class MetadataReader
	{
	public:
		// lookup(key, value): returns false when the key is not in the model.
		using Lookup = std::function<bool(const char* key, std::string& value)>;

		explicit MetadataReader(Lookup lookup);

		// Reads the custom metadata map of an ONNX Runtime session.
		static MetadataReader FromSession(const Ort::Session& session);

		std::string Get(const char* key) const;          // empty when missing
		std::string Require(const char* key) const;      // throws when missing or empty
		float RequireFloat(const char* key) const;       // throws when missing or not a number
		int64_t RequireInt(const char* key) const;       // throws when missing or not an integer

	private:
		Lookup m_lookup;
	};

	// ---- Preprocessing (shared by both engines) -------------------------------------------
	struct Preprocessing
	{
		int cvInterpolation = cv::INTER_LINEAR;   // flag for cv::resize
		std::string interpolation = "bilinear";   // metadata value: nearest|bilinear|bicubic|area|lanczos4
		bool bgrToRgb = true;                     // swap channels after the resize (OpenCV frames are BGR)
	};

	// Checks that the engine can reproduce the training preprocessing and returns it.
	// interpolationRequired: true -> a missing 'preproc_resize_interpolation' is an error
	// (anomaly: the previous engine used a different resize, so a silent default would shift scores);
	// false -> falls back to bilinear with a warning (classification models exported before the contract).
	Preprocessing LoadPreprocessing(const MetadataReader& meta, int64_t modelWidth, int64_t modelHeight,
		bool interpolationRequired);

	// ---- Anomaly --------------------------------------------------------------------------
	struct AnomalyCalibration
	{
		std::string normalizationFormula;

		float classificationMinRaw = 0.0f;
		float classificationMaxRaw = 1.0f;
		float classificationThresholdRaw = 0.5f;
		float classificationThresholdNorm = 0.5f;

		float segmentationMinRaw = 0.0f;
		float segmentationMaxRaw = 1.0f;
		float segmentationThresholdRaw = 0.5f;
		float segmentationThresholdNorm = 0.5f;
	};

	struct AnomalyContract
	{
		std::string version;
		Preprocessing preprocessing;
		AnomalyCalibration calibration;
		bool centeredNormalization = false;   // true: anomalib_centered, false: minmax
		double segmentationMapAlpha = 0.0;    // raw -> uint8: pixel = alpha * raw + beta
		double segmentationMapBeta = 0.0;
	};

	AnomalyContract LoadAnomalyContract(const MetadataReader& meta, int64_t modelWidth, int64_t modelHeight);

	// ---- Classification -------------------------------------------------------------------
	struct ClassificationContract
	{
		std::string version;                  // empty for models exported before the contract existed
		Preprocessing preprocessing;
		int64_t imgsz = 0;
		std::vector<std::string> classNames;  // index = class_id
		std::string namesJson;                // {"0":"label0","1":"label1",...}, echoed to the client
	};

	ClassificationContract LoadClassificationContract(const MetadataReader& meta, int64_t modelWidth, int64_t modelHeight);

	// ---- Helpers (public so they can be unit tested) --------------------------------------
	std::vector<std::string> ParseJsonStringArray(const std::string& json);
	std::string BuildNamesJson(const std::vector<std::string>& names);
}