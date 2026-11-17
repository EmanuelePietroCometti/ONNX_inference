#include "ContractMetadata.h"
#include "AsyncLogger.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace Contract
{
	namespace
	{
		[[noreturn]] void Reject(const std::string& message)
		{
			throw std::runtime_error(message);
		}

		int ToCvInterpolation(const std::string& name)
		{
			if (name == "nearest")  return cv::INTER_NEAREST;
			if (name == "bilinear") return cv::INTER_LINEAR;
			if (name == "bicubic")  return cv::INTER_CUBIC;
			if (name == "area")     return cv::INTER_AREA;
			if (name == "lanczos4") return cv::INTER_LANCZOS4;
			Reject("Model rejected: unsupported preproc_resize_interpolation '" + name +
				"' (expected nearest, bilinear, bicubic, area or lanczos4).");
		}

		int64_t ParseInt(const std::string& key, const std::string& text)
		{
			try {
				size_t used = 0;
				const long long value = std::stoll(text, &used);
				if (used != text.size()) {
					throw std::invalid_argument(text);
				}
				return static_cast<int64_t>(value);
			}
			catch (const std::exception&) {
				Reject("Model rejected: metadata '" + key + "' = '" + text + "' is not an integer.");
			}
		}

		// Major version of "<major>.<minor>"; -1 when it cannot be parsed.
		int MajorOf(const std::string& version)
		{
			try {
				return std::stoi(version.substr(0, version.find('.')));
			}
			catch (const std::exception&) {
				return -1;
			}
		}

		// Contract version: always checked when present, mandatory when 'required'.
		std::string CheckVersion(const MetadataReader& meta, bool required)
		{
			const std::string version = meta.Get("contract_version");
			if (version.empty()) {
				if (required) {
					Reject("Model rejected: metadata 'contract_version' is missing.");
				}
				return version;
			}
			if (MajorOf(version) != kSupportedContractMajor) {
				Reject("Model rejected: contract_version '" + version + "' is not supported (expected major " +
					std::to_string(kSupportedContractMajor) + ").");
			}
			Log::Info("ONNX Contract Version validated: {}", version);
			return version;
		}

		void AppendUtf8(std::string& out, uint32_t cp)
		{
			if (cp < 0x80) {
				out.push_back(static_cast<char>(cp));
			}
			else if (cp < 0x800) {
				out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
				out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
			}
			else if (cp < 0x10000) {
				out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
				out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
				out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
			}
			else {
				out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
				out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
				out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
				out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
			}
		}

		// Four hex digits starting at json[pos] (the XXXX of a \uXXXX escape).
		uint32_t ParseHex4(const std::string& json, size_t pos)
		{
			if (pos + 4 > json.size()) {
				Reject("Model rejected: names_ordered has a truncated \\u escape.");
			}
			uint32_t value = 0;
			for (size_t k = 0; k < 4; ++k) {
				const char c = json[pos + k];
				value <<= 4;
				if (c >= '0' && c <= '9')      value |= static_cast<uint32_t>(c - '0');
				else if (c >= 'a' && c <= 'f') value |= static_cast<uint32_t>(c - 'a' + 10);
				else if (c >= 'A' && c <= 'F') value |= static_cast<uint32_t>(c - 'A' + 10);
				else Reject("Model rejected: names_ordered has an invalid \\u escape.");
			}
			return value;
		}
	}

	// ---- MetadataReader -------------------------------------------------------------------
	MetadataReader::MetadataReader(Lookup lookup)
		: m_lookup(std::move(lookup))
	{
	}

	MetadataReader MetadataReader::FromSession(const Ort::Session& session)
	{
		// The metadata object and the allocator must outlive every lookup: the reader owns them.
		struct Holder
		{
			Ort::AllocatorWithDefaultOptions allocator;
			Ort::ModelMetadata metadata;
			explicit Holder(const Ort::Session& s) : metadata(s.GetModelMetadata()) {}
		};
		auto holder = std::make_shared<Holder>(session);
		return MetadataReader([holder](const char* key, std::string& value) {
			auto text = holder->metadata.LookupCustomMetadataMapAllocated(key, holder->allocator);
			if (!text) {
				return false;
			}
			value = text.get();
			return true;
			});
	}

	std::string MetadataReader::Get(const char* key) const
	{
		std::string value;
		return m_lookup(key, value) ? value : std::string();
	}

	std::string MetadataReader::Require(const char* key) const
	{
		std::string value = Get(key);
		if (value.empty()) {
			Reject(std::string("Model rejected: required metadata '") + key + "' is missing.");
		}
		return value;
	}

	float MetadataReader::RequireFloat(const char* key) const
	{
		const std::string text = Require(key);
		try {
			size_t used = 0;
			const float value = std::stof(text, &used);
			if (used != text.size() || !std::isfinite(value)) {
				throw std::invalid_argument(text);
			}
			return value;
		}
		catch (const std::exception&) {
			Reject(std::string("Model rejected: metadata '") + key + "' = '" + text + "' is not a number.");
		}
	}

	int64_t MetadataReader::RequireInt(const char* key) const
	{
		return ParseInt(key, Require(key));
	}

	// ---- Preprocessing --------------------------------------------------------------------
	Preprocessing LoadPreprocessing(const MetadataReader& meta, int64_t modelWidth, int64_t modelHeight,
		bool interpolationRequired)
	{
		Preprocessing p;

		const std::string interpolation = meta.Get("preproc_resize_interpolation");
		if (interpolation.empty()) {
			if (interpolationRequired) {
				Reject("Model rejected: metadata 'preproc_resize_interpolation' is missing. "
					"Re-export the model with PyTorchTrainingTool.");
			}
			Log::Warning("Metadata 'preproc_resize_interpolation' is missing: using bilinear (cv::INTER_LINEAR).");
		}
		else {
			p.interpolation = interpolation;
			p.cvInterpolation = ToCvInterpolation(interpolation);
		}

		// What this engine really does: plain cv::resize, stretch, no crop. Anything else would
		// change the pixels the model sees compared with training, so it is refused.
		const std::string backend = meta.Get("preproc_resize_backend");
		if (!backend.empty() && backend != "opencv") {
			Reject("Model rejected: preproc_resize_backend='" + backend + "' (the engine resizes with OpenCV).");
		}
		const std::string antialias = meta.Get("preproc_antialias");
		if (!antialias.empty() && antialias != "false") {
			Reject("Model rejected: preproc_antialias='" + antialias + "' but the engine uses plain cv::resize (no antialias).");
		}
		const std::string crop = meta.Get("preproc_center_crop");
		if (!crop.empty() && crop != "none") {
			Reject("Model rejected: preproc_center_crop='" + crop + "' is not supported by the engine.");
		}
		const std::string aspect = meta.Get("preproc_aspect_ratio_policy");
		if (!aspect.empty() && aspect != "stretch") {
			Reject("Model rejected: preproc_aspect_ratio_policy='" + aspect + "' (the engine stretches the frame).");
		}

		// Colour order: OpenCV decodes BGR, the graph was trained on RGB.
		const std::string conversion = meta.Get("preproc_color_conversion");
		if (conversion.empty() || conversion == "bgr2rgb") {
			p.bgrToRgb = true;
		}
		else if (conversion == "none") {
			p.bgrToRgb = false;
		}
		else {
			Reject("Model rejected: preproc_color_conversion='" + conversion + "' (expected bgr2rgb or none).");
		}

		// The resize target written at training time must be the static input of the graph.
		const std::string resizeHeight = meta.Get("preproc_resize_height");
		const std::string resizeWidth = meta.Get("preproc_resize_width");
		if ((!resizeHeight.empty() && ParseInt("preproc_resize_height", resizeHeight) != modelHeight) ||
			(!resizeWidth.empty() && ParseInt("preproc_resize_width", resizeWidth) != modelWidth)) {
			Reject("Model rejected: preproc_resize_height/width (" + resizeHeight + "x" + resizeWidth +
				") differ from the input tensor (" + std::to_string(modelHeight) + "x" + std::to_string(modelWidth) + ").");
		}

		Log::Info("Preprocessing contract verified: cv::resize '{}' to {}x{} (stretch, no crop, no antialias), BGR->RGB {}.",
			p.interpolation, modelWidth, modelHeight, p.bgrToRgb ? "enabled" : "disabled");
		return p;
	}

	// ---- Anomaly --------------------------------------------------------------------------
	AnomalyContract LoadAnomalyContract(const MetadataReader& meta, int64_t modelWidth, int64_t modelHeight)
	{
		AnomalyContract c;
		c.version = CheckVersion(meta, /*required=*/true);

		const std::string calibrated = meta.Get("calibrated");
		if (calibrated != "true") {
			Reject("Anomaly model rejected: metadata 'calibrated' is '" + calibrated +
				"' (expected 'true'). Run the calibrate stage before exporting.");
		}
		const std::string normalization = meta.Get("normalize_inside_graph");
		if (normalization != "true") {
			Reject("Anomaly model rejected: normalize_inside_graph='" + normalization + "' (the contract requires 'true').");
		}
		// The engine converts uint8 -> float with a 1/255 scale; the graph does mean/std itself.
		const std::string scale = meta.Get("preproc_scale");
		if (!scale.empty() && scale != "1/255") {
			Reject("Anomaly model rejected: preproc_scale='" + scale + "' (the engine feeds pixels scaled by 1/255).");
		}
		const std::string colorOrder = meta.Get("input_color_order");
		if (!colorOrder.empty() && colorOrder != "RGB") {
			Reject("Anomaly model rejected: input_color_order='" + colorOrder + "' (the engine feeds RGB).");
		}

		c.preprocessing = LoadPreprocessing(meta, modelWidth, modelHeight, /*interpolationRequired=*/true);

		auto requireFloat = [&](const char* key) -> float {
			const float value = meta.RequireFloat(key);
			Log::Info("Metadata '{}' = {}", key, value);
			return value;
			};
		AnomalyCalibration& cal = c.calibration;
		cal.classificationThresholdRaw = requireFloat("image_threshold_raw");
		cal.segmentationThresholdRaw = requireFloat("pixel_threshold_raw");
		cal.classificationMinRaw = requireFloat("score_min_raw");
		cal.classificationMaxRaw = requireFloat("score_max_raw");
		cal.segmentationMinRaw = requireFloat("map_min_raw");
		cal.segmentationMaxRaw = requireFloat("map_max_raw");

		const double classificationSpan = static_cast<double>(cal.classificationMaxRaw) - cal.classificationMinRaw;
		const double segmentationSpan = static_cast<double>(cal.segmentationMaxRaw) - cal.segmentationMinRaw;
		if (classificationSpan <= 0.0) {
			Reject("Invalid calibration metadata: score_max_raw must be greater than score_min_raw.");
		}
		if (segmentationSpan <= 0.0) {
			Reject("Invalid calibration metadata: map_max_raw must be greater than map_min_raw.");
		}

		const std::string formula = meta.Get("normalization_formula");
		if (formula == "anomalib_centered") {
			c.centeredNormalization = true;
		}
		else if (formula == "minmax") {
			c.centeredNormalization = false;
		}
		else {
			Reject("Anomaly model rejected: normalization_formula '" + formula +
				"' is missing or not supported (expected 'minmax' or 'anomalib_centered').");
		}
		cal.normalizationFormula = formula;

		// Normalized thresholds, same rule as the exporter's normalized_threshold():
		//	- anomalib_centered always maps the threshold to 0.5.
		//	- minmax maps the threshold linearly between minRaw and maxRaw.
		auto thresholdNorm = [&](double threshold, double minRaw, double maxRaw) -> float {
			if (c.centeredNormalization) {
				return 0.5f;
			}
			const double span = maxRaw - minRaw;
			if (std::abs(span) < 1e-9) {
				return 0.0f;
			}
			return static_cast<float>(std::clamp((threshold - minRaw) / span, 0.0, 1.0));
			};
		cal.classificationThresholdNorm = thresholdNorm(cal.classificationThresholdRaw,
			cal.classificationMinRaw, cal.classificationMaxRaw);
		cal.segmentationThresholdNorm = thresholdNorm(cal.segmentationThresholdRaw,
			cal.segmentationMinRaw, cal.segmentationMaxRaw);

		// The exporter also stores the normalized thresholds: if they drifted from the raw
		// calibration, the value derived here wins.
		auto warnIfStale = [&](const char* key, float derived) {
			const std::string stored = meta.Get(key);
			if (stored.empty()) {
				return;
			}
			const float storedValue = meta.RequireFloat(key);
			if (std::fabs(storedValue - derived) > 1e-4f) {
				Log::Warning("Metadata '{}' = {} differs from the value derived from the raw calibration ({}); using the derived value.",
					key, storedValue, derived);
			}
			};
		warnIfStale("image_threshold_norm", cal.classificationThresholdNorm);
		warnIfStale("pixel_threshold_norm", cal.segmentationThresholdNorm);

		// raw float map -> uint8 (alpha * raw + beta)
		c.segmentationMapAlpha = 255.0 / segmentationSpan;
		c.segmentationMapBeta = c.centeredNormalization
			? 255.0 * (0.5 - static_cast<double>(cal.segmentationThresholdRaw) / segmentationSpan)
			: -255.0 * static_cast<double>(cal.segmentationMinRaw) / segmentationSpan;

		Log::Info("Contract limits verified (formula '{}'). classification: min={}, max={}, threshold={} (norm {}); "
			"segmentation: min={}, max={}, threshold={} (norm {}); map->uint8 alpha={:.5f}, beta={:.5f}.",
			cal.normalizationFormula,
			cal.classificationMinRaw, cal.classificationMaxRaw,
			cal.classificationThresholdRaw, cal.classificationThresholdNorm,
			cal.segmentationMinRaw, cal.segmentationMaxRaw,
			cal.segmentationThresholdRaw, cal.segmentationThresholdNorm,
			c.segmentationMapAlpha, c.segmentationMapBeta);
		return c;
	}

	// ---- Classification -------------------------------------------------------------------
	ClassificationContract LoadClassificationContract(const MetadataReader& meta, int64_t modelWidth, int64_t modelHeight)
	{
		ClassificationContract c;
		// Models exported before the contract existed carry no version: accepted, but a version
		// that is present must be the supported one.
		c.version = CheckVersion(meta, /*required=*/false);

		// The graph was exported without embedded names, so it emits class_id only:
		// names_ordered is the sole index -> label mapping available at runtime.
		c.classNames = ParseJsonStringArray(meta.Require("names_ordered"));

		// 'names' is the {"0":"label",...} object echoed to the client. Older exports carry it;
		// PyTorchTrainingTool exports only names_ordered, so it is rebuilt from it.
		c.namesJson = meta.Get("names");
		if (c.namesJson.empty()) {
			c.namesJson = BuildNamesJson(c.classNames);
		}

		// 'none' means the graph divides by 255 itself and expects raw [0,255] floats. Feeding
		// pre-normalized data would silently produce wrong predictions, so refuse to run.
		const std::string normalization = meta.Require("normalization");
		if (normalization != "none") {
			Reject("Classification model rejected: normalization='" + normalization +
				"' but the engine feeds raw [0,255] values.");
		}

		c.imgsz = meta.RequireInt("imgsz");
		if (c.imgsz != modelHeight || c.imgsz != modelWidth) {
			Reject("Classification model rejected: imgsz " + std::to_string(c.imgsz) + " but the input tensor is " +
				std::to_string(modelWidth) + "x" + std::to_string(modelHeight) + ".");
		}

		const std::string colorOrder = meta.Get("input_color_order");
		if (!colorOrder.empty() && colorOrder != "RGB") {
			Reject("Classification model rejected: input_color_order='" + colorOrder + "' (the engine feeds RGB).");
		}

		c.preprocessing = LoadPreprocessing(meta, modelWidth, modelHeight, /*interpolationRequired=*/false);
		if (!c.preprocessing.bgrToRgb) {
			Reject("Classification model rejected: preproc_color_conversion='none' but the engine always converts BGR->RGB.");
		}

		Log::Info("Classification contract: {} classes, imgsz {}, normalization {}.",
			c.classNames.size(), c.imgsz, normalization);
		return c;
	}

	// ---- Helpers --------------------------------------------------------------------------
	// Parser for the flat JSON string array written by json.dumps (ASCII with \uXXXX escapes).
	// Avoids a JSON dependency in the engine for a single well-known, machine-generated field.
	std::vector<std::string> ParseJsonStringArray(const std::string& json)
	{
		std::vector<std::string> out;
		size_t i = json.find('[');
		if (i == std::string::npos) {
			Reject("Model rejected: names_ordered metadata is not a JSON array.");
		}

		for (++i; i < json.size(); ++i) {
			if (json[i] == ']') break;
			if (json[i] != '"') continue;

			std::string value;
			for (++i; i < json.size() && json[i] != '"'; ++i) {
				const char c = json[i];
				if (c != '\\') {
					value.push_back(c);
					continue;
				}
				if (++i >= json.size()) break;
				switch (json[i]) {
				case 'n': value.push_back('\n'); break;
				case 't': value.push_back('\t'); break;
				case 'r': value.push_back('\r'); break;
				case 'b': value.push_back('\b'); break;
				case 'f': value.push_back('\f'); break;
				case 'u': {
					uint32_t cp = ParseHex4(json, i + 1);
					i += 4;
					// UTF-16 surrogate pair -> one code point
					if (cp >= 0xD800 && cp <= 0xDBFF && i + 6 < json.size() && json[i + 1] == '\\' && json[i + 2] == 'u') {
						const uint32_t low = ParseHex4(json, i + 3);
						if (low >= 0xDC00 && low <= 0xDFFF) {
							cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
							i += 6;
						}
					}
					AppendUtf8(value, cp);
					break;
				}
				default: value.push_back(json[i]); break;   // \" \\ \/
				}
			}
			out.push_back(std::move(value));
		}

		if (out.empty()) {
			Reject("Model rejected: names_ordered metadata contains no class names.");
		}
		return out;
	}

	// {"0":"label0","1":"label1",...}
	std::string BuildNamesJson(const std::vector<std::string>& names)
	{
		static const char* const hex = "0123456789abcdef";
		std::string json = "{";
		for (size_t i = 0; i < names.size(); ++i) {
			if (i > 0) json += ",";
			json += "\"" + std::to_string(i) + "\":\"";
			for (const char ch : names[i]) {
				const unsigned char u = static_cast<unsigned char>(ch);
				if (ch == '"' || ch == '\\') {
					json.push_back('\\');
					json.push_back(ch);
				}
				else if (u < 0x20) {
					json += "\\u00";
					json.push_back(hex[u >> 4]);
					json.push_back(hex[u & 0x0F]);
				}
				else {
					json.push_back(ch);   // UTF-8 bytes pass through untouched
				}
			}
			json += "\"";
		}
		json += "}";
		return json;
	}
}