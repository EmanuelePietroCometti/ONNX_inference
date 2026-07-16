#pragma once
#include <onnxruntime_cxx_api.h>
#include <string>

bool ConfigureOrtSessionOptions(Ort::SessionOptions& so, const std::string& tag);

#pragma