// sherpa-onnx/csrc/provider.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_PROVIDER_H_
#define SHERPA_ONNX_CSRC_PROVIDER_H_

#include <string>

#include "sherpa-onnx/csrc/provider-config.h"
namespace sherpa_onnx {

// Please refer to
// https://github.com/microsoft/onnxruntime/blob/main/java/src/main/java/ai/onnxruntime/OrtProvider.java
// for a list of available providers
enum class Provider {
  kCPU = 0,       // CPUExecutionProvider
  kCUDA = 1,      // CUDAExecutionProvider
  kCoreML = 2,    // CoreMLExecutionProvider
  kXnnpack = 3,   // XnnpackExecutionProvider
  kNNAPI = 4,     // NnapiExecutionProvider
  kTRT = 5,       // TensorRTExecutionProvider
  kDirectML = 6,  // DmlExecutionProvider
  kQNN = 7,       // QNNExecutionProvider
};

/**
 * Convert a string to an enum.
 *
 * @param s We will convert it to lowercase before comparing.
 * @return Return an instance of Provider.
 */
Provider StringToProvider(std::string s);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_PROVIDER_H_
