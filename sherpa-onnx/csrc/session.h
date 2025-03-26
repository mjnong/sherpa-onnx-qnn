// sherpa-onnx/csrc/session.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_SESSION_H_
#define SHERPA_ONNX_CSRC_SESSION_H_

#include <string>
#include <type_traits>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-lm-config.h"
#include "sherpa-onnx/csrc/online-lm-config.h"
#include "sherpa-onnx/csrc/online-model-config.h"

// Forward declaration
namespace sherpa_onnx {
struct OfflineTtsModelConfig;
}

namespace sherpa_onnx {

Ort::SessionOptions GetSessionOptionsImpl(
    int32_t num_threads, const std::string &provider_str,
    const ProviderConfig *provider_config = nullptr);

Ort::SessionOptions GetSessionOptions(const OfflineLMConfig &config);
Ort::SessionOptions GetSessionOptions(const OnlineLMConfig &config);

Ort::SessionOptions GetSessionOptions(const OnlineModelConfig &config);

Ort::SessionOptions GetSessionOptions(const OnlineModelConfig &config,
                                      const std::string &model_type);

Ort::SessionOptions GetSessionOptions(int32_t num_threads,
                                      const std::string &provider_str);

// Explicit specialization for OfflineTtsModelConfig
Ort::SessionOptions GetSessionOptions(const OfflineTtsModelConfig &config);

// SFINAE helper to detect if ProviderConfig and IsEmpty() exist
template <typename T>
class HasProviderConfig {
  private:
    // These two function templates are used for detection
    template <typename C> static auto test(int) -> decltype(std::declval<C>().provider_config.IsEmpty(), std::true_type());
    template <typename C> static std::false_type test(...);
  
  public:
    // This will be either std::true_type or std::false_type
    using type = decltype(test<T>(0));
    static constexpr bool value = type::value;
};

// Template specialization for configs with provider_config field and IsEmpty method
template <typename T>
typename std::enable_if<HasProviderConfig<T>::value, Ort::SessionOptions>::type
GetSessionOptions(const T &config) {
  SHERPA_ONNX_LOGE("GetSessionOptions (with provider config): %s", config.ToString().c_str());
  if (config.provider_config.IsEmpty()) {
    return GetSessionOptionsImpl(config.num_threads, config.provider);
  }
  return GetSessionOptionsImpl(config.num_threads, config.provider, &config.provider_config);
}

// Template specialization for configs without provider_config field
template <typename T>
typename std::enable_if<!HasProviderConfig<T>::value, Ort::SessionOptions>::type
GetSessionOptions(const T &config) {
  SHERPA_ONNX_LOGE("GetSessionOptions (without provider config): %s", config.ToString().c_str());
  return GetSessionOptionsImpl(config.num_threads, config.provider);
}

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_SESSION_H_
