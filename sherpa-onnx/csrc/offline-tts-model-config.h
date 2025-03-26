// sherpa-onnx/csrc/offline-tts-model-config.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_TTS_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_TTS_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/offline-tts-kokoro-model-config.h"
#include "sherpa-onnx/csrc/offline-tts-matcha-model-config.h"
#include "sherpa-onnx/csrc/offline-tts-vits-model-config.h"
#include "sherpa-onnx/csrc/parse-options.h"
#include "sherpa-onnx/csrc/provider-config.h"
namespace sherpa_onnx {

struct OfflineTtsModelConfig {
  OfflineTtsVitsModelConfig vits;
  OfflineTtsMatchaModelConfig matcha;
  OfflineTtsKokoroModelConfig kokoro;
  ProviderConfig provider_config;

  int32_t num_threads = 1;
  bool debug = false;
  std::string provider = "cpu";

  OfflineTtsModelConfig() = default;

  OfflineTtsModelConfig(const OfflineTtsVitsModelConfig &vits,
                        const OfflineTtsMatchaModelConfig &matcha,
                        const OfflineTtsKokoroModelConfig &kokoro,
                        int32_t num_threads, bool debug,
                        const std::string &provider,
                        const ProviderConfig &provider_config)
      : vits(vits),
        matcha(matcha),
        kokoro(kokoro),
        num_threads(num_threads),
        debug(debug),
        provider(provider),
        provider_config(provider_config) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_TTS_MODEL_CONFIG_H_
