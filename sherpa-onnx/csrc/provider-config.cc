// sherpa-onnx/csrc/provider-config.cc
//
// Copyright (c)  2024  Uniphore (Author: Manickavela)

#include "sherpa-onnx/csrc/provider-config.h"

#include <sstream>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void CudaConfig::Register(ParseOptions *po) {
  po->Register("cuda-cudnn-conv-algo-search", &cudnn_conv_algo_search,
               "CuDNN convolution algrorithm search");
}

bool CudaConfig::Validate() const {
  if (cudnn_conv_algo_search < 1 || cudnn_conv_algo_search > 3) {
    SHERPA_ONNX_LOGE(
        "cudnn_conv_algo_search: '%d' is not a valid option."
        "Options : [1,3]. Check OnnxRT docs",
        cudnn_conv_algo_search);
    return false;
  }
  return true;
}

std::string CudaConfig::ToString() const {
  std::ostringstream os;

  os << "CudaConfig(";
  os << "cudnn_conv_algo_search=" << cudnn_conv_algo_search << ")";

  return os.str();
}

void TensorrtConfig::Register(ParseOptions *po) {
  po->Register("trt-max-workspace-size", &trt_max_workspace_size,
               "Set TensorRT EP GPU memory usage limit.");
  po->Register("trt-max-partition-iterations", &trt_max_partition_iterations,
               "Limit partitioning iterations for model conversion.");
  po->Register("trt-min-subgraph-size", &trt_min_subgraph_size,
               "Set minimum size for subgraphs in partitioning.");
  po->Register("trt-fp16-enable", &trt_fp16_enable,
               "Enable FP16 precision for faster performance.");
  po->Register("trt-detailed-build-log", &trt_detailed_build_log,
               "Enable detailed logging of build steps.");
  po->Register("trt-engine-cache-enable", &trt_engine_cache_enable,
               "Enable caching of TensorRT engines.");
  po->Register("trt-timing-cache-enable", &trt_timing_cache_enable,
               "Enable use of timing cache to speed up builds.");
  po->Register("trt-engine-cache-path", &trt_engine_cache_path,
               "Set path to store cached TensorRT engines.");
  po->Register("trt-timing-cache-path", &trt_timing_cache_path,
               "Set path for storing timing cache.");
  po->Register("trt-dump-subgraphs", &trt_dump_subgraphs,
               "Dump optimized subgraphs for debugging.");
}

bool TensorrtConfig::Validate() const {
  if (trt_max_workspace_size < 0) {
    std::ostringstream os;
    os << "trt_max_workspace_size: " << trt_max_workspace_size
       << " is not valid.";
    SHERPA_ONNX_LOGE("%s", os.str().c_str());
    return false;
  }
  if (trt_max_partition_iterations < 0) {
    SHERPA_ONNX_LOGE("trt_max_partition_iterations: %d is not valid.",
                     trt_max_partition_iterations);
    return false;
  }
  if (trt_min_subgraph_size < 0) {
    SHERPA_ONNX_LOGE("trt_min_subgraph_size: %d is not valid.",
                     trt_min_subgraph_size);
    return false;
  }

  return true;
}

std::string TensorrtConfig::ToString() const {
  std::ostringstream os;

  os << "TensorrtConfig(";
  os << "trt_max_workspace_size=" << trt_max_workspace_size << ", ";
  os << "trt_max_partition_iterations=" << trt_max_partition_iterations << ", ";
  os << "trt_min_subgraph_size=" << trt_min_subgraph_size << ", ";
  os << "trt_fp16_enable=\"" << (trt_fp16_enable ? "True" : "False") << "\", ";
  os << "trt_detailed_build_log=\""
     << (trt_detailed_build_log ? "True" : "False") << "\", ";
  os << "trt_engine_cache_enable=\""
     << (trt_engine_cache_enable ? "True" : "False") << "\", ";
  os << "trt_engine_cache_path=\"" << trt_engine_cache_path.c_str() << "\", ";
  os << "trt_timing_cache_enable=\""
     << (trt_timing_cache_enable ? "True" : "False") << "\", ";
  os << "trt_timing_cache_path=\"" << trt_timing_cache_path.c_str() << "\",";
  os << "trt_dump_subgraphs=\"" << (trt_dump_subgraphs ? "True" : "False")
     << "\" )";
  return os.str();
}

void QnnConfig::Register(ParseOptions *po) {
  po->Register("qnn-json-config", &json_config,
               "JSON string with all QNN provider options");
}

bool QnnConfig::Validate() const {

  // Validate JSON configuration if provided
  if (!json_config.empty()) {
    try {
      // Validate JSON format - Check for basic syntax
      if (json_config.front() != '{' || json_config.back() != '}') {
        SHERPA_ONNX_LOGE("Invalid JSON format: must start with '{' and end with '}'");
        return false;
      }

      // Parse and validate known QNN options
      std::istringstream json_stream(json_config);
      std::string line;
      
      // Map of supported QNN options and their valid values (empty means any value is allowed)
      const std::unordered_map<std::string, std::vector<std::string>> valid_options = {
        {"backend_path", {}},
        {"profiling_level", {"off", "basic", "detailed"}},
        {"profiling_file_path", {}},
        {"rpc_control_latency", {}},
        {"vtcm_mb", {}},
        {"htp_performance_mode", {
          "burst", "balanced", "default", "high_performance", 
          "high_power_saver", "low_balanced", "extreme_power_saver", 
          "low_power_saver", "power_saver", "sustained_high_performance"
        }},
        {"qnn_saver_path", {}},
        {"qnn_context_priority", {"low", "normal", "normal_high", "high"}},
        {"htp_graph_finalization_optimization_mode", {"0", "1", "2", "3"}},
        {"soc_model", {}},
        {"htp_arch", {"0", "68", "69", "73", "75", "79"}},
        {"device_id", {}},
        {"enable_htp_fp16_precision", {"0", "1"}},
        {"enable_htp_weight_sharing", {"0", "1"}},
        {"offload_graph_io_quantization", {"0", "1"}},
        {"enable_htp_spill_fill_buffer", {"0", "1"}},
        {"enable_htp_shared_memory_allocator", {"0", "1"}},
        {"dump_json_qnn_graph", {"0", "1"}},
        {"json_qnn_graph_dir", {}}
      };

      // Simple JSON parsing with validation
      std::string content = json_config.substr(1, json_config.size() - 2);  // Remove {}
      size_t pos = 0;
      bool in_string = false;
      std::string key, value;
      std::string current_token;
      enum State { KEY, COLON, VALUE, COMMA };
      State state = KEY;

      for (size_t i = 0; i < content.size(); ++i) {
        char c = content[i];
        
        if (c == '"') {
          in_string = !in_string;
          continue;
        }
        
        if (!in_string && (c == ' ' || c == '\t' || c == '\n' || c == '\r')) {
          continue;  // Skip whitespace outside strings
        }
        
        if (in_string) {
          current_token += c;
        } else {
          if (c == ':') {
            if (state != KEY) {
              SHERPA_ONNX_LOGE("Unexpected ':' in JSON");
              return false;
            }
            key = current_token;
            current_token.clear();
            state = VALUE;
            continue;
          } else if (c == ',') {
            if (state != VALUE) {
              SHERPA_ONNX_LOGE("Unexpected ',' in JSON");
              return false;
            }
            value = current_token;
            current_token.clear();
            
            // Validate key and value
            if (valid_options.find(key) == valid_options.end()) {
              SHERPA_ONNX_LOGE("Unknown QNN option: %s", key.c_str());
              return false;
            }
            
            const auto& valid_values = valid_options.at(key);
            if (!valid_values.empty() && 
                std::find(valid_values.begin(), valid_values.end(), value) == valid_values.end()) {
              SHERPA_ONNX_LOGE("Invalid value '%s' for QNN option '%s'. Valid values: ", 
                              value.c_str(), key.c_str());
              for (const auto& v : valid_values) {
                SHERPA_ONNX_LOGE(" - %s", v.c_str());
              }
              return false;
            }
            
            state = KEY;
            continue;
          } else {
            current_token += c;
          }
        }
      }
      
      // Handle last key-value pair
      if (!current_token.empty()) {
        value = current_token;
        
        // Validate key and value
        if (valid_options.find(key) == valid_options.end()) {
          SHERPA_ONNX_LOGE("Unknown QNN option: %s", key.c_str());
          return false;
        }
        
        const auto& valid_values = valid_options.at(key);
        if (!valid_values.empty() && 
            std::find(valid_values.begin(), valid_values.end(), value) == valid_values.end()) {
          SHERPA_ONNX_LOGE("Invalid value '%s' for QNN option '%s'", value.c_str(), key.c_str());
          return false;
        }
      }
    } catch (const std::exception& e) {
      SHERPA_ONNX_LOGE("Error validating JSON config: %s", e.what());
      return false;
    }
  }
  
  return true;
}

std::string QnnConfig::ToString() const {
  std::ostringstream os;

  os << "QnnConfig(";
  os << "json_config=\"" << (json_config.empty() ? "" : "(JSON config provided)") << "\")";
  return os.str();
}

void ProviderConfig::Register(ParseOptions *po) {
  cuda_config.Register(po);
  trt_config.Register(po);
  qnn_config.Register(po);

  po->Register("device", &device, "GPU device index for CUDA and Trt EP");
  po->Register("provider", &provider,
               "Specify a provider to use: cpu, cuda, coreml, qnn");
}

bool ProviderConfig::Validate() const {
  if (device < 0) {
    SHERPA_ONNX_LOGE("device: '%d' is invalid.", device);
    return false;
  }

  if (provider == "cuda" && !cuda_config.Validate()) {
    return false;
  }

  if (provider == "trt" && !trt_config.Validate()) {
    return false;
  }
  
  if (provider == "qnn" && !qnn_config.Validate()) {
    return false;
  }

  return true;
}

std::string ProviderConfig::ToString() const {
  std::ostringstream os;

  os << "ProviderConfig(";
  os << "device=" << device << ", ";
  os << "provider=\"" << provider << "\", ";
  os << "cuda_config=" << cuda_config.ToString() << ", ";
  os << "trt_config=" << trt_config.ToString() << ", ";
  os << "qnn_config=" << qnn_config.ToString() << ")";
  return os.str();
}

}  // namespace sherpa_onnx
