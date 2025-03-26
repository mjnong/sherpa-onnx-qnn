package com.edgeai.chatappv2

// QNN configuration enums for type-safe configuration
object QnnOptions {
    // Performance mode options
    enum class PerformanceMode(val value: String) {
        BURST("burst"),
        BALANCED("balanced"),
        DEFAULT("default"),
        HIGH_PERFORMANCE("high_performance"),
        HIGH_POWER_SAVER("high_power_saver"),
        LOW_BALANCED("low_balanced"),
        EXTREME_POWER_SAVER("extreme_power_saver"),
        LOW_POWER_SAVER("low_power_saver"),
        POWER_SAVER("power_saver"),
        SUSTAINED_HIGH_PERFORMANCE("sustained_high_performance")
    }

    // Profiling level options
    enum class ProfilingLevel(val value: String) {
        OFF("off"),
        BASIC("basic"),
        DETAILED("detailed")
    }

    // Context priority options
    enum class ContextPriority(val value: String) {
        LOW("low"),
        NORMAL("normal"),
        NORMAL_HIGH("normal_high"),
        HIGH("high")
    }

    // Graph finalization optimization mode options
    enum class GraphFinalizationMode(val value: String) {
        DEFAULT("0"),
        FASTER_PREP("1"),
        OPTIMAL_GRAPH("2"),
        MOST_OPTIMAL("3")
    }

    // HTP architecture options
    enum class HtpArch(val value: String) {
        DEFAULT("0"),
        ARCH_68("68"),
        ARCH_69("69"),
        ARCH_73("73"),
        ARCH_75("75"),
        ARCH_79("79")
    }

    // Option keys (constants to avoid typos)
    object Keys {
        const val BACKEND_PATH = "backend_path"
        const val PROFILING_LEVEL = "profiling_level"
        const val PROFILING_FILE_PATH = "profiling_file_path"
        const val RPC_CONTROL_LATENCY = "rpc_control_latency"
        const val VTCM_MB = "vtcm_mb"
        const val HTP_PERFORMANCE_MODE = "htp_performance_mode"
        const val QNN_SAVER_PATH = "qnn_saver_path"
        const val QNN_CONTEXT_PRIORITY = "qnn_context_priority"
        const val HTP_GRAPH_FINALIZATION_MODE = "htp_graph_finalization_optimization_mode"
        const val SOC_MODEL = "soc_model"
        const val HTP_ARCH = "htp_arch"
        const val DEVICE_ID = "device_id"
        const val ENABLE_HTP_FP16_PRECISION = "enable_htp_fp16_precision"
        const val ENABLE_HTP_WEIGHT_SHARING = "enable_htp_weight_sharing"
        const val OFFLOAD_GRAPH_IO_QUANTIZATION = "offload_graph_io_quantization"
        const val ENABLE_HTP_SPILL_FILL_BUFFER = "enable_htp_spill_fill_buffer"
        const val ENABLE_HTP_SHARED_MEMORY_ALLOCATOR = "enable_htp_shared_memory_allocator"
        const val DUMP_JSON_QNN_GRAPH = "dump_json_qnn_graph"
        const val JSON_QNN_GRAPH_DIR = "json_qnn_graph_dir"
        
        // Simplified key names (map to original keys)
        const val NPU_ENABLE = "npu_enable"
        const val FP16_ENABLE = "fp16_enable"
        const val HTP_ENABLE = "htp_enable"
    }
}

// Helper function to create QNN JSON configuration from a map
fun createQnnJsonConfig(
    options: Map<String, String>
): String {
    val jsonBuilder = StringBuilder("{")
    options.entries.forEachIndexed { index, entry ->
        jsonBuilder.append("\"${entry.key}\":\"${entry.value}\"")
        if (index < options.size - 1) {
            jsonBuilder.append(",")
        }
    }
    jsonBuilder.append("}")
    return jsonBuilder.toString()
}

// Builder class for QNN configuration
class QnnConfigBuilder {
    private val options = mutableMapOf<String, String>()
    
    // Basic options
    fun useNpu(enable: Boolean = true): QnnConfigBuilder {
        options[QnnOptions.Keys.NPU_ENABLE] = if (enable) "1" else "0"
        return this
    }
    
    fun useFp16(enable: Boolean = true): QnnConfigBuilder {
        options[QnnOptions.Keys.FP16_ENABLE] = if (enable) "1" else "0"
        return this
    }
    
    fun useHtp(enable: Boolean = true): QnnConfigBuilder {
        options[QnnOptions.Keys.HTP_ENABLE] = if (enable) "1" else "0"
        return this
    }
    
    fun backendPath(path: String): QnnConfigBuilder {
        options[QnnOptions.Keys.BACKEND_PATH] = path
        return this
    }
    
    // Advanced options
    fun performanceMode(mode: QnnOptions.PerformanceMode): QnnConfigBuilder {
        options[QnnOptions.Keys.HTP_PERFORMANCE_MODE] = mode.value
        return this
    }
    
    fun profilingLevel(level: QnnOptions.ProfilingLevel): QnnConfigBuilder {
        options[QnnOptions.Keys.PROFILING_LEVEL] = level.value
        return this
    }
    
    fun profilingFilePath(path: String): QnnConfigBuilder {
        options[QnnOptions.Keys.PROFILING_FILE_PATH] = path
        return this
    }
    
    fun vtcmMb(size: Int): QnnConfigBuilder {
        options[QnnOptions.Keys.VTCM_MB] = size.toString()
        return this
    }
    
    fun saverPath(path: String): QnnConfigBuilder {
        options[QnnOptions.Keys.QNN_SAVER_PATH] = path
        return this
    }
    
    fun contextPriority(priority: QnnOptions.ContextPriority): QnnConfigBuilder {
        options[QnnOptions.Keys.QNN_CONTEXT_PRIORITY] = priority.value
        return this
    }
    
    fun graphFinalizationMode(mode: QnnOptions.GraphFinalizationMode): QnnConfigBuilder {
        options[QnnOptions.Keys.HTP_GRAPH_FINALIZATION_MODE] = mode.value
        return this
    }
    
    fun socModel(model: String): QnnConfigBuilder {
        options[QnnOptions.Keys.SOC_MODEL] = model
        return this
    }
    
    fun htpArch(arch: QnnOptions.HtpArch): QnnConfigBuilder {
        options[QnnOptions.Keys.HTP_ARCH] = arch.value
        return this
    }
    
    fun deviceId(id: Int): QnnConfigBuilder {
        options[QnnOptions.Keys.DEVICE_ID] = id.toString()
        return this
    }
    
    fun enableHtpFp16Precision(enable: Boolean = true): QnnConfigBuilder {
        options[QnnOptions.Keys.ENABLE_HTP_FP16_PRECISION] = if (enable) "1" else "0"
        return this
    }
    
    fun enableHtpWeightSharing(enable: Boolean = true): QnnConfigBuilder {
        options[QnnOptions.Keys.ENABLE_HTP_WEIGHT_SHARING] = if (enable) "1" else "0"
        return this
    }
    
    fun offloadGraphIoQuantization(enable: Boolean = true): QnnConfigBuilder {
        options[QnnOptions.Keys.OFFLOAD_GRAPH_IO_QUANTIZATION] = if (enable) "1" else "0"
        return this
    }
    
    fun enableHtpSpillFillBuffer(enable: Boolean = true): QnnConfigBuilder {
        options[QnnOptions.Keys.ENABLE_HTP_SPILL_FILL_BUFFER] = if (enable) "1" else "0"
        return this
    }
    
    fun enableHtpSharedMemoryAllocator(enable: Boolean = true): QnnConfigBuilder {
        options[QnnOptions.Keys.ENABLE_HTP_SHARED_MEMORY_ALLOCATOR] = if (enable) "1" else "0"
        return this
    }
    
    fun dumpJsonQnnGraph(enable: Boolean = true): QnnConfigBuilder {
        options[QnnOptions.Keys.DUMP_JSON_QNN_GRAPH] = if (enable) "1" else "0"
        return this
    }
    
    fun jsonQnnGraphDir(path: String): QnnConfigBuilder {
        options[QnnOptions.Keys.JSON_QNN_GRAPH_DIR] = path
        return this
    }
    
    // Custom option for extensibility
    fun custom(key: String, value: String): QnnConfigBuilder {
        options[key] = value
        return this
    }
    
    // Build the final JSON config
    fun build(): String {
        return createQnnJsonConfig(options)
    }
}