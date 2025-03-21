// sherpa-onnx/jni/wave-writer.cc
//
// Copyright (c)  2024  Xiaomi Corporation
#include "sherpa-onnx/csrc/wave-writer.h"

#include "sherpa-onnx/jni/common.h"

SHERPA_ONNX_EXTERN_C
JNIEXPORT bool JNICALL Java_com_edgeai_chatappv2_WaveWriter_writeWaveToFile(
    JNIEnv *env, jclass /*obj*/, jstring filename, jfloatArray samples,
    jint sample_rate) {
  jfloat *p = env->GetFloatArrayElements(samples, nullptr);
  jsize n = env->GetArrayLength(samples);

  const char *p_filename = env->GetStringUTFChars(filename, nullptr);

  bool ok = sherpa_onnx::WriteWave(p_filename, sample_rate, p, n);

  env->ReleaseFloatArrayElements(samples, p, JNI_ABORT);
  env->ReleaseStringUTFChars(filename, p_filename);

  return ok;
}
