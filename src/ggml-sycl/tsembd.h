#pragma once

#include <sycl/sycl.hpp>

#include "common.hpp"

#include "ggml.h"
#include "ggml-sycl.h"

#define SYCL_TIMESTEP_EMBEDDING_BLOCK_SIZE 256

void ggml_sycl_op_timestep_embedding(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                  const ggml_tensor *src1, ggml_tensor *dst,
                                  const float *src0_d, const float *src1_d,
                                  float *dst_d, const queue_ptr &stream);
