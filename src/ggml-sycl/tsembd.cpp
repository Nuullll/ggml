#include "tsembd.h"

static void timestep_embedding_f32(const float * timesteps, float * dst, const int nb1, const int dim, const int max_period, const sycl::nd_item<3> &item) {
    int i = item.get_group(1);
    int j = item.get_local_id(2) + item.get_group(2) * item.get_local_range(2);
    float * embed_data = (float *)((char *)dst + i*nb1);

    if (dim % 2 != 0 && j == ((dim + 1) / 2)) {
        embed_data[dim] = 0.f;
    }

    int half = dim / 2;
    if (j >= half) {
        return;
    }

    float timestep = timesteps[i];
    float freq = (float)sycl::exp(-sycl::log((float)max_period) * j / half);
    float arg = timestep * freq;
    embed_data[j] = sycl::cos(arg);
    embed_data[j + half] = sycl::sin(arg);
}

static void timestep_embedding_f32_sycl(const float * x, float * dst, const int ne00, const int nb1,
                                        const int dim, const int max_period, const queue_ptr &stream) {
    int half_ceil = (dim + 1) / 2;
    int num_blocks = (half_ceil + SYCL_TIMESTEP_EMBEDDING_BLOCK_SIZE - 1) / SYCL_TIMESTEP_EMBEDDING_BLOCK_SIZE;
    sycl::range<3> gridDim(1, ne00, num_blocks);
    sycl::range<3> blockDim(1, 1, SYCL_TIMESTEP_EMBEDDING_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<3>(gridDim * blockDim, blockDim),
        [=](sycl::nd_item<3> item) {
            timestep_embedding_f32(x, dst, nb1, dim, max_period, item);
        }
    );
}

void ggml_sycl_op_timestep_embedding(ggml_backend_sycl_context & ctx, const ggml_tensor *src0,
                                  const ggml_tensor *src1, ggml_tensor *dst,
                                  const float *src0_d, const float *src1_d,
                                  float *dst_d, const queue_ptr &stream) {
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int dim = dst->op_params[0];
    const int max_period = dst->op_params[1];

    timestep_embedding_f32_sycl(src0_d, dst_d, src0->ne[0], dst->nb[1], dim, max_period, stream);
}