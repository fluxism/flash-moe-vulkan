/*
 * test_matvec.c — Test for the FMA-optimized 4-bit dequant matvec shader.
 * Generates random quantized weights, runs CPU reference and GPU shader,
 * compares results across multiple dimension configurations.
 */

#include "vk_compute.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

static float bf16_to_f32(uint16_t bf16) {
    uint32_t tmp = (uint32_t)bf16 << 16;
    float f;
    memcpy(&f, &tmp, 4);
    return f;
}

static uint16_t f32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    return (uint16_t)(bits >> 16);
}

static void cpu_dequant_matvec(const uint32_t* W, const uint16_t* scales,
                                const uint16_t* biases, const float* x,
                                float* out, uint32_t out_dim, uint32_t in_dim,
                                uint32_t group_size) {
    uint32_t packed_cols = in_dim / 8;
    uint32_t num_groups = in_dim / group_size;
    uint32_t packed_per_group = group_size / 8;
    for (uint32_t row = 0; row < out_dim; row++) {
        float acc = 0.0f;
        for (uint32_t g = 0; g < num_groups; g++) {
            float scale = bf16_to_f32(scales[row * num_groups + g]);
            float bias = bf16_to_f32(biases[row * num_groups + g]);
            for (uint32_t p = 0; p < packed_per_group; p++) {
                uint32_t packed = W[row * packed_cols + g * packed_per_group + p];
                for (uint32_t n = 0; n < 8; n++) {
                    uint32_t nibble = (packed >> (n * 4)) & 0xF;
                    acc += ((float)nibble * scale + bias) * x[g * group_size + p * 8 + n];
                }
            }
        }
        out[row] = acc;
    }
}

static uint16_t rand_bf16(void) {
    float f = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    return f32_to_bf16(f);
}

typedef struct {
    uint32_t out_dim;
    uint32_t in_dim;
    uint32_t group_size;
    const char* name;
} TestConfig;

static int run_test(VkCtx* ctx, VkPipe* pipe, const TestConfig* cfg) {
    uint32_t out_dim = cfg->out_dim;
    uint32_t in_dim = cfg->in_dim;
    uint32_t group_size = cfg->group_size;
    uint32_t packed_cols = in_dim / 8;
    uint32_t num_groups = in_dim / group_size;

    size_t w_bytes = (size_t)out_dim * packed_cols * sizeof(uint32_t);
    size_t sb_elems = (size_t)out_dim * num_groups;
    /* scales and biases are uint16 (bf16), but GPU buffer reads uint32 pairs.
       Round up to even number of uint16 elements so the uint32 buffer is aligned. */
    size_t sb_elems_aligned = (sb_elems + 1) & ~(size_t)1;
    size_t sb_bytes = sb_elems_aligned * sizeof(uint16_t);
    size_t x_bytes = in_dim * sizeof(float);
    size_t out_bytes = out_dim * sizeof(float);

    /* CPU-side arrays */
    uint32_t* cpu_w = malloc(w_bytes);
    uint16_t* cpu_scales = calloc(sb_elems_aligned, sizeof(uint16_t));
    uint16_t* cpu_biases = calloc(sb_elems_aligned, sizeof(uint16_t));
    float* cpu_x = malloc(x_bytes);
    float* cpu_out = malloc(out_bytes);

    /* Fill random data */
    for (size_t i = 0; i < out_dim * packed_cols; i++)
        cpu_w[i] = (uint32_t)rand();
    for (size_t i = 0; i < sb_elems; i++)
        cpu_scales[i] = rand_bf16();
    for (size_t i = 0; i < sb_elems; i++)
        cpu_biases[i] = rand_bf16();
    for (uint32_t i = 0; i < in_dim; i++)
        cpu_x[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

    /* CPU reference */
    cpu_dequant_matvec(cpu_w, cpu_scales, cpu_biases, cpu_x, cpu_out,
                       out_dim, in_dim, group_size);

    /* GPU buffers */
    VkBuf* buf_w = vk_buf_create(ctx, w_bytes);
    VkBuf* buf_s = vk_buf_create(ctx, sb_bytes);
    VkBuf* buf_b = vk_buf_create(ctx, sb_bytes);
    VkBuf* buf_x = vk_buf_create(ctx, x_bytes);
    VkBuf* buf_out = vk_buf_create(ctx, out_bytes);

    memcpy(vk_buf_map(buf_w), cpu_w, w_bytes);
    memcpy(vk_buf_map(buf_s), cpu_scales, sb_bytes);
    memcpy(vk_buf_map(buf_b), cpu_biases, sb_bytes);
    memcpy(vk_buf_map(buf_x), cpu_x, x_bytes);
    memset(vk_buf_map(buf_out), 0, out_bytes);

    /* Dispatch */
    uint32_t push[3] = { out_dim, in_dim, group_size };
    /* rows_per_tg = 256/subgroupSize. Conservative: assume wave64 (4 rows/WG).
       Extra workgroups are harmless — OOB rows early-return in the shader. */
    uint32_t num_wg = (out_dim + 3) / 4;

    VkCmd* cmd = vk_cmd_begin(ctx);
    VkBuf* bufs[] = { buf_w, buf_s, buf_b, buf_x, buf_out };
    vk_cmd_bind(cmd, pipe, bufs, NULL, NULL, 5, push, sizeof(push));
    vk_cmd_dispatch(cmd, num_wg, 1, 1);
    vk_cmd_submit(cmd);

    /* Compare */
    float* gpu_out = vk_buf_map(buf_out);
    int errors = 0;
    float max_rel_err = 0.0f;
    for (uint32_t i = 0; i < out_dim; i++) {
        float ref = cpu_out[i];
        float got = gpu_out[i];
        float denom = fabsf(ref) > 1e-6f ? fabsf(ref) : 1e-6f;
        float rel_err = fabsf(got - ref) / denom;
        if (rel_err > max_rel_err) max_rel_err = rel_err;
        if (rel_err > 1e-3f) {
            if (errors < 5) {
                fprintf(stderr, "  MISMATCH row %u: gpu=%.6f cpu=%.6f rel_err=%.6f\n",
                        i, got, ref, rel_err);
            }
            errors++;
        }
    }

    printf("  %-30s  %u/%u passed  max_rel_err=%.2e  %s\n",
           cfg->name, out_dim - errors, out_dim, max_rel_err,
           errors == 0 ? "OK" : "FAIL");

    /* Cleanup */
    vk_cmd_destroy(ctx, cmd);
    vk_buf_destroy(ctx, buf_w);
    vk_buf_destroy(ctx, buf_s);
    vk_buf_destroy(ctx, buf_b);
    vk_buf_destroy(ctx, buf_x);
    vk_buf_destroy(ctx, buf_out);
    free(cpu_w);
    free(cpu_scales);
    free(cpu_biases);
    free(cpu_x);
    free(cpu_out);

    return errors;
}

int main(void) {
    srand((unsigned)time(NULL));

    VkCtx* ctx = vk_create();
    if (!ctx) { fprintf(stderr, "Failed to create Vulkan context\n"); return 1; }

    VkPipe* pipe = vk_pipe_create(ctx, "shaders/dequant_matvec_4bit.spv",
                                   3 * sizeof(uint32_t), 5);
    if (!pipe) {
        fprintf(stderr, "Failed to create pipeline (compile shader first)\n");
        vk_destroy(ctx);
        return 1;
    }

    TestConfig configs[] = {
        { 128,  4096, 64, "small (128x4096)" },
        { 1024, 4096, 64, "gate/up (1024x4096)" },
        { 4096, 1024, 64, "down (4096x1024)" },
    };
    int num_configs = sizeof(configs) / sizeof(configs[0]);

    printf("dequant_matvec_4bit tests:\n");
    int total_errors = 0;
    for (int i = 0; i < num_configs; i++) {
        vk_descriptor_pool_reset(ctx);
        total_errors += run_test(ctx, pipe, &configs[i]);
    }

    vk_pipe_destroy(ctx, pipe);
    vk_destroy(ctx);

    printf("\n%s\n", total_errors == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return total_errors > 0 ? 1 : 0;
}
