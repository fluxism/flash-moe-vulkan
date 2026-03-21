/*
 * test_vk.c — Smoke test for the Vulkan compute abstraction layer.
 * Runs a simple vector add shader and verifies correctness.
 */

#include "vk_compute.h"
#include <stdio.h>
#include <math.h>

int main(void) {
    VkCtx* ctx = vk_create();
    if (!ctx) { fprintf(stderr, "Failed to create Vulkan context\n"); return 1; }

    uint32_t n = 1024;
    VkBuf* a = vk_buf_create(ctx, n * sizeof(float));
    VkBuf* b = vk_buf_create(ctx, n * sizeof(float));
    VkBuf* c = vk_buf_create(ctx, n * sizeof(float));

    float* ap = vk_buf_map(a);
    float* bp = vk_buf_map(b);
    for (uint32_t i = 0; i < n; i++) { ap[i] = (float)i; bp[i] = (float)(i * 2); }

    VkPipe* pipe = vk_pipe_create(ctx, "shaders/test_add.spv", sizeof(uint32_t), 3);
    if (!pipe) { fprintf(stderr, "Failed to create pipeline\n"); vk_destroy(ctx); return 1; }

    VkCmd* cmd = vk_cmd_begin(ctx);
    VkBuf* bufs[] = {a, b, c};
    size_t offsets[] = {0, 0, 0};
    size_t ranges[] = {n * sizeof(float), n * sizeof(float), n * sizeof(float)};
    vk_cmd_bind(cmd, pipe, bufs, offsets, ranges, 3, &n, sizeof(n));
    vk_cmd_dispatch(cmd, (n + 255) / 256, 1, 1);
    vk_cmd_submit(cmd);

    float* cp = vk_buf_map(c);
    int errors = 0;
    for (uint32_t i = 0; i < n; i++) {
        float expected = (float)i + (float)(i * 2);
        if (fabsf(cp[i] - expected) > 1e-5f) {
            if (errors < 5) {
                fprintf(stderr, "  MISMATCH at %u: got %f, expected %f\n", i, cp[i], expected);
            }
            errors++;
        }
    }
    printf("test_add: %d/%u passed\n", n - errors, n);

    vk_cmd_destroy(ctx, cmd);
    vk_pipe_destroy(ctx, pipe);
    vk_buf_destroy(ctx, a);
    vk_buf_destroy(ctx, b);
    vk_buf_destroy(ctx, c);
    vk_destroy(ctx);
    return errors > 0 ? 1 : 0;
}
