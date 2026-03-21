#ifndef LINEAR_ATTN_H
#define LINEAR_ATTN_H

#include <stdint.h>

#define LINEAR_NUM_V_HEADS  64
#define LINEAR_NUM_K_HEADS  16
#define LINEAR_KEY_DIM      128
#define LINEAR_VALUE_DIM    128
#define LINEAR_TOTAL_KEY    (LINEAR_NUM_K_HEADS * LINEAR_KEY_DIM)   // 2048
#define LINEAR_TOTAL_VALUE  (LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM) // 8192
#define LINEAR_CONV_DIM     (LINEAR_TOTAL_KEY * 2 + LINEAR_TOTAL_VALUE) // 12288
#define CONV_KERNEL_SIZE    4

typedef struct {
    // Delta-net state: [num_v_heads, value_dim, key_dim]
    float* ssm_state;   // 64 * 128 * 128 = 1M floats
    // Conv1d state: [(kernel_size-1), conv_dim]
    float* conv_state;  // 3 * 12288 floats
} LinearAttnState;

LinearAttnState* linear_attn_state_create(void);
void linear_attn_state_destroy(LinearAttnState* s);

// Run one token through the linear attention pipeline.
// Takes pre-computed projections from GPU matvec.
// qkv_proj: [LINEAR_CONV_DIM=12288] from in_proj_qkv matvec
// z_proj: [LINEAR_TOTAL_VALUE=8192] from in_proj_z matvec
// beta_proj: [LINEAR_NUM_V_HEADS=64] from in_proj_b matvec
// alpha_proj: [LINEAR_NUM_V_HEADS=64] from in_proj_a matvec
// conv_weights: bf16 [LINEAR_CONV_DIM * CONV_KERNEL_SIZE] -- conv1d weights
// A_log: float [LINEAR_NUM_V_HEADS=64] -- decay base (log space)
// dt_bias: bf16 [LINEAR_NUM_V_HEADS=64] -- dt bias
// gated_norm_w: bf16 [LINEAR_VALUE_DIM=128] -- gated norm weights (shared across heads)
// output: [LINEAR_TOTAL_VALUE=8192] -- attention output before o_proj
void linear_attn_forward(
    LinearAttnState* state,
    const float* qkv_proj,
    const float* z_proj,
    const float* beta_proj,
    const float* alpha_proj,
    const uint16_t* conv_weights,
    const float* A_log,
    const uint16_t* dt_bias,
    const uint16_t* gated_norm_w,
    float* output
);

#endif
