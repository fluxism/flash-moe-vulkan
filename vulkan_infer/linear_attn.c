#include "linear_attn.h"

#include <cblas.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int g_debug;

// ============================================================================
// Helpers
// ============================================================================

static inline float bf16_to_f32(uint16_t bf16) {
    uint32_t tmp = (uint32_t)bf16 << 16;
    float f;
    memcpy(&f, &tmp, sizeof(f));
    return f;
}

static inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

// ============================================================================
// State management
// ============================================================================

LinearAttnState* linear_attn_state_create(void) {
    LinearAttnState* s = calloc(1, sizeof(LinearAttnState));
    s->ssm_state = calloc((size_t)LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM * LINEAR_KEY_DIM, sizeof(float));
    s->conv_state = calloc((size_t)(CONV_KERNEL_SIZE - 1) * LINEAR_CONV_DIM, sizeof(float));
    return s;
}

void linear_attn_state_destroy(LinearAttnState* s) {
    if (!s) return;
    free(s->ssm_state);
    free(s->conv_state);
    free(s);
}

// ============================================================================
// Conv1d step: causal depthwise conv with SiLU activation
// ============================================================================
// conv_state: [(kernel_size-1) * conv_dim] row-major (3 rows of conv_dim)
// input: [conv_dim] current token projection
// weights: bf16 [conv_dim * kernel_size] — weight[c * 4 + k]
// output: [conv_dim] convolution result with SiLU applied
//
// After computing output, shifts conv_state left and appends input.

static void conv1d_step(
    float* conv_state,
    const float* input,
    const uint16_t* weights,
    float* output,
    int conv_dim
) {
    for (int c = 0; c < conv_dim; c++) {
        int w_base = c * CONV_KERNEL_SIZE;
        float acc = 0.0f;

        // 3 history slots (k=0,1,2)
        acc += conv_state[0 * conv_dim + c] * bf16_to_f32(weights[w_base + 0]);
        acc += conv_state[1 * conv_dim + c] * bf16_to_f32(weights[w_base + 1]);
        acc += conv_state[2 * conv_dim + c] * bf16_to_f32(weights[w_base + 2]);

        // New input (k=3)
        acc += input[c] * bf16_to_f32(weights[w_base + 3]);

        // SiLU activation
        output[c] = silu(acc);
    }

    // Shift history: slots 1,2 -> 0,1, then append input at slot 2
    memmove(conv_state, conv_state + conv_dim,
            (CONV_KERNEL_SIZE - 2) * conv_dim * sizeof(float));
    memcpy(conv_state + (CONV_KERNEL_SIZE - 2) * conv_dim, input,
           conv_dim * sizeof(float));
}

// ============================================================================
// RMS norm (bare, no learned weights)
// ============================================================================

static void rms_norm_bare(float* x, int dim, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) sum_sq += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(sum_sq / dim + eps);
    for (int i = 0; i < dim; i++) x[i] *= inv_rms;
}

// ============================================================================
// RMS norm gated: out = rms_norm(x) * silu(z) * weight
// ============================================================================

static void rms_norm_gated(
    const float* x,
    const float* z,
    const uint16_t* w_bf16,
    float* out,
    int dim,
    float eps
) {
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) sum_sq += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(sum_sq / dim + eps);
    for (int i = 0; i < dim; i++) {
        float w = bf16_to_f32(w_bf16[i]);
        float silu_z = silu(z[i]);
        out[i] = x[i] * inv_rms * w * silu_z;
    }
}

// ============================================================================
// GatedDeltaNet linear attention forward (single token, incremental)
// ============================================================================

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
) {
    const float RMS_EPS = 1e-6f;
    const int k_heads_per_v = LINEAR_NUM_V_HEADS / LINEAR_NUM_K_HEADS; // 4

    // ---- Conv1d step ----
    float conv_out[LINEAR_CONV_DIM];
    conv1d_step(state->conv_state, qkv_proj, conv_weights, conv_out, LINEAR_CONV_DIM);

    // ---- Split into q, k, v ----
    float* lin_q = conv_out;                             // [2048]
    float* lin_k = conv_out + LINEAR_TOTAL_KEY;          // [2048]
    float* lin_v = conv_out + 2 * LINEAR_TOTAL_KEY;      // [8192]

    if (g_debug) {
        // Count non-zero elements and check where divergence starts
        int nonzero = 0;
        float sq = 0;
        int first_wrong = -1;
        for (int i = 0; i < LINEAR_CONV_DIM; i++) {
            sq += conv_out[i]*conv_out[i];
            if (conv_out[i] != 0.0f) nonzero++;
        }
        fprintf(stderr, "[LA] conv_out rms=%.6f nonzero=%d/%d\n",
                sqrtf(sq/LINEAR_CONV_DIM), nonzero, LINEAR_CONV_DIM);
        fprintf(stderr, "[LA] conv_out[6000..6004]=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                conv_out[6000], conv_out[6001], conv_out[6002], conv_out[6003], conv_out[6004]);
        fprintf(stderr, "[LA] conv_out[10000..10004]=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                conv_out[10000], conv_out[10001], conv_out[10002], conv_out[10003], conv_out[10004]);
        // Don't reset g_debug — keep printing for other stages
    }

    // ---- RMS normalize q and k per k-head, apply scaling ----
    // inv_scale = 1/sqrt(key_dim) = 1/sqrt(128)
    // q scaled by inv_scale^2 = 1/128
    // k scaled by inv_scale   = 1/sqrt(128)
    float inv_scale = 1.0f / sqrtf((float)LINEAR_KEY_DIM);
    float q_scale = inv_scale * inv_scale;  // 1/128

    for (int h = 0; h < LINEAR_NUM_K_HEADS; h++) {
        float* qh = lin_q + h * LINEAR_KEY_DIM;
        rms_norm_bare(qh, LINEAR_KEY_DIM, RMS_EPS);
        cblas_sscal(LINEAR_KEY_DIM, q_scale, qh, 1);
    }
    for (int h = 0; h < LINEAR_NUM_K_HEADS; h++) {
        float* kh = lin_k + h * LINEAR_KEY_DIM;
        rms_norm_bare(kh, LINEAR_KEY_DIM, RMS_EPS);
        cblas_sscal(LINEAR_KEY_DIM, inv_scale, kh, 1);
    }

    // ---- Precompute per-head decay (g) and beta gate ----
    float g_decay[LINEAR_NUM_V_HEADS];
    float beta_gate[LINEAR_NUM_V_HEADS];
    for (int vh = 0; vh < LINEAR_NUM_V_HEADS; vh++) {
        float a_val = alpha_proj[vh];
        float dt_b = bf16_to_f32(dt_bias[vh]);
        float A_val = expf(A_log[vh]);
        float softplus_val = logf(1.0f + expf(a_val + dt_b));
        g_decay[vh] = expf(-A_val * softplus_val);

        // beta = sigmoid(b) — no dt_bias
        beta_gate[vh] = 1.0f / (1.0f + expf(-beta_proj[vh]));
    }

    // ---- Delta-net recurrence (BLAS-accelerated) ----
    float kv_mem[LINEAR_VALUE_DIM];
    float delta[LINEAR_VALUE_DIM];

    for (int vh = 0; vh < LINEAR_NUM_V_HEADS; vh++) {
        int kh = vh / k_heads_per_v;

        float* S   = state->ssm_state + (size_t)vh * LINEAR_VALUE_DIM * LINEAR_KEY_DIM;
        float* v_h = lin_v + vh * LINEAR_VALUE_DIM;
        float* k_h = lin_k + kh * LINEAR_KEY_DIM;
        float* q_h = lin_q + kh * LINEAR_KEY_DIM;
        float* o_h = output + vh * LINEAR_VALUE_DIM;

        // Step 1: Decay state S[128x128] *= g
        cblas_sscal(LINEAR_VALUE_DIM * LINEAR_KEY_DIM, g_decay[vh], S, 1);

        // Step 2: kv_mem = S @ k  (matrix-vector: [128x128] @ [128] -> [128])
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    LINEAR_VALUE_DIM, LINEAR_KEY_DIM,
                    1.0f, S, LINEAR_KEY_DIM,
                    k_h, 1,
                    0.0f, kv_mem, 1);

        // Step 3: delta = (v - kv_mem) * beta
        float b = beta_gate[vh];
        for (int vi = 0; vi < LINEAR_VALUE_DIM; vi++) {
            delta[vi] = (v_h[vi] - kv_mem[vi]) * b;
        }

        // Step 4: S += outer(delta, k) — rank-1 update
        cblas_sger(CblasRowMajor,
                   LINEAR_VALUE_DIM, LINEAR_KEY_DIM,
                   1.0f,
                   delta, 1,
                   k_h, 1,
                   S, LINEAR_KEY_DIM);

        // Step 5: output = S @ q  (matrix-vector)
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    LINEAR_VALUE_DIM, LINEAR_KEY_DIM,
                    1.0f, S, LINEAR_KEY_DIM,
                    q_h, 1,
                    0.0f, o_h, 1);
    }

    if (g_debug) {
        float sq = 0;
        for (int i = 0; i < LINEAR_TOTAL_VALUE; i++) sq += output[i]*output[i];
        fprintf(stderr, "[LA] delta_net_out rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                sqrtf(sq/LINEAR_TOTAL_VALUE), output[0], output[1], output[2], output[3], output[4]);
    }

    // ---- Gated RMS norm per v-head ----
    float pre_norm[LINEAR_TOTAL_VALUE];
    memcpy(pre_norm, output, LINEAR_TOTAL_VALUE * sizeof(float));

    for (int vh = 0; vh < LINEAR_NUM_V_HEADS; vh++) {
        float* oh = pre_norm + vh * LINEAR_VALUE_DIM;
        float* zh = (float*)z_proj + vh * LINEAR_VALUE_DIM;
        float* gh = output + vh * LINEAR_VALUE_DIM;
        rms_norm_gated(oh, zh, gated_norm_w, gh, LINEAR_VALUE_DIM, RMS_EPS);
    }

    if (g_debug) {
        float sq = 0;
        for (int i = 0; i < LINEAR_TOTAL_VALUE; i++) sq += output[i]*output[i];
        fprintf(stderr, "[LA] gated_rms_out rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                sqrtf(sq/LINEAR_TOTAL_VALUE), output[0], output[1], output[2], output[3], output[4]);
    }
}
