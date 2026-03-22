// full_attn.c — CPU-side full attention with RoPE and KV cache
// For the 15 full-attention layers (every 4th layer) of Qwen3.5-397B-A17B.
//
// RoPE uses MLX non-traditional pairing: (x[i], x[i + half_rotary_dim])
// not the adjacent-pair (x[2i], x[2i+1]) convention.

#include "full_attn.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// KV cache lifecycle
// ---------------------------------------------------------------------------

KVCache* kv_cache_create(void) {
    KVCache* kv = calloc(1, sizeof(KVCache));
    if (!kv) return NULL;
    kv->k_cache = calloc((size_t)FA_MAX_SEQ_LEN * FA_KV_DIM, sizeof(float));
    kv->v_cache = calloc((size_t)FA_MAX_SEQ_LEN * FA_KV_DIM, sizeof(float));
    if (!kv->k_cache || !kv->v_cache) {
        kv_cache_destroy(kv);
        return NULL;
    }
    kv->seq_len = 0;
    return kv;
}

void kv_cache_destroy(KVCache* kv) {
    if (kv) {
        free(kv->k_cache);
        free(kv->v_cache);
        free(kv);
    }
}

// ---------------------------------------------------------------------------
// Softmax in-place
// ---------------------------------------------------------------------------

static void softmax_inplace(float* x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++) {
        x[i] *= inv_sum;
    }
}

// ---------------------------------------------------------------------------
// RoPE: MLX non-traditional pairing (x[i], x[i + half]) not (x[2i], x[2i+1])
//
// Applied to first ROTARY_DIM=64 dimensions of each head.
// half = ROTARY_DIM / 2 = 32
// For i in [0, half):
//   freq  = 1 / ROPE_THETA^(2i / ROTARY_DIM)
//   angle = pos * freq
//   new[i]       = x[i]       * cos(angle) - x[i+half] * sin(angle)
//   new[i+half]  = x[i+half]  * cos(angle) + x[i]      * sin(angle)
// Dimensions [ROTARY_DIM, HEAD_DIM) are unchanged.
// ---------------------------------------------------------------------------

static void apply_rope_to_head(float* h, int pos) {
    const int half = FA_ROTARY_DIM / 2;  // 32
    for (int i = 0; i < half; i++) {
        float freq  = 1.0f / powf(FA_ROPE_THETA, (float)(2 * i) / FA_ROTARY_DIM);
        float angle = (float)pos * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);

        float x0 = h[i];
        float x1 = h[i + half];
        h[i]        = x0 * cos_a - x1 * sin_a;
        h[i + half] = x0 * sin_a + x1 * cos_a;
    }
}

// ---------------------------------------------------------------------------
// full_attn_forward
//
// q_proj_interleaved layout (per head, total NUM_ATTN_HEADS * 2 * HEAD_DIM):
//   head h: [h * 2*HEAD_DIM .. h*2*HEAD_DIM + HEAD_DIM)  -> Q values
//            [h * 2*HEAD_DIM + HEAD_DIM .. h*2*HEAD_DIM + 2*HEAD_DIM) -> gate values
//
// k_proj: [NUM_KV_HEADS * HEAD_DIM] = [512], contiguous heads, no gate
// v_proj: [NUM_KV_HEADS * HEAD_DIM] = [512], contiguous heads
// ---------------------------------------------------------------------------

void full_attn_forward(
    KVCache* kv,
    const float* q_proj_interleaved,
    const float* k_proj,
    const float* v_proj,
    int pos,
    float* output
) {
    const int q_dim  = FA_NUM_ATTN_HEADS * FA_HEAD_DIM;  // 8192
    const int kv_dim = FA_KV_DIM;                        // 512

    // ------------------------------------------------------------------
    // Step 1: Deinterleave Q and extract gate, then apply RoPE to Q.
    //
    // The Q projection output has HEAD_DIM Q values followed by HEAD_DIM
    // gate values for each head (per-head interleaved, not element-wise).
    // ------------------------------------------------------------------
    float* q      = malloc((size_t)q_dim * sizeof(float));
    float* q_gate = malloc((size_t)q_dim * sizeof(float));

    for (int h = 0; h < FA_NUM_ATTN_HEADS; h++) {
        const float* src = q_proj_interleaved + h * (2 * FA_HEAD_DIM);
        float* qh  = q      + h * FA_HEAD_DIM;
        float* gh  = q_gate + h * FA_HEAD_DIM;
        memcpy(qh, src,               FA_HEAD_DIM * sizeof(float));
        memcpy(gh, src + FA_HEAD_DIM, FA_HEAD_DIM * sizeof(float));
        // RoPE on Q head (in-place)
        apply_rope_to_head(qh, pos);
    }

    // ------------------------------------------------------------------
    // Step 2: Apply RoPE to K (k_proj is not interleaved with a gate).
    // We operate on a local copy so we don't mutate the caller's buffer.
    // ------------------------------------------------------------------
    float* k = malloc((size_t)kv_dim * sizeof(float));
    memcpy(k, k_proj, kv_dim * sizeof(float));

    for (int h = 0; h < FA_NUM_KV_HEADS; h++) {
        apply_rope_to_head(k + h * FA_HEAD_DIM, pos);
    }

    // ------------------------------------------------------------------
    // Step 3: Append K and V to the KV cache at the current position.
    // ------------------------------------------------------------------
    int cache_pos = kv->seq_len;
    memcpy(kv->k_cache + (size_t)cache_pos * kv_dim, k,      kv_dim * sizeof(float));
    memcpy(kv->v_cache + (size_t)cache_pos * kv_dim, v_proj, kv_dim * sizeof(float));
    kv->seq_len++;

    // ------------------------------------------------------------------
    // Step 4: Grouped-query attention.
    //
    // 32 query heads, 2 KV heads → heads_per_kv = 16.
    // For each query head h:
    //   kv_h = h / heads_per_kv
    //   scores[p] = dot(Q[h], K_cache[p][kv_h]) / sqrt(HEAD_DIM)
    //   softmax(scores)
    //   out[h] = sum_p scores[p] * V_cache[p][kv_h]
    // ------------------------------------------------------------------
    const int heads_per_kv = FA_NUM_ATTN_HEADS / FA_NUM_KV_HEADS;  // 16
    const float scale = 1.0f / sqrtf((float)FA_HEAD_DIM);
    const int seq = kv->seq_len;  // includes current token

    float* attn_out = calloc((size_t)q_dim, sizeof(float));
    float* scores   = malloc((size_t)seq * sizeof(float));

    for (int h = 0; h < FA_NUM_ATTN_HEADS; h++) {
        int kv_h = h / heads_per_kv;
        const float* qh = q + h * FA_HEAD_DIM;
        float* oh = attn_out + h * FA_HEAD_DIM;

        // Compute dot-product scores for every cached position
        for (int p = 0; p < seq; p++) {
            const float* kp = kv->k_cache + (size_t)p * kv_dim + kv_h * FA_HEAD_DIM;
            float dot = 0.0f;
            for (int d = 0; d < FA_HEAD_DIM; d++) {
                dot += qh[d] * kp[d];
            }
            scores[p] = dot * scale;
        }

        softmax_inplace(scores, seq);

        // Weighted sum of value vectors
        for (int p = 0; p < seq; p++) {
            const float* vp = kv->v_cache + (size_t)p * kv_dim + kv_h * FA_HEAD_DIM;
            float s = scores[p];
            for (int d = 0; d < FA_HEAD_DIM; d++) {
                oh[d] += s * vp[d];
            }
        }
    }

    free(scores);

    // ------------------------------------------------------------------
    // Step 5: Apply sigmoid gate element-wise to attention output.
    //
    // MLX: output = attn_out * sigmoid(gate)
    // ------------------------------------------------------------------
    for (int i = 0; i < q_dim; i++) {
        float g = 1.0f / (1.0f + expf(-q_gate[i]));
        output[i] = attn_out[i] * g;
    }

    free(q);
    free(q_gate);
    free(k);
    free(attn_out);
}
