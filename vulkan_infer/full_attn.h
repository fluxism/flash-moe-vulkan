#ifndef FULL_ATTN_H
#define FULL_ATTN_H

#include <stdint.h>

#define FA_NUM_ATTN_HEADS  32
#define FA_NUM_KV_HEADS    2
#define FA_HEAD_DIM        256
#define FA_ROTARY_DIM      64
#define FA_ROPE_THETA      10000000.0f
#define FA_KV_DIM          (FA_NUM_KV_HEADS * FA_HEAD_DIM)  // 512
#define FA_MAX_SEQ_LEN     8192

typedef struct {
    float* k_cache; // [FA_MAX_SEQ_LEN, FA_KV_DIM]
    float* v_cache; // [FA_MAX_SEQ_LEN, FA_KV_DIM]
    int seq_len;    // current position (0-indexed, increments each token)
} KVCache;

KVCache* kv_cache_create(void);
void kv_cache_destroy(KVCache* kv);

// Full attention forward pass for one token.
//
// q_proj_interleaved: [NUM_ATTN_HEADS * HEAD_DIM * 2] — per-head interleaved format:
//   for each head h, src[h * 2*HEAD_DIM .. h*2*HEAD_DIM + HEAD_DIM) = Q values,
//                    src[h * 2*HEAD_DIM + HEAD_DIM .. h*2*HEAD_DIM + 2*HEAD_DIM) = gate values
// k_proj: [KV_DIM=512] — K projection output (not interleaved)
// v_proj: [KV_DIM=512] — V projection output
// pos: position index for RoPE
// output: [NUM_ATTN_HEADS * HEAD_DIM = 8192] — attention output (gated, before o_proj)
void full_attn_forward(
    KVCache* kv,
    const float* q_proj_interleaved,
    const float* k_proj,
    const float* v_proj,
    int pos,
    float* output
);

#endif
