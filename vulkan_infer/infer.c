/*
 * infer.c — Main inference engine for Flash-MoE Vulkan port
 *
 * Wires together Vulkan compute shaders, io_uring expert streaming,
 * CPU attention (linear + full), and BPE tokenizer into a complete
 * token-by-token generation pipeline for Qwen3.5-397B-A17B.
 *
 * Usage:
 *   ./infer --prompt "text" --tokens N [--timing] [--temperature T] [--top-p P]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <float.h>

// Debug helpers
int g_debug = 0;
static float vec_rms(const float* v, int n) {
    float sq = 0; for (int i = 0; i < n; i++) sq += v[i]*v[i];
    return sqrtf(sq / n);
}
static void dbg_vec(const char* label, const float* v, int n) {
    if (!g_debug) return;
    fprintf(stderr, "[DBG] %s: rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
            label, vec_rms(v, n), v[0], v[1], v[2], v[3], v[4]);
}

#include "vk_compute.h"
#include "io_ring.h"
#include "weights.h"
#include "linear_attn.h"
#include "full_attn.h"
#include "expert_cache.h"

#define TOKENIZER_IMPL
#include "tokenizer.h"

// ============================================================================
// Model constants
// ============================================================================

#define HIDDEN_DIM          4096
#define NUM_LAYERS          60
#define VOCAB_SIZE          248320
#define RMS_NORM_EPS        1e-6f
#define NUM_EXPERTS         512
#define MOE_INTERMEDIATE    1024
#define SHARED_INTERMEDIATE 1024
#define GROUP_SIZE          64
#define EXPERT_SIZE         7077888
#define FULL_ATTN_INTERVAL  4
#define K_EXPERTS           4
#define EOS_TOKEN_1         248046
#define EOS_TOKEN_2         248044
#define NUM_FULL_ATTN_LAYERS (NUM_LAYERS / FULL_ATTN_INTERVAL)  // 15
#define NUM_LINEAR_LAYERS   (NUM_LAYERS - NUM_FULL_ATTN_LAYERS) // 45

// Expert 4-bit layout offsets (bytes within each expert)
#define GATE_W_OFF  0
#define GATE_S_OFF  2097152
#define GATE_B_OFF  2228224
#define UP_W_OFF    2359296
#define UP_S_OFF    4456448
#define UP_B_OFF    4587520
#define DOWN_W_OFF  4718592
#define DOWN_S_OFF  6815744
#define DOWN_B_OFF  6946816

// Full attention constants (from full_attn.h)
#define FA_Q_PROJ_DIM       (FA_NUM_ATTN_HEADS * FA_HEAD_DIM * 2) // 16384
#define FA_O_PROJ_IN_DIM    (FA_NUM_ATTN_HEADS * FA_HEAD_DIM)     // 8192

// ============================================================================
// Timing
// ============================================================================

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

typedef struct {
    double input_norm;
    double attn_proj;
    double cpu_attn;
    double o_proj;
    double post_norm;
    double routing_proj;
    double cpu_routing;
    double expert_io;
    double expert_compute;
    double moe_combine;
    int    count;
} LayerTiming;

static LayerTiming g_timing;
static int g_timing_enabled = 0;

static void timing_reset(void) {
    memset(&g_timing, 0, sizeof(g_timing));
}

static void timing_print(void) {
    if (g_timing.count == 0) return;
    int n = g_timing.count;
    fprintf(stderr, "\n--- Per-token timing (avg over %d layers) ---\n", n);
    fprintf(stderr, "  input_norm:     %.3f ms\n", g_timing.input_norm / n);
    fprintf(stderr, "  attn_proj:      %.3f ms\n", g_timing.attn_proj / n);
    fprintf(stderr, "  cpu_attn:       %.3f ms\n", g_timing.cpu_attn / n);
    fprintf(stderr, "  o_proj:         %.3f ms\n", g_timing.o_proj / n);
    fprintf(stderr, "  post_norm:      %.3f ms\n", g_timing.post_norm / n);
    fprintf(stderr, "  routing_proj:   %.3f ms\n", g_timing.routing_proj / n);
    fprintf(stderr, "  cpu_routing:    %.3f ms\n", g_timing.cpu_routing / n);
    fprintf(stderr, "  expert_io:      %.3f ms\n", g_timing.expert_io / n);
    fprintf(stderr, "  expert_compute: %.3f ms\n", g_timing.expert_compute / n);
    fprintf(stderr, "  moe_combine:    %.3f ms\n", g_timing.moe_combine / n);
    double total = (g_timing.input_norm + g_timing.attn_proj + g_timing.cpu_attn +
                    g_timing.o_proj + g_timing.post_norm + g_timing.routing_proj +
                    g_timing.cpu_routing + g_timing.expert_io + g_timing.expert_compute +
                    g_timing.moe_combine) / n;
    fprintf(stderr, "  TOTAL:          %.3f ms/layer (%.2f ms/token est)\n", total, total * NUM_LAYERS);
}

// ============================================================================
// bf16 conversion helper
// ============================================================================

static inline float bf16_to_f32(uint16_t bf16) {
    uint32_t bits = (uint32_t)bf16 << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

// ============================================================================
// Layer weight cache — precomputed offsets into the weight buffer
// ============================================================================

typedef struct {
    // Input norm
    size_t input_norm_w_off;
    // Full attention (15 layers)
    size_t q_w_off, q_s_off, q_b_off;
    size_t k_w_off, k_s_off, k_b_off;
    size_t v_w_off, v_s_off, v_b_off;
    size_t o_w_off, o_s_off, o_b_off;
    // Linear attention (45 layers)
    size_t qkv_w_off, qkv_s_off, qkv_b_off;
    size_t z_w_off, z_s_off, z_b_off;
    size_t beta_w_off, beta_s_off, beta_b_off;
    size_t alpha_w_off, alpha_s_off, alpha_b_off;
    size_t out_proj_w_off, out_proj_s_off, out_proj_b_off;
    // Linear attention extra (CPU-side data pointers, not offsets)
    void* conv1d_w;   // bf16, accessed by CPU
    void* A_log;      // float, accessed by CPU
    void* dt_bias;    // bf16, accessed by CPU
    void* gated_norm_w; // bf16, accessed by CPU
    // GPU-side byte offsets into the weight VkBuf
    size_t conv1d_w_off;   // bf16: [LINEAR_CONV_DIM * CONV_KERNEL_SIZE] packed as uint16
    size_t A_log_off;      // float32: [LINEAR_NUM_V_HEADS]
    size_t dt_bias_off;    // bf16: [LINEAR_NUM_V_HEADS] packed as uint16
    size_t gated_norm_w_off; // bf16: [LINEAR_VALUE_DIM] packed as uint16
    // Post-attention norm
    size_t post_attn_norm_w_off;
    // Routing
    size_t gate_w_off, gate_s_off, gate_b_off;
    // Shared expert
    size_t sg_w_off, sg_s_off, sg_b_off;  // gate_proj
    size_t su_w_off, su_s_off, su_b_off;  // up_proj
    size_t sd_w_off, sd_s_off, sd_b_off;  // down_proj
    // Shared expert gate
    size_t seg_w_off, seg_s_off, seg_b_off;
    // Dimensions
    int is_full;
    int q_proj_dim;    // full: 16384; linear: 12288
    int kv_dim;        // full: 512
    int o_proj_in_dim; // full: 8192; linear: 8192
} LayerWeightCache;

static LayerWeightCache layer_cache[NUM_LAYERS];
static int layer_cache_built = 0;

static size_t get_tensor_offset(WeightFile* wf, const char* name) {
    TensorInfo* ti = weights_get_tensor(wf, name);
    if (!ti) {
        fprintf(stderr, "WARNING: tensor '%s' not found\n", name);
        return 0;
    }
    return ti->offset;
}

static void* get_tensor_ptr(WeightFile* wf, const char* name) {
    TensorInfo* ti = weights_get_tensor(wf, name);
    if (!ti) return NULL;
    return (char*)wf->mapped + ti->offset;
}

static void build_layer_cache(WeightFile* wf) {
    char name[256];
    for (int l = 0; l < NUM_LAYERS; l++) {
        LayerWeightCache* lc = &layer_cache[l];
        int is_full = ((l + 1) % FULL_ATTN_INTERVAL == 0);
        lc->is_full = is_full;

        // Input norm
        snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", l);
        lc->input_norm_w_off = get_tensor_offset(wf, name);

        if (is_full) {
            // Full attention projections
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.weight", l);
            lc->q_w_off = get_tensor_offset(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.scales", l);
            lc->q_s_off = get_tensor_offset(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.biases", l);
            lc->q_b_off = get_tensor_offset(wf, name);

            snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.weight", l);
            lc->k_w_off = get_tensor_offset(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.scales", l);
            lc->k_s_off = get_tensor_offset(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.biases", l);
            lc->k_b_off = get_tensor_offset(wf, name);

            snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.weight", l);
            lc->v_w_off = get_tensor_offset(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.scales", l);
            lc->v_s_off = get_tensor_offset(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.biases", l);
            lc->v_b_off = get_tensor_offset(wf, name);

            // O projection
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.weight", l);
            lc->o_w_off = get_tensor_offset(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.scales", l);
            lc->o_s_off = get_tensor_offset(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.biases", l);
            lc->o_b_off = get_tensor_offset(wf, name);

            lc->q_proj_dim = FA_Q_PROJ_DIM;        // 16384
            lc->kv_dim = FA_KV_DIM;                // 512
            lc->o_proj_in_dim = FA_O_PROJ_IN_DIM;  // 8192
        } else {
            // Linear attention projections
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_qkv.weight", l);
            lc->qkv_w_off = get_tensor_offset(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_qkv.scales", l);
            lc->qkv_s_off = get_tensor_offset(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_qkv.biases", l);
            lc->qkv_b_off = get_tensor_offset(wf, name);

            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_z.weight", l);
            lc->z_w_off = get_tensor_offset(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_z.scales", l);
            lc->z_s_off = get_tensor_offset(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_z.biases", l);
            lc->z_b_off = get_tensor_offset(wf, name);

            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_b.weight", l);
            lc->beta_w_off = get_tensor_offset(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_b.scales", l);
            lc->beta_s_off = get_tensor_offset(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_b.biases", l);
            lc->beta_b_off = get_tensor_offset(wf, name);

            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_a.weight", l);
            lc->alpha_w_off = get_tensor_offset(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_a.scales", l);
            lc->alpha_s_off = get_tensor_offset(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_a.biases", l);
            lc->alpha_b_off = get_tensor_offset(wf, name);

            // Out projection (linear_attn.out_proj)
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.out_proj.weight", l);
            lc->out_proj_w_off = get_tensor_offset(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.out_proj.scales", l);
            lc->out_proj_s_off = get_tensor_offset(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.out_proj.biases", l);
            lc->out_proj_b_off = get_tensor_offset(wf, name);

            // CPU-side linear attn params (direct pointers)
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.conv1d.weight", l);
            lc->conv1d_w = get_tensor_ptr(wf, name);
            lc->conv1d_w_off = get_tensor_offset(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.A_log", l);
            lc->A_log = get_tensor_ptr(wf, name);
            lc->A_log_off = get_tensor_offset(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.dt_bias", l);
            lc->dt_bias = get_tensor_ptr(wf, name);
            lc->dt_bias_off = get_tensor_offset(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.norm.weight", l);
            lc->gated_norm_w = get_tensor_ptr(wf, name);
            lc->gated_norm_w_off = get_tensor_offset(wf, name);

            lc->q_proj_dim = LINEAR_CONV_DIM;           // 12288
            lc->kv_dim = 0;
            lc->o_proj_in_dim = LINEAR_TOTAL_VALUE;     // 8192
        }

        // Post-attention norm
        snprintf(name, sizeof(name), "model.layers.%d.post_attention_layernorm.weight", l);
        lc->post_attn_norm_w_off = get_tensor_offset(wf, name);

        // Routing gate
        snprintf(name, sizeof(name), "model.layers.%d.mlp.gate.weight", l);
        lc->gate_w_off = get_tensor_offset(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.gate.scales", l);
        lc->gate_s_off = get_tensor_offset(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.gate.biases", l);
        lc->gate_b_off = get_tensor_offset(wf, name);

        // Shared expert gate_proj
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.gate_proj.weight", l);
        lc->sg_w_off = get_tensor_offset(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.gate_proj.scales", l);
        lc->sg_s_off = get_tensor_offset(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.gate_proj.biases", l);
        lc->sg_b_off = get_tensor_offset(wf, name);

        // Shared expert up_proj
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.up_proj.weight", l);
        lc->su_w_off = get_tensor_offset(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.up_proj.scales", l);
        lc->su_s_off = get_tensor_offset(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.up_proj.biases", l);
        lc->su_b_off = get_tensor_offset(wf, name);

        // Shared expert down_proj
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.down_proj.weight", l);
        lc->sd_w_off = get_tensor_offset(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.down_proj.scales", l);
        lc->sd_s_off = get_tensor_offset(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.down_proj.biases", l);
        lc->sd_b_off = get_tensor_offset(wf, name);

        // Shared expert gate (scalar)
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert_gate.weight", l);
        lc->seg_w_off = get_tensor_offset(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert_gate.scales", l);
        lc->seg_s_off = get_tensor_offset(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert_gate.biases", l);
        lc->seg_b_off = get_tensor_offset(wf, name);
    }
    layer_cache_built = 1;
    fprintf(stderr, "[init] Layer weight cache built (%d layers)\n", NUM_LAYERS);
}

// ============================================================================
// GPU pipeline handles
// ============================================================================

typedef struct {
    VkPipe* dequant_matvec;
    VkPipe* rms_norm;
    VkPipe* residual_add;
    VkPipe* fused_gate_up_swiglu;
    VkPipe* moe_combine;
    VkPipe* sigmoid_gate;
    VkPipe* attn_scores;
    VkPipe* attn_softmax;
    VkPipe* attn_values;
    // GPU linear attention (GatedDeltaNet)
    VkPipe* conv1d_step;
    VkPipe* rms_norm_qk;
    VkPipe* compute_decay_beta;
    VkPipe* gated_delta_net_step;
    VkPipe* gated_rms_norm;
} Pipelines;

// ============================================================================
// GPU scratch buffers
// ============================================================================

typedef struct {
    // Main hidden state / working buffers
    VkBuf* hidden;        // [HIDDEN_DIM] current hidden state on GPU
    VkBuf* normed;        // [HIDDEN_DIM] after rms_norm
    VkBuf* residual;      // [HIDDEN_DIM] saved for residual connection

    // Attention projection outputs
    VkBuf* attn_proj;     // [max(FA_Q_PROJ_DIM, LINEAR_CONV_DIM)] = [16384]
    VkBuf* attn_proj2;    // [max(FA_KV_DIM, LINEAR_TOTAL_VALUE)] = [8192] for k/z
    VkBuf* attn_proj3;    // [FA_KV_DIM] = [512] for v
    VkBuf* attn_proj4;    // [LINEAR_NUM_V_HEADS] = [64] for beta
    VkBuf* attn_proj5;    // [LINEAR_NUM_V_HEADS] = [64] for alpha

    // Attention output (from CPU)
    VkBuf* attn_out;      // [FA_O_PROJ_IN_DIM] = [8192]

    // O projection output
    VkBuf* o_proj_out;    // [HIDDEN_DIM]

    // MoE routing
    VkBuf* routing_scores;// [NUM_EXPERTS]

    // Shared expert
    VkBuf* shared_act;    // [SHARED_INTERMEDIATE] after fused_gate_up_swiglu
    VkBuf* shared_out;    // [HIDDEN_DIM] after down_proj

    // Shared expert gate scalar
    VkBuf* shared_gate_score; // [1]

    // Expert data (4 experts)
    VkBuf* expert_data[K_EXPERTS]; // [EXPERT_SIZE] each
    VkBuf* expert_act[K_EXPERTS];  // [MOE_INTERMEDIATE]
    VkBuf* expert_out[K_EXPERTS];  // [HIDDEN_DIM]

    // MoE combine params [5 floats: w0,w1,w2,w3,shared_gate]
    VkBuf* moe_params;

    // Logits
    VkBuf* logits;        // [VOCAB_SIZE]

    // Full attention extra
    VkBuf* kv_k[NUM_FULL_ATTN_LAYERS]; // [FA_MAX_SEQ_LEN * FA_KV_DIM]
    VkBuf* kv_v[NUM_FULL_ATTN_LAYERS];
    VkBuf* attn_scores_buf; // [FA_NUM_ATTN_HEADS * FA_MAX_SEQ_LEN]

    // GPU linear attention persistent state (per linear layer)
    VkBuf* conv_state[NUM_LINEAR_LAYERS];     // [(CONV_KERNEL_SIZE-1) * LINEAR_CONV_DIM] float per layer
    VkBuf* delta_state[NUM_LINEAR_LAYERS];    // [LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM * LINEAR_KEY_DIM] float per layer

    // GPU linear attention scratch buffers
    VkBuf* conv_output;        // [LINEAR_CONV_DIM] float
    VkBuf* delta_g_decay;      // [LINEAR_NUM_V_HEADS] float
    VkBuf* delta_beta;         // [LINEAR_NUM_V_HEADS] float
    VkBuf* delta_output;       // [LINEAR_TOTAL_VALUE] float
} Buffers;

// ============================================================================
// Embedding lookup (CPU, 4-bit quantized)
// ============================================================================

static void embed_lookup(WeightFile* wf, int token_id, float* out) {
    TensorInfo* w_info = weights_get_tensor(wf, "model.embed_tokens.weight");
    TensorInfo* s_info = weights_get_tensor(wf, "model.embed_tokens.scales");
    TensorInfo* b_info = weights_get_tensor(wf, "model.embed_tokens.biases");

    if (!w_info || !s_info || !b_info) {
        fprintf(stderr, "ERROR: embedding tensors not found\n");
        memset(out, 0, HIDDEN_DIM * sizeof(float));
        return;
    }

    int packed_cols = w_info->shape[1]; // 512
    int num_groups = s_info->shape[1];  // 64

    uint32_t* W = (uint32_t*)((char*)wf->mapped + w_info->offset);
    uint16_t* S = (uint16_t*)((char*)wf->mapped + s_info->offset);
    uint16_t* B = (uint16_t*)((char*)wf->mapped + b_info->offset);

    const uint32_t* w_row = W + (size_t)token_id * packed_cols;
    const uint16_t* s_row = S + (size_t)token_id * num_groups;
    const uint16_t* b_row = B + (size_t)token_id * num_groups;

    int group_size = HIDDEN_DIM / num_groups;  // 64
    int packed_per_group = group_size / 8;     // 8

    for (int g = 0; g < num_groups; g++) {
        float scale = bf16_to_f32(s_row[g]);
        float bias  = bf16_to_f32(b_row[g]);

        for (int p = 0; p < packed_per_group; p++) {
            uint32_t packed = w_row[g * packed_per_group + p];
            int base = g * group_size + p * 8;
            for (int n = 0; n < 8; n++) {
                uint32_t nibble = (packed >> (n * 4)) & 0xF;
                out[base + n] = (float)nibble * scale + bias;
            }
        }
    }
}

// ============================================================================
// GPU dispatch helpers
// ============================================================================

static void gpu_dequant_matvec(VkCtx* ctx __attribute__((unused)), VkCmd* cmd, VkPipe* pipe,
    VkBuf* w_buf, size_t w_off,
    VkBuf* s_buf, size_t s_off,
    VkBuf* b_buf, size_t b_off,
    VkBuf* x_buf,
    VkBuf* out_buf,
    uint32_t out_dim, uint32_t in_dim, uint32_t group_size)
{
    // Binding: 0=weights, 1=scales, 2=biases, 3=input, 4=output
    VkBuf* bufs[5] = { w_buf, s_buf, b_buf, x_buf, out_buf };

    // Weight: out_dim rows, each row has in_dim/8 uint32s = in_dim/2 bytes
    size_t w_range = (size_t)out_dim * (in_dim / 8) * sizeof(uint32_t);
    // Scales: out_dim * (in_dim/group_size) bf16 values packed as uint16
    uint32_t num_groups = in_dim / group_size;
    size_t s_range = (size_t)out_dim * num_groups * sizeof(uint16_t);
    size_t b_range = s_range;

    size_t offsets[5] = { w_off, s_off, b_off, 0, 0 };
    size_t ranges[5]  = { w_range, s_range, b_range, in_dim * sizeof(float),
                          out_dim * sizeof(float) };

    struct { uint32_t out_dim; uint32_t in_dim; uint32_t group_size; } pc = {
        out_dim, in_dim, group_size
    };

    vk_cmd_bind(cmd, pipe, bufs, offsets, ranges, 5, &pc, sizeof(pc));
    // workgroup size 256, each subgroup handles one output row.
    // Assuming subgroup size 32 -> 8 rows/workgroup.
    // For safety, assume 1 row per workgroup (worst case subgroup=256)
    // Actually: rows_per_tg = 256/subgroup_size. Use conservative estimate.
    // At subgroup_size=32: rows_per_tg=8, need out_dim/8 workgroups.
    // At subgroup_size=64: rows_per_tg=4, need out_dim/4 workgroups.
    // Since we don't know subgroup size, dispatch ceil(out_dim/1) = out_dim workgroups
    // and let excess early-exit via the `if (row >= out_dim) return;` guard.
    // Dispatch conservatively: assume rows_per_tg = 4 (wave64).
    // Excess workgroups early-exit via `if (row >= out_dim) return;` in shader.
    uint32_t wg = (out_dim + 3) / 4;
    vk_cmd_dispatch(cmd, wg, 1, 1);
}

static void gpu_rms_norm(VkCtx* ctx __attribute__((unused)), VkCmd* cmd, VkPipe* pipe,
    VkBuf* x_buf, VkBuf* weight_buf, size_t w_off, VkBuf* out_buf,
    uint32_t dim, float eps)
{
    // Binding: 0=x, 1=weight(bf16), 2=output
    VkBuf* bufs[3] = { x_buf, weight_buf, out_buf };
    size_t offsets[3] = { 0, w_off, 0 };
    // Weight: dim bf16 values packed as uint16, but shader reads as uint pairs
    size_t w_range = (size_t)dim * sizeof(uint16_t);
    size_t ranges[3] = { dim * sizeof(float), w_range, dim * sizeof(float) };

    struct { uint32_t dim; float eps; } pc = { dim, eps };

    vk_cmd_bind(cmd, pipe, bufs, offsets, ranges, 3, &pc, sizeof(pc));
    vk_cmd_dispatch(cmd, 1, 1, 1); // Single workgroup of 256 threads
}

static void gpu_residual_add(VkCtx* ctx __attribute__((unused)), VkCmd* cmd, VkPipe* pipe,
    VkBuf* a_buf, VkBuf* b_buf, VkBuf* out_buf, uint32_t dim)
{
    VkBuf* bufs[3] = { a_buf, b_buf, out_buf };
    size_t ranges[3] = { dim * sizeof(float), dim * sizeof(float), dim * sizeof(float) };

    struct { uint32_t dim; } pc = { dim };
    vk_cmd_bind(cmd, pipe, bufs, NULL, ranges, 3, &pc, sizeof(pc));
    vk_cmd_dispatch(cmd, (dim + 255) / 256, 1, 1);
}

static void gpu_fused_gate_up_swiglu(VkCtx* ctx __attribute__((unused)), VkCmd* cmd, VkPipe* pipe,
    VkBuf* gate_w_buf, size_t gw_off,
    VkBuf* gate_s_buf, size_t gs_off,
    VkBuf* gate_b_buf, size_t gb_off,
    VkBuf* up_w_buf,   size_t uw_off,
    VkBuf* up_s_buf,   size_t us_off,
    VkBuf* up_b_buf,   size_t ub_off,
    VkBuf* x_buf, VkBuf* out_buf,
    uint32_t out_dim, uint32_t in_dim, uint32_t group_size)
{
    VkBuf* bufs[8] = { gate_w_buf, gate_s_buf, gate_b_buf,
                        up_w_buf, up_s_buf, up_b_buf,
                        x_buf, out_buf };

    size_t w_range = (size_t)out_dim * (in_dim / 8) * sizeof(uint32_t);
    uint32_t num_groups = in_dim / group_size;
    size_t s_range = (size_t)out_dim * num_groups * sizeof(uint16_t);

    size_t offsets[8] = { gw_off, gs_off, gb_off, uw_off, us_off, ub_off, 0, 0 };
    size_t ranges[8] = { w_range, s_range, s_range,
                         w_range, s_range, s_range,
                         in_dim * sizeof(float), out_dim * sizeof(float) };

    struct { uint32_t out_dim; uint32_t in_dim; uint32_t group_size; } pc = {
        out_dim, in_dim, group_size
    };

    vk_cmd_bind(cmd, pipe, bufs, offsets, ranges, 8, &pc, sizeof(pc));
    uint32_t wg = (out_dim + 3) / 4; // wave64: 4 rows per workgroup
    vk_cmd_dispatch(cmd, wg, 1, 1);
}

static void gpu_moe_combine(VkCtx* ctx __attribute__((unused)), VkCmd* cmd, VkPipe* pipe,
    VkBuf* h_mid_buf, VkBuf* shared_out_buf, VkBuf* hidden_out_buf,
    VkBuf* expert_out_bufs[K_EXPERTS], VkBuf* params_buf,
    uint32_t dim, uint32_t K)
{
    VkBuf* bufs[8] = { h_mid_buf, shared_out_buf, hidden_out_buf,
                        expert_out_bufs[0], expert_out_bufs[1],
                        expert_out_bufs[2], expert_out_bufs[3],
                        params_buf };
    size_t ranges[8];
    for (int i = 0; i < 8; i++) ranges[i] = dim * sizeof(float);
    ranges[7] = 5 * sizeof(float); // params = [w0, w1, w2, w3, shared_gate_score]

    struct { uint32_t dim; uint32_t K; } pc = { dim, K };
    vk_cmd_bind(cmd, pipe, bufs, NULL, ranges, 8, &pc, sizeof(pc));
    vk_cmd_dispatch(cmd, (dim + 255) / 256, 1, 1);
}

// ============================================================================
// CPU softmax / top-K / sampling
// ============================================================================

static void cpu_softmax(float* x, int n) {
    float max_val = -FLT_MAX;
    for (int i = 0; i < n; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

static void cpu_topk(const float* scores, int n, int k,
                     int* indices, float* weights) {
    // Simple selection sort for small k
    for (int i = 0; i < k; i++) {
        float best = -FLT_MAX;
        int best_idx = 0;
        for (int j = 0; j < n; j++) {
            // Skip already selected
            int skip = 0;
            for (int s = 0; s < i; s++) if (indices[s] == j) { skip = 1; break; }
            if (!skip && scores[j] > best) { best = scores[j]; best_idx = j; }
        }
        indices[i] = best_idx;
        weights[i] = best;
    }
    // Normalize weights
    float sum = 0.0f;
    for (int i = 0; i < k; i++) sum += weights[i];
    if (sum > 0.0f) {
        float inv = 1.0f / sum;
        for (int i = 0; i < k; i++) weights[i] *= inv;
    }
}

static int cpu_argmax(const float* x, int n) {
    float best = -FLT_MAX;
    int best_idx = 0;
    for (int i = 0; i < n; i++) {
        if (x[i] > best) { best = x[i]; best_idx = i; }
    }
    return best_idx;
}

static int sample_top_p(const float* logits, int vocab_size,
                         float temperature, float top_p) {
    // Temperature-scaled logits
    float* probs = malloc(vocab_size * sizeof(float));
    for (int i = 0; i < vocab_size; i++) probs[i] = logits[i] / temperature;
    cpu_softmax(probs, vocab_size);

    // Sort indices by probability (descending) — use simple selection for top-p
    // Accumulate until top-p threshold
    float cumsum = 0.0f;
    int* indices = malloc(vocab_size * sizeof(int));
    for (int i = 0; i < vocab_size; i++) indices[i] = i;

    // Partial sort: find nucleus
    int nucleus_size = 0;
    float nucleus_sum = 0.0f;
    while (nucleus_sum < top_p && nucleus_size < vocab_size) {
        // Find max in remaining
        float best = -1.0f;
        int best_pos = nucleus_size;
        for (int i = nucleus_size; i < vocab_size; i++) {
            if (probs[indices[i]] > best) {
                best = probs[indices[i]];
                best_pos = i;
            }
        }
        // Swap
        int tmp = indices[nucleus_size];
        indices[nucleus_size] = indices[best_pos];
        indices[best_pos] = tmp;
        nucleus_sum += best;
        nucleus_size++;
    }
    (void)cumsum;

    // Re-normalize nucleus
    for (int i = 0; i < nucleus_size; i++) {
        probs[indices[i]] /= nucleus_sum;
    }

    // Sample from nucleus
    float r = (float)rand() / (float)RAND_MAX;
    float acc = 0.0f;
    int chosen = indices[0];
    for (int i = 0; i < nucleus_size; i++) {
        acc += probs[indices[i]];
        if (r <= acc) { chosen = indices[i]; break; }
    }

    free(probs);
    free(indices);
    return chosen;
}

// ============================================================================
// Token decoding — uses the bpe_tokenizer's vocab directly
// ============================================================================

static const char* decode_token(bpe_tokenizer* tok, int token_id) {
    if (!tok || token_id < 0 || (uint32_t)token_id >= tok->vocab_size) return "<unk>";
    return tok->vocab[token_id].str ? tok->vocab[token_id].str : "<unk>";
}

// ============================================================================
// Deferred command state — tracks a GPU command buffer submitted without
// waiting, so expert compute + moe_combine can overlap with next layer's
// CPU routing + I/O.
// ============================================================================

typedef struct {
    VkCmd* cmd;     // Deferred command buffer (NULL if none pending)
    VkCtx* ctx;     // Context for destroy
} DeferredState;

static DeferredState g_deferred = {0};

// Wait for and clean up any pending deferred command.
// After overlap, this wait should be near-instant since the GPU already
// executed the deferred work while we were doing CPU routing + I/O.
static void deferred_wait(void) {
    if (g_deferred.cmd) {
        vk_cmd_wait(g_deferred.cmd);
        vk_cmd_destroy(g_deferred.ctx, g_deferred.cmd);
        g_deferred.cmd = NULL;
        g_deferred.ctx = NULL;
    }
}

// ============================================================================
// fused_layer_forward — core per-layer computation
//
// Optimized pipeline: merge GPU phases into fewer command buffers and defer
// expert compute + moe_combine to overlap with next layer's setup.
//
// Linear attention layers (45/60):
//   CMD_A: input_norm + attn_proj + GPU_linear_attn + o_proj + residual
//          + post_norm + routing + shared_expert  [submit+wait]
//   CPU: softmax + topK + io_uring expert read
//   CMD_B: expert compute + moe_combine  [submit NO WAIT — deferred]
//
// Full attention layers (15/60):
//   CMD_A: input_norm + attn_proj  [submit+wait — need Q/K/V on CPU]
//   CPU: full_attn_forward
//   CMD_B: o_proj + residual + post_norm + routing + shared_expert  [submit+wait]
//   CPU: softmax + topK + io_uring expert read
//   CMD_C: expert compute + moe_combine  [submit NO WAIT — deferred]
// ============================================================================

static void fused_layer_forward(
    VkCtx* ctx,
    WeightFile* wf,
    Pipelines* pipes,
    Buffers* bufs,
    int layer_idx,
    float* hidden,       // [HIDDEN_DIM] CPU in/out
    KVCache* kv,         // non-NULL for full attention layers
    LinearAttnState* la_state __attribute__((unused)), // non-NULL for linear attention layers
    int pos,
    int packed_fd,
    IoRing* ring,
    ExpertCache* ecache)
{
    double t0 = 0, t1 = 0;
    LayerWeightCache* lc = &layer_cache[layer_idx];
    int is_full = lc->is_full;

    // Wait for any deferred expert compute from previous layer.
    // This should be near-instant: the GPU executed CMD_B/CMD_C while we
    // did CPU routing + I/O for this layer's setup.
    deferred_wait();

    // Upload hidden state to GPU (only needed for first layer of each token,
    // or after CPU full attention modified hidden on CPU side).
    // For subsequent linear-attn layers, hidden is already correct on GPU
    // from the previous layer's moe_combine.
    if (layer_idx == 0) {
        memcpy(vk_buf_map(bufs->hidden), hidden, HIDDEN_DIM * sizeof(float));
    }

    if (is_full) {
        // =================================================================
        // FULL ATTENTION PATH: 3 command buffers (2 waited, 1 deferred)
        // =================================================================

        // CMD_A: input_norm + attn projections (Q, K, V)
        // Must wait because CPU needs Q/K/V for full_attn_forward
        if (g_timing_enabled) t0 = now_ms();
        {
            VkCmd* cmd = vk_cmd_begin(ctx);

            // Phase 1: Input RMS norm
            gpu_rms_norm(ctx, cmd, pipes->rms_norm,
                         bufs->hidden, wf->buf, lc->input_norm_w_off, bufs->normed,
                         HIDDEN_DIM, RMS_NORM_EPS);
            vk_cmd_barrier(cmd);

            // Phase 2: Q, K, V projections
            // Q: [4096] -> [16384] (interleaved Q + gate)
            gpu_dequant_matvec(ctx, cmd, pipes->dequant_matvec,
                wf->buf, lc->q_w_off, wf->buf, lc->q_s_off, wf->buf, lc->q_b_off,
                bufs->normed, bufs->attn_proj,
                FA_Q_PROJ_DIM, HIDDEN_DIM, GROUP_SIZE);
            vk_cmd_barrier(cmd);

            // K: [4096] -> [512]
            gpu_dequant_matvec(ctx, cmd, pipes->dequant_matvec,
                wf->buf, lc->k_w_off, wf->buf, lc->k_s_off, wf->buf, lc->k_b_off,
                bufs->normed, bufs->attn_proj2,
                FA_KV_DIM, HIDDEN_DIM, GROUP_SIZE);
            vk_cmd_barrier(cmd);

            // V: [4096] -> [512]
            gpu_dequant_matvec(ctx, cmd, pipes->dequant_matvec,
                wf->buf, lc->v_w_off, wf->buf, lc->v_s_off, wf->buf, lc->v_b_off,
                bufs->normed, bufs->attn_proj3,
                FA_KV_DIM, HIDDEN_DIM, GROUP_SIZE);

            vk_cmd_submit(cmd);
            vk_cmd_destroy(ctx, cmd);
        }
        if (g_timing_enabled) { t1 = now_ms(); g_timing.input_norm += t1 - t0; g_timing.attn_proj += t1 - t0; }

        // Phase 3: CPU full attention
        if (g_timing_enabled) t0 = now_ms();
        {
            float* q_proj = (float*)vk_buf_map(bufs->attn_proj);   // [16384]
            float* k_proj = (float*)vk_buf_map(bufs->attn_proj2);  // [512]
            float* v_proj = (float*)vk_buf_map(bufs->attn_proj3);  // [512]

            float attn_out[FA_O_PROJ_IN_DIM]; // [8192]
            full_attn_forward(kv, q_proj, k_proj, v_proj, pos, attn_out);

            memcpy(vk_buf_map(bufs->attn_out), attn_out, FA_O_PROJ_IN_DIM * sizeof(float));
        }
        if (g_timing_enabled) { t1 = now_ms(); g_timing.cpu_attn += t1 - t0; }

        // CMD_B: o_proj + residual + post_norm + routing + shared expert
        // Must wait because CPU needs routing scores
        if (g_timing_enabled) t0 = now_ms();
        {
            VkCmd* cmd = vk_cmd_begin(ctx);

            // Phase 4: O projection
            gpu_dequant_matvec(ctx, cmd, pipes->dequant_matvec,
                wf->buf, lc->o_w_off, wf->buf, lc->o_s_off, wf->buf, lc->o_b_off,
                bufs->attn_out, bufs->o_proj_out,
                HIDDEN_DIM, FA_O_PROJ_IN_DIM, GROUP_SIZE);
            vk_cmd_barrier(cmd);

            // Residual add: hidden + o_proj_out -> hidden
            gpu_residual_add(ctx, cmd, pipes->residual_add,
                bufs->o_proj_out, bufs->hidden, bufs->hidden, HIDDEN_DIM);
            vk_cmd_barrier(cmd);

            // Copy hidden -> residual for MoE (on GPU via residual_add with zero trick:
            // actually just use a barrier and read hidden as residual later in moe_combine).
            // We need residual saved. Use residual_add: hidden + 0 -> residual? No, just copy.
            // Simplest: do the memcpy after this cmd. But that adds a round-trip.
            // Better: encode the post-norm chain here and save residual in the moe_combine shader.
            // For now: copy via GPU by dispatching residual_add(hidden, hidden, residual) with
            // a special "copy" semantic. Actually residual_add does a[i]+b[i]->out[i], so
            // we'd get 2*hidden. Instead, just use the memcpy approach but on mapped memory
            // AFTER this cmd completes. Actually we need to save residual BEFORE post_norm
            // modifies hidden. But post_norm reads hidden and writes to normed, not hidden.
            // So hidden is unchanged after post_norm. We can save it later.

            // Phase 5: Post-attention norm
            gpu_rms_norm(ctx, cmd, pipes->rms_norm,
                         bufs->hidden, wf->buf, lc->post_attn_norm_w_off, bufs->normed,
                         HIDDEN_DIM, RMS_NORM_EPS);
            vk_cmd_barrier(cmd);

            // Routing: [4096] -> [512]
            gpu_dequant_matvec(ctx, cmd, pipes->dequant_matvec,
                wf->buf, lc->gate_w_off, wf->buf, lc->gate_s_off, wf->buf, lc->gate_b_off,
                bufs->normed, bufs->routing_scores,
                NUM_EXPERTS, HIDDEN_DIM, GROUP_SIZE);
            vk_cmd_barrier(cmd);

            // Shared expert fused gate_up_swiglu: [4096] -> [1024]
            gpu_fused_gate_up_swiglu(ctx, cmd, pipes->fused_gate_up_swiglu,
                wf->buf, lc->sg_w_off, wf->buf, lc->sg_s_off, wf->buf, lc->sg_b_off,
                wf->buf, lc->su_w_off, wf->buf, lc->su_s_off, wf->buf, lc->su_b_off,
                bufs->normed, bufs->shared_act,
                SHARED_INTERMEDIATE, HIDDEN_DIM, GROUP_SIZE);
            vk_cmd_barrier(cmd);

            // Shared expert down_proj: [1024] -> [4096]
            gpu_dequant_matvec(ctx, cmd, pipes->dequant_matvec,
                wf->buf, lc->sd_w_off, wf->buf, lc->sd_s_off, wf->buf, lc->sd_b_off,
                bufs->shared_act, bufs->shared_out,
                HIDDEN_DIM, SHARED_INTERMEDIATE, GROUP_SIZE);
            vk_cmd_barrier(cmd);

            // Shared expert gate: [4096] -> [1]
            gpu_dequant_matvec(ctx, cmd, pipes->dequant_matvec,
                wf->buf, lc->seg_w_off, wf->buf, lc->seg_s_off, wf->buf, lc->seg_b_off,
                bufs->normed, bufs->shared_gate_score,
                1, HIDDEN_DIM, GROUP_SIZE);

            vk_cmd_submit(cmd);
            vk_cmd_destroy(ctx, cmd);
        }
        if (g_timing_enabled) { t1 = now_ms(); g_timing.o_proj += t1 - t0; g_timing.post_norm += t1 - t0; }

    } else {
        // =================================================================
        // LINEAR ATTENTION PATH: 2 command buffers (1 waited, 1 deferred)
        // Merge phases 1+2+4+5 into ONE command buffer.
        // =================================================================

        if (g_timing_enabled) t0 = now_ms();
        {
            VkCmd* cmd = vk_cmd_begin(ctx);

            // Phase 1: Input RMS norm
            gpu_rms_norm(ctx, cmd, pipes->rms_norm,
                         bufs->hidden, wf->buf, lc->input_norm_w_off, bufs->normed,
                         HIDDEN_DIM, RMS_NORM_EPS);
            vk_cmd_barrier(cmd);

            // Phase 2: Attention projections
            // QKV: [4096] -> [12288]
            gpu_dequant_matvec(ctx, cmd, pipes->dequant_matvec,
                wf->buf, lc->qkv_w_off, wf->buf, lc->qkv_s_off, wf->buf, lc->qkv_b_off,
                bufs->normed, bufs->attn_proj,
                LINEAR_CONV_DIM, HIDDEN_DIM, GROUP_SIZE);
            vk_cmd_barrier(cmd);

            // Z: [4096] -> [8192]
            gpu_dequant_matvec(ctx, cmd, pipes->dequant_matvec,
                wf->buf, lc->z_w_off, wf->buf, lc->z_s_off, wf->buf, lc->z_b_off,
                bufs->normed, bufs->attn_proj2,
                LINEAR_TOTAL_VALUE, HIDDEN_DIM, GROUP_SIZE);
            vk_cmd_barrier(cmd);

            // Beta: [4096] -> [64]
            gpu_dequant_matvec(ctx, cmd, pipes->dequant_matvec,
                wf->buf, lc->beta_w_off, wf->buf, lc->beta_s_off, wf->buf, lc->beta_b_off,
                bufs->normed, bufs->attn_proj4,
                LINEAR_NUM_V_HEADS, HIDDEN_DIM, GROUP_SIZE);
            vk_cmd_barrier(cmd);

            // Alpha: [4096] -> [64]
            gpu_dequant_matvec(ctx, cmd, pipes->dequant_matvec,
                wf->buf, lc->alpha_w_off, wf->buf, lc->alpha_s_off, wf->buf, lc->alpha_b_off,
                bufs->normed, bufs->attn_proj5,
                LINEAR_NUM_V_HEADS, HIDDEN_DIM, GROUP_SIZE);
            vk_cmd_barrier(cmd);

            // Phase 2b: GPU linear attention (GatedDeltaNet)
            int linear_layer_idx = layer_idx - (layer_idx + 1) / FULL_ATTN_INTERVAL;

            // L1: conv1d_step
            {
                uint32_t conv_dim = LINEAR_CONV_DIM;
                VkBuf* conv_bufs[] = { bufs->conv_state[linear_layer_idx],
                                       bufs->attn_proj, wf->buf, bufs->conv_output };
                size_t conv_offsets[] = { 0, 0, lc->conv1d_w_off, 0 };
                size_t conv_ranges[] = {
                    (CONV_KERNEL_SIZE-1) * LINEAR_CONV_DIM * sizeof(float),
                    LINEAR_CONV_DIM * sizeof(float),
                    LINEAR_CONV_DIM * CONV_KERNEL_SIZE * sizeof(uint16_t),
                    LINEAR_CONV_DIM * sizeof(float)
                };
                vk_cmd_bind(cmd, pipes->conv1d_step, conv_bufs, conv_offsets, conv_ranges, 4,
                            &conv_dim, sizeof(conv_dim));
                vk_cmd_dispatch(cmd, (LINEAR_CONV_DIM + 255) / 256, 1, 1);
            }
            vk_cmd_barrier(cmd);

            // L2: rms_norm_qk
            {
                struct { uint32_t key_dim; float inv_scale; } pc = {
                    LINEAR_KEY_DIM, 1.0f / sqrtf((float)LINEAR_KEY_DIM)
                };
                VkBuf* qk_bufs[] = { bufs->conv_output, bufs->conv_output };
                size_t qk_offsets[] = { 0, LINEAR_TOTAL_KEY * sizeof(float) };
                size_t qk_ranges[] = {
                    LINEAR_TOTAL_KEY * sizeof(float),
                    LINEAR_TOTAL_KEY * sizeof(float)
                };
                vk_cmd_bind(cmd, pipes->rms_norm_qk, qk_bufs, qk_offsets, qk_ranges, 2,
                            &pc, sizeof(pc));
                vk_cmd_dispatch(cmd, LINEAR_NUM_K_HEADS, 1, 1);
            }
            vk_cmd_barrier(cmd);

            // L3: compute_decay_beta
            {
                VkBuf* db_bufs[] = { bufs->attn_proj5, bufs->attn_proj4,
                                     wf->buf, wf->buf,
                                     bufs->delta_g_decay, bufs->delta_beta };
                size_t db_offsets[] = { 0, 0, lc->A_log_off, lc->dt_bias_off, 0, 0 };
                size_t db_ranges[] = {
                    LINEAR_NUM_V_HEADS * sizeof(float),
                    LINEAR_NUM_V_HEADS * sizeof(float),
                    LINEAR_NUM_V_HEADS * sizeof(float),
                    LINEAR_NUM_V_HEADS * sizeof(uint16_t),
                    LINEAR_NUM_V_HEADS * sizeof(float),
                    LINEAR_NUM_V_HEADS * sizeof(float)
                };
                vk_cmd_bind(cmd, pipes->compute_decay_beta, db_bufs, db_offsets, db_ranges, 6,
                            NULL, 0);
                vk_cmd_dispatch(cmd, 1, 1, 1);
            }
            vk_cmd_barrier(cmd);

            // L4: gated_delta_net_step
            {
                uint32_t k_heads_per_v = LINEAR_NUM_V_HEADS / LINEAR_NUM_K_HEADS;
                VkBuf* dn_bufs[] = {
                    bufs->delta_state[linear_layer_idx],
                    bufs->conv_output, bufs->conv_output, bufs->conv_output,
                    bufs->delta_g_decay, bufs->delta_beta, bufs->delta_output
                };
                size_t dn_offsets[] = {
                    0,
                    0,
                    LINEAR_TOTAL_KEY * sizeof(float),
                    2 * LINEAR_TOTAL_KEY * sizeof(float),
                    0, 0, 0
                };
                size_t dn_ranges[] = {
                    (size_t)LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM * LINEAR_KEY_DIM * sizeof(float),
                    LINEAR_TOTAL_KEY * sizeof(float),
                    LINEAR_TOTAL_KEY * sizeof(float),
                    LINEAR_TOTAL_VALUE * sizeof(float),
                    LINEAR_NUM_V_HEADS * sizeof(float),
                    LINEAR_NUM_V_HEADS * sizeof(float),
                    LINEAR_TOTAL_VALUE * sizeof(float)
                };
                vk_cmd_bind(cmd, pipes->gated_delta_net_step, dn_bufs, dn_offsets, dn_ranges, 7,
                            &k_heads_per_v, sizeof(k_heads_per_v));
                vk_cmd_dispatch(cmd, LINEAR_NUM_V_HEADS, 1, 1);
            }
            vk_cmd_barrier(cmd);

            // L5: gated_rms_norm
            {
                struct { uint32_t value_dim; float eps; } pc = {
                    LINEAR_VALUE_DIM, 1e-6f
                };
                VkBuf* gn_bufs[] = { bufs->delta_output, bufs->attn_proj2,
                                     wf->buf, bufs->attn_out };
                size_t gn_offsets[] = { 0, 0, lc->gated_norm_w_off, 0 };
                size_t gn_ranges[] = {
                    LINEAR_TOTAL_VALUE * sizeof(float),
                    LINEAR_TOTAL_VALUE * sizeof(float),
                    LINEAR_VALUE_DIM * sizeof(uint16_t),
                    LINEAR_TOTAL_VALUE * sizeof(float)
                };
                vk_cmd_bind(cmd, pipes->gated_rms_norm, gn_bufs, gn_offsets, gn_ranges, 4,
                            &pc, sizeof(pc));
                vk_cmd_dispatch(cmd, LINEAR_NUM_V_HEADS, 1, 1);
            }
            vk_cmd_barrier(cmd);

            // Phase 4: O projection + residual add
            gpu_dequant_matvec(ctx, cmd, pipes->dequant_matvec,
                wf->buf, lc->out_proj_w_off, wf->buf, lc->out_proj_s_off, wf->buf, lc->out_proj_b_off,
                bufs->attn_out, bufs->o_proj_out,
                HIDDEN_DIM, LINEAR_TOTAL_VALUE, GROUP_SIZE);
            vk_cmd_barrier(cmd);

            gpu_residual_add(ctx, cmd, pipes->residual_add,
                bufs->o_proj_out, bufs->hidden, bufs->hidden, HIDDEN_DIM);
            vk_cmd_barrier(cmd);

            // Phase 5: Post-attention norm + routing + shared expert
            gpu_rms_norm(ctx, cmd, pipes->rms_norm,
                         bufs->hidden, wf->buf, lc->post_attn_norm_w_off, bufs->normed,
                         HIDDEN_DIM, RMS_NORM_EPS);
            vk_cmd_barrier(cmd);

            // Routing: [4096] -> [512]
            gpu_dequant_matvec(ctx, cmd, pipes->dequant_matvec,
                wf->buf, lc->gate_w_off, wf->buf, lc->gate_s_off, wf->buf, lc->gate_b_off,
                bufs->normed, bufs->routing_scores,
                NUM_EXPERTS, HIDDEN_DIM, GROUP_SIZE);
            vk_cmd_barrier(cmd);

            // Shared expert fused gate_up_swiglu: [4096] -> [1024]
            gpu_fused_gate_up_swiglu(ctx, cmd, pipes->fused_gate_up_swiglu,
                wf->buf, lc->sg_w_off, wf->buf, lc->sg_s_off, wf->buf, lc->sg_b_off,
                wf->buf, lc->su_w_off, wf->buf, lc->su_s_off, wf->buf, lc->su_b_off,
                bufs->normed, bufs->shared_act,
                SHARED_INTERMEDIATE, HIDDEN_DIM, GROUP_SIZE);
            vk_cmd_barrier(cmd);

            // Shared expert down_proj: [1024] -> [4096]
            gpu_dequant_matvec(ctx, cmd, pipes->dequant_matvec,
                wf->buf, lc->sd_w_off, wf->buf, lc->sd_s_off, wf->buf, lc->sd_b_off,
                bufs->shared_act, bufs->shared_out,
                HIDDEN_DIM, SHARED_INTERMEDIATE, GROUP_SIZE);
            vk_cmd_barrier(cmd);

            // Shared expert gate: [4096] -> [1]
            gpu_dequant_matvec(ctx, cmd, pipes->dequant_matvec,
                wf->buf, lc->seg_w_off, wf->buf, lc->seg_s_off, wf->buf, lc->seg_b_off,
                bufs->normed, bufs->shared_gate_score,
                1, HIDDEN_DIM, GROUP_SIZE);

            vk_cmd_submit(cmd);
            vk_cmd_destroy(ctx, cmd);
        }
        if (g_timing_enabled) {
            t1 = now_ms();
            double elapsed = t1 - t0;
            // Attribute time across merged phases for comparison
            g_timing.input_norm += elapsed * 0.05;   // ~5% of merged cmd
            g_timing.attn_proj  += elapsed * 0.45;   // ~45%
            g_timing.o_proj     += elapsed * 0.20;    // ~20%
            g_timing.post_norm  += elapsed * 0.30;    // ~30%
        }

        // Phase 3: no CPU attention for linear layers
        if (g_timing_enabled) { g_timing.cpu_attn += 0.0; }
    }

    // =================================================================
    // Phase 6: CPU routing (softmax + top-K) — both paths converge here
    // =================================================================
    if (g_timing_enabled) t0 = now_ms();

    // Save hidden as residual for MoE (hidden is on GPU, copy GPU-side)
    memcpy(vk_buf_map(bufs->residual), vk_buf_map(bufs->hidden), HIDDEN_DIM * sizeof(float));

    float* routing_scores = (float*)vk_buf_map(bufs->routing_scores);
    float scores_copy[NUM_EXPERTS];
    memcpy(scores_copy, routing_scores, NUM_EXPERTS * sizeof(float));
    cpu_softmax(scores_copy, NUM_EXPERTS);

    int expert_indices[K_EXPERTS];
    float expert_weights[K_EXPERTS];
    cpu_topk(scores_copy, NUM_EXPERTS, K_EXPERTS, expert_indices, expert_weights);

    if (g_timing_enabled) { t1 = now_ms(); g_timing.cpu_routing += t1 - t0; }

    // =================================================================
    // Phase 7: Expert I/O — via DRAM cache + CAR + io_uring fallback
    // =================================================================
    if (g_timing_enabled) t0 = now_ms();

    if (packed_fd >= 0) {
        void* expert_dests[K_EXPERTS];
        for (int k = 0; k < K_EXPERTS; k++)
            expert_dests[k] = vk_buf_map(bufs->expert_data[k]);

        if (ecache) {
            expert_cache_route_and_load(ecache, layer_idx, packed_fd,
                scores_copy, expert_indices, expert_weights,
                expert_dests, EXPERT_SIZE, ring);
        } else {
            // Fallback: direct io_uring (no cache)
            uint64_t io_offsets[K_EXPERTS];
            size_t   io_sizes[K_EXPERTS];
            for (int k = 0; k < K_EXPERTS; k++) {
                io_offsets[k] = (uint64_t)expert_indices[k] * EXPERT_SIZE;
                io_sizes[k]   = EXPERT_SIZE;
            }
            int ret = io_ring_read_experts(ring, packed_fd, io_offsets, expert_dests, io_sizes, K_EXPERTS);
            if (ret != 0) {
                fprintf(stderr, "WARNING: layer %d expert io_ring read failed\n", layer_idx);
            }
        }
    }

    if (g_timing_enabled) { t1 = now_ms(); g_timing.expert_io += t1 - t0; }

    // =================================================================
    // Phase 8+9: Expert compute + MoE combine — DEFERRED (no wait)
    // GPU executes this while we start the next layer's CPU work.
    // =================================================================
    if (g_timing_enabled) t0 = now_ms();

    {
        // Write moe_params: [w0, w1, w2, w3, shared_gate_score_raw]
        float* params = (float*)vk_buf_map(bufs->moe_params);
        for (int k = 0; k < K_EXPERTS; k++) params[k] = expert_weights[k];
        float shared_gate_raw = ((float*)vk_buf_map(bufs->shared_gate_score))[0];
        params[4] = shared_gate_raw;

        VkCmd* cmd = vk_cmd_begin(ctx);

        // Expert forward passes
        if (packed_fd >= 0) {
            for (int k = 0; k < K_EXPERTS; k++) {
                gpu_fused_gate_up_swiglu(ctx, cmd, pipes->fused_gate_up_swiglu,
                    bufs->expert_data[k], GATE_W_OFF,
                    bufs->expert_data[k], GATE_S_OFF,
                    bufs->expert_data[k], GATE_B_OFF,
                    bufs->expert_data[k], UP_W_OFF,
                    bufs->expert_data[k], UP_S_OFF,
                    bufs->expert_data[k], UP_B_OFF,
                    bufs->normed, bufs->expert_act[k],
                    MOE_INTERMEDIATE, HIDDEN_DIM, GROUP_SIZE);
                vk_cmd_barrier(cmd);

                gpu_dequant_matvec(ctx, cmd, pipes->dequant_matvec,
                    bufs->expert_data[k], DOWN_W_OFF,
                    bufs->expert_data[k], DOWN_S_OFF,
                    bufs->expert_data[k], DOWN_B_OFF,
                    bufs->expert_act[k], bufs->expert_out[k],
                    HIDDEN_DIM, MOE_INTERMEDIATE, GROUP_SIZE);
                vk_cmd_barrier(cmd);
            }
        }

        // MoE combine: residual + shared_out + expert_outs -> hidden
        gpu_moe_combine(ctx, cmd, pipes->moe_combine,
            bufs->residual, bufs->shared_out, bufs->hidden,
            bufs->expert_out, bufs->moe_params,
            HIDDEN_DIM, K_EXPERTS);

        // Submit WITHOUT waiting — deferred execution.
        // The GPU will execute this while we do CPU routing + I/O for the next layer.
        // Vulkan single-queue guarantees the next layer's CMD_A runs AFTER this completes.
        vk_cmd_submit_no_wait(cmd);

        // Store for deferred cleanup at next layer start
        g_deferred.cmd = cmd;
        g_deferred.ctx = ctx;
    }

    if (g_timing_enabled) {
        t1 = now_ms();
        g_timing.expert_compute += t1 - t0;
        // moe_combine is merged into expert_compute timing (deferred)
        g_timing.moe_combine += 0.0;
        g_timing.count++;
    } else {
        g_timing.count++;
    }

    // Do NOT read back hidden — it stays on GPU in bufs->hidden.
    // The next layer reads it directly. Only read back at end of all layers.

    if (g_debug && layer_idx < 2) {
        // For debug, force a wait to read hidden
        deferred_wait();
        float* h = (float*)vk_buf_map(bufs->hidden);
        fprintf(stderr, "[DBG] L%d FINAL hidden: rms=%.6f first3=[%.6f,%.6f,%.6f]\n",
                layer_idx, vec_rms(h, HIDDEN_DIM), h[0], h[1], h[2]);
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    // ---- CLI parsing ----
    const char* prompt = "Hello";
    int max_tokens = 100;
    float temperature = 0.0f;
    float top_p = 0.9f;
    const char* model_dir = ".";  // Directory containing model_weights.bin/json and packed_experts/

    // Expert cache / CAR options
    float car_threshold = 0.35f;   // CAR threshold (0=always substitute, 1=disabled)
    int   car_dampen = 1;          // dampening enabled by default
    int   warmup_tokens = 0;       // force full fidelity for first N tokens
    const char* freq_profile_path = NULL;   // load frequency profile
    const char* profile_out_path = NULL;    // save frequency profile
    int   no_cache = 0;            // disable expert cache entirely

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--timing") == 0) {
            g_timing_enabled = 1;
        } else if (strcmp(argv[i], "--debug") == 0) {
            g_debug = 1;
        } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            temperature = atof(argv[++i]);
        } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            top_p = atof(argv[++i]);
        } else if (strcmp(argv[i], "--model-dir") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (strcmp(argv[i], "--car-threshold") == 0 && i + 1 < argc) {
            car_threshold = atof(argv[++i]);
        } else if (strcmp(argv[i], "--car-dampen") == 0) {
            car_dampen = 1;
        } else if (strcmp(argv[i], "--no-car-dampen") == 0) {
            car_dampen = 0;
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--freq-profile") == 0 && i + 1 < argc) {
            freq_profile_path = argv[++i];
        } else if (strcmp(argv[i], "--profile-out") == 0 && i + 1 < argc) {
            profile_out_path = argv[++i];
        } else if (strcmp(argv[i], "--no-cache") == 0) {
            no_cache = 1;
        } else {
            fprintf(stderr, "Usage: %s --prompt \"text\" --tokens N [--timing] "
                    "[--temperature T] [--top-p P] [--model-dir DIR]\n"
                    "  Expert cache: [--car-threshold F] [--warmup N]\n"
                    "                [--car-dampen] [--no-car-dampen] [--no-cache]\n"
                    "  Profiling:    [--freq-profile FILE] [--profile-out FILE]\n", argv[0]);
            return 1;
        }
    }

    fprintf(stderr, "[config] prompt=\"%.60s%s\" tokens=%d temp=%.2f top_p=%.2f timing=%d\n",
            prompt, strlen(prompt) > 60 ? "..." : "", max_tokens, temperature, top_p,
            g_timing_enabled);
    if (!no_cache) {
        fprintf(stderr, "[config] CAR threshold=%.2f dampen=%d warmup=%d\n",
                car_threshold, car_dampen, warmup_tokens);
    }

    double t0_total = now_ms();

    // ---- Initialize Vulkan ----
    fprintf(stderr, "[init] Creating Vulkan context...\n");
    VkCtx* ctx = vk_create();
    if (!ctx) {
        fprintf(stderr, "FATAL: vk_create() failed\n");
        return 1;
    }

    // ---- Load weights ----
    char bin_path[512], json_path[512];
    snprintf(bin_path, sizeof(bin_path), "%s/model_weights.bin", model_dir);
    snprintf(json_path, sizeof(json_path), "%s/model_weights.json", model_dir);

    fprintf(stderr, "[init] Loading weights from %s...\n", bin_path);
    WeightFile* wf = weights_load(ctx, bin_path, json_path);
    if (!wf) {
        fprintf(stderr, "FATAL: weights_load() failed\n");
        vk_destroy(ctx);
        return 1;
    }
    fprintf(stderr, "[init] Weights loaded: %zu bytes, %d tensors\n",
            wf->total_size, wf->num_tensors);

    // ---- Build layer weight cache ----
    build_layer_cache(wf);

    // ---- Load tokenizer ----
    char tok_path[512];
    snprintf(tok_path, sizeof(tok_path), "%s/tokenizer.bin", model_dir);

    bpe_tokenizer tok;
    if (bpe_load(&tok, tok_path) != 0) {
        fprintf(stderr, "FATAL: bpe_load() failed\n");
        weights_destroy(ctx, wf);
        vk_destroy(ctx);
        return 1;
    }

    // ---- Create io_uring ----
    IoRing* ring = io_ring_create(16);
    if (!ring) {
        fprintf(stderr, "FATAL: io_ring_create() failed\n");
        // vocab is part of bpe_tokenizer, freed by bpe_free
        bpe_free(&tok);
        weights_destroy(ctx, wf);
        vk_destroy(ctx);
        return 1;
    }

    // ---- Open expert layer files ----
    int layer_fds[NUM_LAYERS];
    int expert_layers_available = 0;
    for (int i = 0; i < NUM_LAYERS; i++) {
        char path[512];
        snprintf(path, sizeof(path), "%s/packed_experts/layer_%02d.bin", model_dir, i);
        layer_fds[i] = open(path, O_RDONLY);
        if (layer_fds[i] >= 0) expert_layers_available++;
    }
    fprintf(stderr, "[init] Expert layers: %d/%d available\n", expert_layers_available, NUM_LAYERS);

    // ---- Create expert cache (mmap + mincore for CAR routing) ----
    ExpertCache* ecache = NULL;
    if (!no_cache && expert_layers_available > 0) {
        ecache = expert_cache_create(NUM_LAYERS, EXPERT_SIZE, layer_fds);
        if (ecache) {
            expert_cache_set_car(ecache, car_threshold, car_dampen, warmup_tokens);
            if (freq_profile_path) {
                if (expert_cache_load_profile(ecache, freq_profile_path) == 0) {
                    expert_cache_warmup_from_profile(ecache);
                }
            }
        } else {
            fprintf(stderr, "WARNING: expert cache creation failed, running without cache\n");
        }
    }

    // ---- Create pipelines ----
    Pipelines pipes;
    char shader_path[512];

    snprintf(shader_path, sizeof(shader_path), "%s/shaders/dequant_matvec_4bit.spv", model_dir);
    pipes.dequant_matvec = vk_pipe_create(ctx, shader_path, 12, 5); // 3x uint32 push, 5 bindings

    snprintf(shader_path, sizeof(shader_path), "%s/shaders/rms_norm.spv", model_dir);
    pipes.rms_norm = vk_pipe_create(ctx, shader_path, 8, 3); // uint32+float push, 3 bindings

    snprintf(shader_path, sizeof(shader_path), "%s/shaders/residual_add.spv", model_dir);
    pipes.residual_add = vk_pipe_create(ctx, shader_path, 4, 3); // uint32 push, 3 bindings

    snprintf(shader_path, sizeof(shader_path), "%s/shaders/fused_gate_up_swiglu.spv", model_dir);
    pipes.fused_gate_up_swiglu = vk_pipe_create(ctx, shader_path, 12, 8); // 3x uint32, 8 bindings

    snprintf(shader_path, sizeof(shader_path), "%s/shaders/moe_combine.spv", model_dir);
    pipes.moe_combine = vk_pipe_create(ctx, shader_path, 8, 8); // 2x uint32, 8 bindings

    snprintf(shader_path, sizeof(shader_path), "%s/shaders/sigmoid_gate.spv", model_dir);
    pipes.sigmoid_gate = vk_pipe_create(ctx, shader_path, 4, 2); // uint32, 2 bindings

    snprintf(shader_path, sizeof(shader_path), "%s/shaders/attn_scores.spv", model_dir);
    pipes.attn_scores = vk_pipe_create(ctx, shader_path, 28, 3); // 7x uint/float push, 3 bindings

    snprintf(shader_path, sizeof(shader_path), "%s/shaders/attn_softmax.spv", model_dir);
    pipes.attn_softmax = vk_pipe_create(ctx, shader_path, 8, 1); // 2x uint32, 1 binding

    snprintf(shader_path, sizeof(shader_path), "%s/shaders/attn_values.spv", model_dir);
    pipes.attn_values = vk_pipe_create(ctx, shader_path, 20, 3); // 5x uint32, 3 bindings

    // GPU linear attention (GatedDeltaNet) pipelines
    snprintf(shader_path, sizeof(shader_path), "%s/shaders/conv1d_step.spv", model_dir);
    pipes.conv1d_step = vk_pipe_create(ctx, shader_path, 4, 4); // 1x uint32, 4 bindings

    snprintf(shader_path, sizeof(shader_path), "%s/shaders/rms_norm_qk.spv", model_dir);
    pipes.rms_norm_qk = vk_pipe_create(ctx, shader_path, 8, 2); // uint32+float, 2 bindings

    snprintf(shader_path, sizeof(shader_path), "%s/shaders/compute_decay_beta.spv", model_dir);
    pipes.compute_decay_beta = vk_pipe_create(ctx, shader_path, 0, 6); // no push, 6 bindings

    snprintf(shader_path, sizeof(shader_path), "%s/shaders/gated_delta_net_step.spv", model_dir);
    pipes.gated_delta_net_step = vk_pipe_create(ctx, shader_path, 4, 7); // 1x uint32, 7 bindings

    snprintf(shader_path, sizeof(shader_path), "%s/shaders/gated_rms_norm.spv", model_dir);
    pipes.gated_rms_norm = vk_pipe_create(ctx, shader_path, 8, 4); // uint32+float, 4 bindings

    fprintf(stderr, "[init] Pipelines created\n");

    // ---- Allocate scratch buffers ----
    Buffers buf;
    memset(&buf, 0, sizeof(buf));

    buf.hidden   = vk_buf_create(ctx, HIDDEN_DIM * sizeof(float));
    buf.normed   = vk_buf_create(ctx, HIDDEN_DIM * sizeof(float));
    buf.residual = vk_buf_create(ctx, HIDDEN_DIM * sizeof(float));

    // Attention projection outputs — sized for the larger of full/linear
    buf.attn_proj  = vk_buf_create(ctx, FA_Q_PROJ_DIM * sizeof(float));  // 16384 >= 12288
    buf.attn_proj2 = vk_buf_create(ctx, LINEAR_TOTAL_VALUE * sizeof(float)); // 8192 >= 512
    buf.attn_proj3 = vk_buf_create(ctx, FA_KV_DIM * sizeof(float));     // 512
    buf.attn_proj4 = vk_buf_create(ctx, LINEAR_NUM_V_HEADS * sizeof(float)); // 64
    buf.attn_proj5 = vk_buf_create(ctx, LINEAR_NUM_V_HEADS * sizeof(float)); // 64

    buf.attn_out   = vk_buf_create(ctx, FA_O_PROJ_IN_DIM * sizeof(float)); // 8192
    buf.o_proj_out = vk_buf_create(ctx, HIDDEN_DIM * sizeof(float));

    buf.routing_scores = vk_buf_create(ctx, NUM_EXPERTS * sizeof(float));

    buf.shared_act = vk_buf_create(ctx, SHARED_INTERMEDIATE * sizeof(float));
    buf.shared_out = vk_buf_create(ctx, HIDDEN_DIM * sizeof(float));
    buf.shared_gate_score = vk_buf_create(ctx, sizeof(float));

    for (int k = 0; k < K_EXPERTS; k++) {
        buf.expert_data[k] = vk_buf_create(ctx, EXPERT_SIZE);
        buf.expert_act[k]  = vk_buf_create(ctx, MOE_INTERMEDIATE * sizeof(float));
        buf.expert_out[k]  = vk_buf_create(ctx, HIDDEN_DIM * sizeof(float));
    }

    buf.moe_params = vk_buf_create(ctx, 5 * sizeof(float));
    buf.logits     = vk_buf_create(ctx, VOCAB_SIZE * sizeof(float));

    // Full attention KV caches on GPU (for GPU-accelerated attention)
    buf.attn_scores_buf = vk_buf_create(ctx, FA_NUM_ATTN_HEADS * FA_MAX_SEQ_LEN * sizeof(float));
    for (int i = 0; i < NUM_FULL_ATTN_LAYERS; i++) {
        buf.kv_k[i] = vk_buf_create(ctx, FA_MAX_SEQ_LEN * FA_KV_DIM * sizeof(float));
        buf.kv_v[i] = vk_buf_create(ctx, FA_MAX_SEQ_LEN * FA_KV_DIM * sizeof(float));
    }

    // GPU linear attention state
    for (int i = 0; i < NUM_LINEAR_LAYERS; i++) {
        buf.conv_state[i] = vk_buf_create(ctx, (CONV_KERNEL_SIZE-1) * LINEAR_CONV_DIM * sizeof(float));
        memset(vk_buf_map(buf.conv_state[i]), 0, (CONV_KERNEL_SIZE-1) * LINEAR_CONV_DIM * sizeof(float));
        buf.delta_state[i] = vk_buf_create(ctx, LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM * LINEAR_KEY_DIM * sizeof(float));
        memset(vk_buf_map(buf.delta_state[i]), 0, LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM * LINEAR_KEY_DIM * sizeof(float));
    }
    buf.conv_output = vk_buf_create(ctx, LINEAR_CONV_DIM * sizeof(float));
    buf.delta_g_decay = vk_buf_create(ctx, LINEAR_NUM_V_HEADS * sizeof(float));
    buf.delta_beta = vk_buf_create(ctx, LINEAR_NUM_V_HEADS * sizeof(float));
    buf.delta_output = vk_buf_create(ctx, LINEAR_TOTAL_VALUE * sizeof(float));

    fprintf(stderr, "[init] Scratch buffers allocated\n");

    // ---- Allocate per-layer attention state ----
    LinearAttnState* la_states[NUM_LAYERS];
    KVCache* kv_caches[NUM_LAYERS];
    memset(la_states, 0, sizeof(la_states));
    memset(kv_caches, 0, sizeof(kv_caches));

    for (int i = 0; i < NUM_LAYERS; i++) {
        int is_full = ((i + 1) % FULL_ATTN_INTERVAL == 0);
        if (is_full) {
            kv_caches[i] = kv_cache_create();
        } else {
            la_states[i] = linear_attn_state_create();
        }
    }

    fprintf(stderr, "[init] Attention state allocated (%d KVCache + %d LinearAttn)\n",
            NUM_FULL_ATTN_LAYERS, NUM_LINEAR_LAYERS);

    double t_init = now_ms();
    fprintf(stderr, "[init] Setup: %.1f ms\n\n", t_init - t0_total);

    // ---- Working buffers (declared early for cleanup label) ----
    float* hidden = NULL;
    float* logits = NULL;

    // ---- Tokenize prompt ----
    uint32_t prompt_ids[8192];
    int num_prompt = bpe_encode(&tok, prompt, prompt_ids, 8192);
    fprintf(stderr, "[prompt] \"%s\" -> %d tokens\n", prompt, num_prompt);

    if (num_prompt == 0) {
        fprintf(stderr, "ERROR: empty prompt after tokenization\n");
        goto cleanup;
    }

    hidden = calloc(HIDDEN_DIM, sizeof(float));
    logits = calloc(VOCAB_SIZE, sizeof(float));

    // Final norm weight
    TensorInfo* final_norm_ti = weights_get_tensor(wf, "model.norm.weight");
    size_t final_norm_off = final_norm_ti ? final_norm_ti->offset : 0;

    // LM head weight offsets
    size_t lm_w_off = get_tensor_offset(wf, "lm_head.weight");
    size_t lm_s_off = get_tensor_offset(wf, "lm_head.scales");
    size_t lm_b_off = get_tensor_offset(wf, "lm_head.biases");

    // ---- Prefill: process prompt tokens ----
    fprintf(stderr, "--- Prefilling %d prompt tokens ---\n", num_prompt);
    int pos = 0;

    double t_prefill_start = now_ms();

    for (int t = 0; t < num_prompt; t++) {
        embed_lookup(wf, prompt_ids[t], hidden);
        if (t == 0) dbg_vec("embed_lookup", hidden, HIDDEN_DIM);

        for (int layer = 0; layer < NUM_LAYERS; layer++) {
            int is_full = ((layer + 1) % FULL_ATTN_INTERVAL == 0);
            fused_layer_forward(ctx, wf, &pipes, &buf, layer, hidden,
                                is_full ? kv_caches[layer] : NULL,
                                is_full ? NULL : la_states[layer],
                                pos, layer_fds[layer], ring, ecache);
        }
        // Wait for last layer's deferred expert compute
        deferred_wait();
        // Read hidden back from GPU (moe_combine wrote final result to bufs->hidden)
        memcpy(hidden, vk_buf_map(buf.hidden), HIDDEN_DIM * sizeof(float));
        // Reset descriptor pool between tokens
        vk_descriptor_pool_reset(ctx);
        pos++;

        if (ecache) expert_cache_new_token(ecache);

        if (t == 0) {
            fprintf(stderr, "  [prefill] first token: %.0f ms\n", now_ms() - t_prefill_start);
        }
    }

    double t_prefill_end = now_ms();
    fprintf(stderr, "[prefill] %d tokens: %.0f ms (%.1f ms/token)\n",
            num_prompt, t_prefill_end - t_prefill_start,
            (t_prefill_end - t_prefill_start) / num_prompt);

    // ---- Final norm + LM head for last prefill token ----
    // buf.hidden already has the correct hidden state from deferred_wait above
    {
        VkCmd* cmd = vk_cmd_begin(ctx);
        gpu_rms_norm(ctx, cmd, pipes.rms_norm,
                     buf.hidden, wf->buf, final_norm_off, buf.normed,
                     HIDDEN_DIM, RMS_NORM_EPS);
        vk_cmd_barrier(cmd);
        gpu_dequant_matvec(ctx, cmd, pipes.dequant_matvec,
            wf->buf, lm_w_off, wf->buf, lm_s_off, wf->buf, lm_b_off,
            buf.normed, buf.logits,
            VOCAB_SIZE, HIDDEN_DIM, GROUP_SIZE);
        vk_cmd_submit(cmd);
        vk_cmd_destroy(ctx, cmd);
        memcpy(hidden, vk_buf_map(buf.normed), HIDDEN_DIM * sizeof(float));
        memcpy(logits, vk_buf_map(buf.logits), VOCAB_SIZE * sizeof(float));
    }

    if (g_debug) {
        // CPU reference lm_head for first 5 outputs
        uint32_t* lm_W = (uint32_t*)((char*)wf->mapped + lm_w_off);
        uint16_t* lm_S = (uint16_t*)((char*)wf->mapped + lm_s_off);
        uint16_t* lm_B = (uint16_t*)((char*)wf->mapped + lm_b_off);
        int lm_packed = HIDDEN_DIM / 8;
        int lm_ngroups = HIDDEN_DIM / GROUP_SIZE;
        int lm_ppg = GROUP_SIZE / 8;
        for (int row = 0; row < 3; row++) {
            float acc = 0;
            for (int g2 = 0; g2 < lm_ngroups; g2++) {
                float sc = bf16_to_f32(lm_S[row*lm_ngroups+g2]);
                float bi = bf16_to_f32(lm_B[row*lm_ngroups+g2]);
                for (int p2 = 0; p2 < lm_ppg; p2++) {
                    uint32_t pk = lm_W[row*lm_packed + g2*lm_ppg + p2];
                    for (int nn = 0; nn < 8; nn++) {
                        uint32_t nib = (pk >> (nn*4)) & 0xF;
                        acc += ((float)nib * sc + bi) * hidden[g2*GROUP_SIZE + p2*8 + nn];
                    }
                }
            }
            fprintf(stderr, "[DBG] CPU lm_head[%d]=%.4f GPU=%.4f\n", row, acc, logits[row]);
        }

        // Show top-5 logits
        fprintf(stderr, "[DBG] logits: rms=%.4f max=%.4f min=%.4f\n",
                vec_rms(logits, VOCAB_SIZE),
                logits[cpu_argmax(logits, VOCAB_SIZE)],
                logits[0]); // just first for reference
        // Top 5 tokens
        for (int rank = 0; rank < 5; rank++) {
            float best = -FLT_MAX;
            int best_idx = 0;
            for (int i = 0; i < VOCAB_SIZE; i++) {
                int skip = 0;
                for (int r2 = 0; r2 < rank; r2++) { /* crude but fine for debug */ }
                if (!skip && logits[i] > best) { best = logits[i]; best_idx = i; }
            }
            fprintf(stderr, "[DBG] top-%d: token=%d score=%.4f '%s'\n",
                    rank+1, best_idx, best, decode_token(&tok, best_idx));
            logits[best_idx] = -FLT_MAX; // mark used for next iteration
        }
        // Restore logits (re-read from GPU)
        memcpy(logits, vk_buf_map(buf.logits), VOCAB_SIZE * sizeof(float));
    }

    // ---- Sample first token ----
    int next_token;
    if (temperature == 0.0f) {
        next_token = cpu_argmax(logits, VOCAB_SIZE);
    } else {
        next_token = sample_top_p(logits, VOCAB_SIZE, temperature, top_p);
    }

    double ttft_ms = now_ms() - t0_total;
    fprintf(stderr, "[ttft] %.0f ms (prefill %d tokens)\n", ttft_ms, num_prompt);

    // Print first token
    printf("%s", decode_token(&tok, next_token));
    fflush(stdout);

    // ---- Autoregressive generation loop ----
    int total_generated = 1;
    if (g_timing_enabled) timing_reset();

    for (int gen = 1; gen < max_tokens; gen++) {
        double t_gen_start = now_ms();

        // Check EOS
        if (next_token == EOS_TOKEN_1 || next_token == EOS_TOKEN_2) {
            fprintf(stderr, "\n[eos] Token %d at position %d\n", next_token, gen);
            break;
        }

        // Embed
        embed_lookup(wf, next_token, hidden);

        // Run 60 layers
        for (int layer = 0; layer < NUM_LAYERS; layer++) {
            int is_full = ((layer + 1) % FULL_ATTN_INTERVAL == 0);
            fused_layer_forward(ctx, wf, &pipes, &buf, layer, hidden,
                                is_full ? kv_caches[layer] : NULL,
                                is_full ? NULL : la_states[layer],
                                pos, layer_fds[layer], ring, ecache);
        }
        // Wait for last layer's deferred expert compute
        deferred_wait();
        // Read hidden back from GPU
        memcpy(hidden, vk_buf_map(buf.hidden), HIDDEN_DIM * sizeof(float));
        vk_descriptor_pool_reset(ctx);
        pos++;
        if (ecache) expert_cache_new_token(ecache);

        // Final norm + LM head (merged into one command buffer)
        {
            VkCmd* cmd = vk_cmd_begin(ctx);
            gpu_rms_norm(ctx, cmd, pipes.rms_norm,
                         buf.hidden, wf->buf, final_norm_off, buf.normed,
                         HIDDEN_DIM, RMS_NORM_EPS);
            vk_cmd_barrier(cmd);
            gpu_dequant_matvec(ctx, cmd, pipes.dequant_matvec,
                wf->buf, lm_w_off, wf->buf, lm_s_off, wf->buf, lm_b_off,
                buf.normed, buf.logits,
                VOCAB_SIZE, HIDDEN_DIM, GROUP_SIZE);
            vk_cmd_submit(cmd);
            vk_cmd_destroy(ctx, cmd);
            memcpy(logits, vk_buf_map(buf.logits), VOCAB_SIZE * sizeof(float));
        }

        // Sample
        if (temperature == 0.0f) {
            next_token = cpu_argmax(logits, VOCAB_SIZE);
        } else {
            next_token = sample_top_p(logits, VOCAB_SIZE, temperature, top_p);
        }
        total_generated++;

        // Print decoded token
        printf("%s", decode_token(&tok, next_token));
        fflush(stdout);

        double t_gen_end = now_ms();
        double tok_time = t_gen_end - t_gen_start;
        fprintf(stderr, "  [gen %d/%d] token_id=%d (%.0f ms, %.2f tok/s)\n",
                gen, max_tokens, next_token, tok_time, 1000.0 / tok_time);
    }

    if (g_timing_enabled) timing_print();

    // ---- Statistics ----
    printf("\n\n--- Statistics ---\n");
    double total_time = now_ms() - t0_total;
    printf("Total time:     %.1f s\n", total_time / 1000.0);
    printf("TTFT:           %.0f ms\n", ttft_ms);
    printf("Tokens:         %d generated\n", total_generated);
    if (total_generated > 1) {
        double gen_time = total_time - ttft_ms;
        printf("Generation:     %.1f s (%.2f tok/s)\n",
               gen_time / 1000.0, (total_generated - 1) * 1000.0 / gen_time);
    }
    printf("Config:         K=%d experts, %d layers\n", K_EXPERTS, NUM_LAYERS);

    // Expert cache stats and profile save
    if (ecache) {
        expert_cache_print_stats(ecache);
        if (profile_out_path) {
            expert_cache_save_profile(ecache, profile_out_path);
        }
    }

    // ---- Cleanup ----
cleanup:
    free(hidden);
    free(logits);

    // Destroy expert cache before closing layer_fds (cache uses them for backfill)
    if (ecache) expert_cache_destroy(ecache);

    for (int i = 0; i < NUM_LAYERS; i++) {
        if (kv_caches[i]) kv_cache_destroy(kv_caches[i]);
        if (la_states[i]) linear_attn_state_destroy(la_states[i]);
        if (layer_fds[i] >= 0) close(layer_fds[i]);
    }

    // Destroy GPU buffers
    vk_buf_destroy(ctx, buf.hidden);
    vk_buf_destroy(ctx, buf.normed);
    vk_buf_destroy(ctx, buf.residual);
    vk_buf_destroy(ctx, buf.attn_proj);
    vk_buf_destroy(ctx, buf.attn_proj2);
    vk_buf_destroy(ctx, buf.attn_proj3);
    vk_buf_destroy(ctx, buf.attn_proj4);
    vk_buf_destroy(ctx, buf.attn_proj5);
    vk_buf_destroy(ctx, buf.attn_out);
    vk_buf_destroy(ctx, buf.o_proj_out);
    vk_buf_destroy(ctx, buf.routing_scores);
    vk_buf_destroy(ctx, buf.shared_act);
    vk_buf_destroy(ctx, buf.shared_out);
    vk_buf_destroy(ctx, buf.shared_gate_score);
    for (int k = 0; k < K_EXPERTS; k++) {
        vk_buf_destroy(ctx, buf.expert_data[k]);
        vk_buf_destroy(ctx, buf.expert_act[k]);
        vk_buf_destroy(ctx, buf.expert_out[k]);
    }
    vk_buf_destroy(ctx, buf.moe_params);
    vk_buf_destroy(ctx, buf.logits);
    vk_buf_destroy(ctx, buf.attn_scores_buf);
    for (int i = 0; i < NUM_FULL_ATTN_LAYERS; i++) {
        vk_buf_destroy(ctx, buf.kv_k[i]);
        vk_buf_destroy(ctx, buf.kv_v[i]);
    }
    for (int i = 0; i < NUM_LINEAR_LAYERS; i++) {
        vk_buf_destroy(ctx, buf.conv_state[i]);
        vk_buf_destroy(ctx, buf.delta_state[i]);
    }
    vk_buf_destroy(ctx, buf.conv_output);
    vk_buf_destroy(ctx, buf.delta_g_decay);
    vk_buf_destroy(ctx, buf.delta_beta);
    vk_buf_destroy(ctx, buf.delta_output);

    // Destroy pipelines
    vk_pipe_destroy(ctx, pipes.dequant_matvec);
    vk_pipe_destroy(ctx, pipes.rms_norm);
    vk_pipe_destroy(ctx, pipes.residual_add);
    vk_pipe_destroy(ctx, pipes.fused_gate_up_swiglu);
    vk_pipe_destroy(ctx, pipes.moe_combine);
    vk_pipe_destroy(ctx, pipes.sigmoid_gate);
    vk_pipe_destroy(ctx, pipes.attn_scores);
    vk_pipe_destroy(ctx, pipes.attn_softmax);
    vk_pipe_destroy(ctx, pipes.attn_values);
    vk_pipe_destroy(ctx, pipes.conv1d_step);
    vk_pipe_destroy(ctx, pipes.rms_norm_qk);
    vk_pipe_destroy(ctx, pipes.compute_decay_beta);
    vk_pipe_destroy(ctx, pipes.gated_delta_net_step);
    vk_pipe_destroy(ctx, pipes.gated_rms_norm);

    io_ring_destroy(ring);
    // vocab is part of bpe_tokenizer, freed by bpe_free
    bpe_free(&tok);
    weights_destroy(ctx, wf);
    vk_destroy(ctx);

    return 0;
}
