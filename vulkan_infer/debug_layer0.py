#!/usr/bin/env python3
"""Reference computation for layer 0 linear attention (GatedDeltaNet).

Loads model weights from the binary file and JSON manifest,
computes each step of the linear attention for token "Hello" (id=9419),
and prints intermediate values for comparison against the C/Vulkan implementation.
"""

import json
import mmap
import numpy as np
import struct

# ---- Constants ----
HIDDEN_DIM = 4096
GROUP_SIZE = 64
LINEAR_NUM_V_HEADS = 64
LINEAR_NUM_K_HEADS = 16
LINEAR_KEY_DIM = 128
LINEAR_VALUE_DIM = 128
LINEAR_TOTAL_KEY = 2048   # NUM_K_HEADS * KEY_DIM
LINEAR_TOTAL_VALUE = 8192 # NUM_V_HEADS * VALUE_DIM
LINEAR_CONV_DIM = 12288   # TOTAL_KEY + TOTAL_KEY + TOTAL_VALUE
CONV_KERNEL_SIZE = 4
RMS_NORM_EPS = 1e-6
TOKEN_ID = 9419

WEIGHTS_BIN = "/home/flux/flash-moe/metal_infer/model_weights.bin"
WEIGHTS_JSON = "/home/flux/flash-moe/metal_infer/model_weights.json"


def bf16_to_f32(raw_u16):
    """Convert bf16 (stored as uint16) to float32."""
    return np.frombuffer((raw_u16.astype(np.uint32) << 16).tobytes(), dtype=np.float32)


def vec_rms(x):
    return np.sqrt(np.mean(x * x))


def fmt5(arr):
    return "[" + ", ".join(f"{v:.6f}" for v in arr[:5]) + "]"


def print_step(name, arr):
    print(f"[step] {name}: rms={vec_rms(arr):.6f} first5={fmt5(arr)}")


class WeightFile:
    def __init__(self, bin_path, json_path):
        with open(json_path, "r") as f:
            manifest = json.load(f)
        self.tensors = manifest["tensors"]

        self.f = open(bin_path, "rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)

    def close(self):
        self.mm.close()
        self.f.close()

    def get(self, name):
        """Return tensor as numpy array."""
        info = self.tensors[name]
        offset = info["offset"]
        size = info["size"]
        shape = info["shape"]
        dtype_str = info["dtype"]

        raw = self.mm[offset:offset + size]

        if dtype_str == "U32":
            arr = np.frombuffer(raw, dtype=np.uint32)
        elif dtype_str in ("BF16", "uint16"):
            arr = np.frombuffer(raw, dtype=np.uint16)
        elif dtype_str in ("F32", "float32"):
            arr = np.frombuffer(raw, dtype=np.float32)
        else:
            raise ValueError(f"Unknown dtype: {dtype_str}")

        return arr.reshape(shape) if len(shape) > 1 else arr.copy()


def dequant_matvec(W_packed, scales_bf16, biases_bf16, x, out_dim, in_dim, group_size):
    """4-bit dequantized matrix-vector multiply. Matches C cpu_dequant_matvec exactly."""
    num_groups = in_dim // group_size
    packed_per_group = group_size // 8
    packed_cols = in_dim // 8

    out = np.zeros(out_dim, dtype=np.float32)

    # Convert all scales and biases to float32
    scales_f32 = bf16_to_f32(scales_bf16.ravel()).reshape(scales_bf16.shape)
    biases_f32 = bf16_to_f32(biases_bf16.ravel()).reshape(biases_bf16.shape)

    for row in range(out_dim):
        acc = 0.0
        w_row = W_packed[row]  # [packed_cols]
        s_row = scales_f32[row]  # [num_groups]
        b_row = biases_f32[row]  # [num_groups]

        for g in range(num_groups):
            scale = s_row[g]
            bias = b_row[g]
            for p in range(packed_per_group):
                packed = int(w_row[g * packed_per_group + p])
                x_base = g * group_size + p * 8
                for n in range(8):
                    nibble = (packed >> (n * 4)) & 0xF
                    acc += (float(nibble) * scale + bias) * x[x_base + n]

        out[row] = acc

    return out


def dequant_embed_row(W_packed, scales_bf16, biases_bf16, token_id, hidden_dim):
    """Dequantize a single embedding row."""
    packed_cols = W_packed.shape[1]  # 512
    num_groups = scales_bf16.shape[1]  # 64
    group_size = hidden_dim // num_groups  # 64
    packed_per_group = group_size // 8  # 8

    w_row = W_packed[token_id]
    s_row = bf16_to_f32(scales_bf16[token_id])
    b_row = bf16_to_f32(biases_bf16[token_id])

    out = np.zeros(hidden_dim, dtype=np.float32)
    for g in range(num_groups):
        scale = s_row[g]
        bias = b_row[g]
        for p in range(packed_per_group):
            packed = int(w_row[g * packed_per_group + p])
            base = g * group_size + p * 8
            for n in range(8):
                nibble = (packed >> (n * 4)) & 0xF
                out[base + n] = float(nibble) * scale + bias

    return out


def rms_norm(x, w_bf16, eps=RMS_NORM_EPS):
    """RMS norm with bf16 weights."""
    w = bf16_to_f32(w_bf16)
    sum_sq = np.sum(x * x)
    inv_rms = 1.0 / np.sqrt(sum_sq / len(x) + eps)
    return x * inv_rms * w


def rms_norm_bare(x, eps=1e-6):
    """RMS norm without weights."""
    sum_sq = np.sum(x * x)
    inv_rms = 1.0 / np.sqrt(sum_sq / len(x) + eps)
    return x * inv_rms


def silu(x):
    return x / (1.0 + np.exp(-x))


def conv1d_step(conv_state, new_input, weight_bf16, channels, kernel_size):
    """Conv1d step matching C implementation. conv_state: [(kernel_size-1)*channels]."""
    w_f32 = bf16_to_f32(weight_bf16.ravel())  # [channels * kernel_size]

    out = np.zeros(channels, dtype=np.float32)
    for c in range(channels):
        acc = 0.0
        for k in range(kernel_size - 1):
            w = w_f32[c * kernel_size + k]
            acc += conv_state[k * channels + c] * w
        # New input (last kernel position)
        w = w_f32[c * kernel_size + (kernel_size - 1)]
        acc += new_input[c] * w
        out[c] = acc

    # Apply SiLU
    out = silu(out)
    return out


def rms_norm_gated(x, z, w_bf16, eps=RMS_NORM_EPS):
    """RMSNormGated: rms_norm(x) * silu(z) * w."""
    w = bf16_to_f32(w_bf16)
    sum_sq = np.sum(x * x)
    inv_rms = 1.0 / np.sqrt(sum_sq / len(x) + eps)
    silu_z = z / (1.0 + np.exp(-z))
    return x * inv_rms * w * silu_z


def main():
    print("Loading weight manifest...")
    wf = WeightFile(WEIGHTS_BIN, WEIGHTS_JSON)

    # ---- Step 0: Embedding lookup ----
    print("Computing embedding for token_id=9419 ('Hello')...")
    emb_w = wf.get("model.embed_tokens.weight")
    emb_s = wf.get("model.embed_tokens.scales")
    emb_b = wf.get("model.embed_tokens.biases")
    hidden = dequant_embed_row(emb_w, emb_s, emb_b, TOKEN_ID, HIDDEN_DIM)
    print_step("embedding", hidden)

    # Save residual
    residual = hidden.copy()

    # ---- Step 1: Input LayerNorm ----
    norm_w = wf.get("model.layers.0.input_layernorm.weight")
    normed = rms_norm(hidden, norm_w)
    print_step("input_layernorm", normed)

    # ---- Step 2: QKV projection (dequant matvec) ----
    print("Computing QKV projection (12288 x 4096)... this may take a moment")
    qkv_w = wf.get("model.layers.0.linear_attn.in_proj_qkv.weight")
    qkv_s = wf.get("model.layers.0.linear_attn.in_proj_qkv.scales")
    qkv_b = wf.get("model.layers.0.linear_attn.in_proj_qkv.biases")
    qkv = dequant_matvec(qkv_w, qkv_s, qkv_b, normed, LINEAR_CONV_DIM, HIDDEN_DIM, GROUP_SIZE)
    print_step("qkv_proj", qkv)

    # ---- Step 2b: Z projection ----
    print("Computing Z projection (8192 x 4096)...")
    z_w = wf.get("model.layers.0.linear_attn.in_proj_z.weight")
    z_s = wf.get("model.layers.0.linear_attn.in_proj_z.scales")
    z_b = wf.get("model.layers.0.linear_attn.in_proj_z.biases")
    z = dequant_matvec(z_w, z_s, z_b, normed, LINEAR_TOTAL_VALUE, HIDDEN_DIM, GROUP_SIZE)
    print_step("z_proj", z)

    # ---- Step 2c: Beta projection ----
    print("Computing beta projection (64 x 4096)...")
    b_w = wf.get("model.layers.0.linear_attn.in_proj_b.weight")
    b_s = wf.get("model.layers.0.linear_attn.in_proj_b.scales")
    b_b = wf.get("model.layers.0.linear_attn.in_proj_b.biases")
    beta = dequant_matvec(b_w, b_s, b_b, normed, LINEAR_NUM_V_HEADS, HIDDEN_DIM, GROUP_SIZE)
    print_step("beta_proj", beta)

    # ---- Step 2d: Alpha projection ----
    print("Computing alpha projection (64 x 4096)...")
    a_w = wf.get("model.layers.0.linear_attn.in_proj_a.weight")
    a_s = wf.get("model.layers.0.linear_attn.in_proj_a.scales")
    a_b = wf.get("model.layers.0.linear_attn.in_proj_a.biases")
    alpha = dequant_matvec(a_w, a_s, a_b, normed, LINEAR_NUM_V_HEADS, HIDDEN_DIM, GROUP_SIZE)
    print_step("alpha_proj", alpha)

    # ---- Step 3: Conv1d step ----
    # First token: conv_state is all zeros
    conv_state = np.zeros((CONV_KERNEL_SIZE - 1) * LINEAR_CONV_DIM, dtype=np.float32)
    conv_w = wf.get("model.layers.0.linear_attn.conv1d.weight")  # [12288, 4, 1]
    conv_out = conv1d_step(conv_state, qkv, conv_w, LINEAR_CONV_DIM, CONV_KERNEL_SIZE)
    print_step("conv1d_out", conv_out)

    # ---- Step 4: Split into Q, K, V ----
    lin_q = conv_out[:LINEAR_TOTAL_KEY].copy()        # [2048]
    lin_k = conv_out[LINEAR_TOTAL_KEY:2*LINEAR_TOTAL_KEY].copy()  # [2048]
    lin_v = conv_out[2*LINEAR_TOTAL_KEY:].copy()      # [8192]
    print_step("lin_q (before norm)", lin_q)
    print_step("lin_k (before norm)", lin_k)
    print_step("lin_v", lin_v)

    # ---- Step 5: RMS normalize Q and K per head ----
    inv_scale = 1.0 / np.sqrt(float(LINEAR_KEY_DIM))  # 1/sqrt(128)

    for h in range(LINEAR_NUM_K_HEADS):
        qh = lin_q[h*LINEAR_KEY_DIM:(h+1)*LINEAR_KEY_DIM]
        qh_normed = rms_norm_bare(qh)
        q_scale = inv_scale * inv_scale  # 1/128
        lin_q[h*LINEAR_KEY_DIM:(h+1)*LINEAR_KEY_DIM] = qh_normed * q_scale

    for h in range(LINEAR_NUM_K_HEADS):
        kh = lin_k[h*LINEAR_KEY_DIM:(h+1)*LINEAR_KEY_DIM]
        kh_normed = rms_norm_bare(kh)
        lin_k[h*LINEAR_KEY_DIM:(h+1)*LINEAR_KEY_DIM] = kh_normed * inv_scale

    print_step("lin_q (after norm)", lin_q)
    print_step("lin_k (after norm)", lin_k)

    # ---- Step 6: Gated Delta-Net recurrence ----
    A_log = wf.get("model.layers.0.linear_attn.A_log")  # float32 [64]
    dt_bias_bf16 = wf.get("model.layers.0.linear_attn.dt_bias")  # bf16 [64]
    dt_bias = bf16_to_f32(dt_bias_bf16)

    # Zero initial state
    ssm_state = np.zeros((LINEAR_NUM_V_HEADS, LINEAR_VALUE_DIM, LINEAR_KEY_DIM), dtype=np.float32)
    out_values = np.zeros(LINEAR_TOTAL_VALUE, dtype=np.float32)

    k_heads_per_v = LINEAR_NUM_V_HEADS // LINEAR_NUM_K_HEADS  # 4

    # Precompute decay and beta_gate
    g_decay = np.zeros(LINEAR_NUM_V_HEADS, dtype=np.float32)
    beta_gate = np.zeros(LINEAR_NUM_V_HEADS, dtype=np.float32)

    for vh in range(LINEAR_NUM_V_HEADS):
        a_val = alpha[vh]
        dt_b = dt_bias[vh]
        A_val = np.exp(A_log[vh])
        softplus_val = np.log(1.0 + np.exp(a_val + dt_b))
        g_decay[vh] = np.exp(-A_val * softplus_val)
        beta_gate[vh] = 1.0 / (1.0 + np.exp(-beta[vh]))

    print_step("g_decay", g_decay)
    print_step("beta_gate", beta_gate)

    for vh in range(LINEAR_NUM_V_HEADS):
        kh = vh // k_heads_per_v
        g = g_decay[vh]
        b_gate = beta_gate[vh]

        S = ssm_state[vh]  # [VALUE_DIM, KEY_DIM]
        v_h = lin_v[vh*LINEAR_VALUE_DIM:(vh+1)*LINEAR_VALUE_DIM]
        k_h = lin_k[kh*LINEAR_KEY_DIM:(kh+1)*LINEAR_KEY_DIM]

        # Step 1: Decay
        S *= g

        # Step 2: Update
        for vi in range(LINEAR_VALUE_DIM):
            kv_mem = np.dot(S[vi], k_h)
            delta = (v_h[vi] - kv_mem) * b_gate
            S[vi] += k_h * delta

        # Step 3: Output
        q_h = lin_q[kh*LINEAR_KEY_DIM:(kh+1)*LINEAR_KEY_DIM]
        for vi in range(LINEAR_VALUE_DIM):
            out_values[vh*LINEAR_VALUE_DIM + vi] = np.dot(S[vi], q_h)

    print_step("delta_net_out", out_values)

    # ---- Step 7: Gated RMS norm ----
    gated_norm_w = wf.get("model.layers.0.linear_attn.norm.weight")  # bf16 [128]
    gated_out = np.zeros(LINEAR_TOTAL_VALUE, dtype=np.float32)

    for vh in range(LINEAR_NUM_V_HEADS):
        oh = out_values[vh*LINEAR_VALUE_DIM:(vh+1)*LINEAR_VALUE_DIM]
        zh = z[vh*LINEAR_VALUE_DIM:(vh+1)*LINEAR_VALUE_DIM]
        gated_out[vh*LINEAR_VALUE_DIM:(vh+1)*LINEAR_VALUE_DIM] = rms_norm_gated(oh, zh, gated_norm_w)

    print_step("gated_rms_out", gated_out)

    # ---- Step 8: Output projection ----
    print("Computing output projection (4096 x 8192)... this may take a moment")
    out_w = wf.get("model.layers.0.linear_attn.out_proj.weight")
    out_s = wf.get("model.layers.0.linear_attn.out_proj.scales")
    out_b = wf.get("model.layers.0.linear_attn.out_proj.biases")
    attn_out = dequant_matvec(out_w, out_s, out_b, gated_out, HIDDEN_DIM, LINEAR_TOTAL_VALUE, GROUP_SIZE)
    print_step("out_proj", attn_out)

    # ---- Step 9: Residual add ----
    final_hidden = residual + attn_out
    print_step("residual_add", final_hidden)

    print("\nDone. Compare these values against C/Vulkan debug output.")

    wf.close()


if __name__ == "__main__":
    main()
