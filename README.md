# Flash-MoE Vulkan: 397B Parameters on AMD/Linux via SSD Streaming

**Vulkan/Linux port of [Flash-MoE](https://github.com/danielwoods/flash-moe)** — runs the full Qwen3.5-397B-A17B (397 billion parameter MoE) model on an AMD Ryzen AI Max+ 395 mini PC by streaming 209GB of expert weights from NVMe SSD through Vulkan compute shaders.

No Python runtime. No frameworks. Pure C + GLSL compute shaders + io_uring.

> The original Flash-MoE engine runs on Apple Silicon via Metal. This fork adds a complete Vulkan backend for Linux, targeting AMD (tested on RDNA 3.5) but portable to any Vulkan 1.3 GPU.

## Vulkan Results

| Hardware | tok/s | Quality | Bottleneck |
|----------|-------|---------|------------|
| AMD Ryzen AI Max+ 395 (Radeon 8060S) | **3.13** | Excellent | SSD throughput (I/O bound) |

### Per-Layer Timing Breakdown

```
input_norm:       0.20 ms   (GPU)
attn_proj:        0.55 ms   (GPU dequant matvec)
gpu_attn:         0.09 ms   (GPU linear attention — was 9.83ms on CPU)
o_proj:           0.27 ms   (GPU)
post_norm:        0.36 ms   (GPU)
expert_io:        3.78 ms   (NVMe SSD, ~7.4 GB/s)
expert_compute:   0.02 ms   (GPU fused SwiGLU, deferred)
moe_combine:      0.00 ms   (GPU, deferred)
─────────────────────────────
TOTAL:            5.32 ms/layer × 60 layers ≈ 3.13 tok/s
```

Expert I/O dominates at 71% of per-layer time. GatedDeltaNet linear attention now runs entirely on GPU (5 compute shaders), down from 9.83ms on CPU BLAS to 0.09ms — a 109x speedup. Deferred command buffers overlap expert compute with next layer's I/O. SSD throughput (~7.4 GB/s) is the remaining bottleneck (vs Apple's 17.5 GB/s).

## Vulkan Quick Start

### Prerequisites

```bash
# Arch/Manjaro/CachyOS
sudo pacman -S vulkan-devel shaderc liburing openblas

# Ubuntu/Debian
sudo apt install libvulkan-dev glslc liburing-dev libopenblas-dev
```

### Build

```bash
cd vulkan_infer
make          # compiles shaders to SPIR-V + builds infer binary
```

### Prepare Model Weights (~415GB temp, 215GB final)

```bash
pip install huggingface_hub safetensors numpy

# Download model (~200GB)
huggingface-cli download mlx-community/Qwen3.5-397B-A17B-4bit --local-dir ./model

# Extract non-expert weights (5.5GB)
cd metal_infer && python extract_weights.py --model ../model && cd ..

# Pack experts into per-layer binary files (209GB, ~2.5 min)
python repack_experts.py --index expert_index.json

# Export tokenizer
python metal_infer/export_tokenizer.py ./model/tokenizer.json metal_infer/tokenizer.bin

# Symlink into vulkan_infer/
cd vulkan_infer
ln -s ../metal_infer/model_weights.bin ../metal_infer/model_weights.json ../metal_infer/tokenizer.bin .
ln -s ../model/packed_experts .
```

### Run

```bash
./infer --prompt "Explain quantum computing" --tokens 100
./infer --prompt "Hello" --tokens 20 --timing    # per-layer breakdown
./infer --prompt "Hello" --tokens 10 --debug      # intermediate value traces
```

### Hardware Requirements (Vulkan)

| Requirement | Minimum |
|-------------|---------|
| **GPU** | Any Vulkan 1.3 GPU with `GL_KHR_shader_subgroup_arithmetic` |
| **RAM** | 32GB (16GB may work — only ~6GB used, rest is OS page cache) |
| **SSD** | 256GB+ free, NVMe strongly recommended (SATA will be very slow) |
| **OS** | Linux, kernel 5.1+ (for io_uring) |

Faster SSD = faster inference. The engine uses ~6GB RAM for weights and scratch buffers. All 209GB of expert data streams from SSD on demand.

---

## Background: How Flash-MoE Works

> **[Read the paper](paper/flash_moe.pdf)** for full technical details, 90+ experiments, and the story behind the original Metal implementation.

The model has 60 transformer layers: 45 GatedDeltaNet (linear attention) + 15 standard full attention. Each layer has 512 experts, of which K=4 are activated per token (plus one shared expert). At 4-bit quantization, each expert is ~7MB.

The key insight: only ~28MB of expert data is needed per layer per token. Instead of loading the entire 209GB model into RAM, stream just the active experts from SSD on demand. The OS page cache naturally caches hot experts (~71% hit rate).

### Key Techniques

1. **SSD Expert Streaming** — Read K=4 active experts (~28MB) per layer from NVMe via io_uring (Linux) or GCD+pread (macOS). OS page cache handles caching. Inspired by Apple's [LLM in a Flash](https://arxiv.org/abs/2312.11514).

2. **FMA-Optimized 4-bit Dequant** — Rearrange `(nibble * scale + bias) * x` into `fma(nibble, scale*x, bias*x)` for one GPU FMA instruction per element. +12% throughput.

3. **Fused Compute Kernels** — Gate+Up+SwiGLU in one kernel (single input read), MoE combine+residual fused, two-pass RMS norm with subgroup reduction.

4. **GPU Linear Attention** — GatedDeltaNet recurrence (64 heads x 128x128 state) via 5 Vulkan compute shaders: `conv1d_step`, `gated_rms_norm`, `rms_norm_qk`, `compute_decay_beta`, `gated_delta_net_step`. 109x faster than CPU BLAS.

5. **Deferred Command Buffers** — Expert compute and MoE combine run deferred, overlapping GPU work with next layer's SSD I/O. Merged 5 submit+waits per layer into 1–2.

## All Results

| Hardware | Backend | tok/s | Quality | Notes |
|----------|---------|-------|---------|-------|
| M3 Max (48GB) | Metal | **4.36** | Excellent | Full tool calling. Production config. |
| Ryzen AI Max+ 395 | Vulkan | **3.13** | Excellent | I/O-bound. GPU linear attn + deferred CMDs. |
| M3 Max (2-bit) | Metal | 5.74 | Good* | *Breaks JSON/tool calling. |

The Vulkan port is slower due to hardware differences: 2.6x less memory bandwidth (212 vs 546 GB/s) and 2.4x slower SSD (7.4 vs 17.5 GB/s).

## Project Structure

```
vulkan_infer/             # Vulkan/Linux backend (this port)
  infer.c                 #   inference engine (~1500 lines C)
  vk_compute.h/c          #   Vulkan abstraction (~600 lines)
  io_ring.h/c             #   io_uring expert streaming
  weights.h/c             #   weight loading + JSON manifest parser
  linear_attn.h/c         #   CPU GatedDeltaNet via OpenBLAS
  full_attn.h/c           #   CPU full attention with RoPE + KV cache
  shaders/                #   14 GLSL compute shaders → SPIR-V
    dequant_matvec_4bit.comp    FMA-optimized 4-bit dequant (critical path)
    fused_gate_up_swiglu.comp   Fused expert gate+up+SwiGLU
    rms_norm.comp               Two-pass RMS normalization
    moe_combine.comp            Expert combine + residual
    attn_{scores,softmax,values}.comp   Full attention
    residual_add.comp           Element-wise residual
    sigmoid_gate.comp           Sigmoid gating
    gated_delta_net_step.comp   GPU GatedDeltaNet recurrence
    conv1d_step.comp            Short convolution for linear attn
    compute_decay_beta.comp     Decay/beta gate computation
    gated_rms_norm.comp         Gated RMS norm for linear attn
    rms_norm_qk.comp            Q/K normalization

metal_infer/              # Original Metal/macOS backend
  infer.m                 #   inference engine (~7000 lines Obj-C)
  shaders.metal           #   Metal compute kernels (~1300 lines)
  chat.m                  #   interactive chat TUI with tool calling

repack_experts.py         # 4-bit expert packing from safetensors
expert_index.json         # expert file layout metadata
paper/                    # research paper (PDF + LaTeX)
```

## Vulkan Port Architecture

The Vulkan backend is a from-scratch reimplementation of the Metal engine.

| Aspect | Metal (original) | Vulkan (this port) |
|--------|-----------------|-------------------|
| GPU API | Metal compute | Vulkan 1.3 (RADV) |
| Memory | `newBufferWithBytesNoCopy` | Raw `vkAllocateMemory` (UMA) |
| Async I/O | GCD dispatch groups + `pread` | io_uring batch submission |
| Shaders | MSL (~1300 lines) | GLSL 450 → SPIR-V (~800 lines) |
| BLAS | Apple Accelerate | OpenBLAS |
| Subgroup | SIMD width 32 (fixed) | Dynamic via `gl_SubgroupSize` (wave64 on AMD) |

### Porting Lessons

**Wave64 matters.** AMD RDNA 3.5 defaults to 64-wide subgroups, not 32 like Metal. Hardcoding `ROWS_PER_TG=8` (assuming wave32) silently computed only half the output vectors — the shader's bounds check prevented crashes but produced zeros for the missing rows.

**Shared memory sizes aren't portable.** `shared float x_shared[4096]` worked for most projections (`in_dim=4096`), but the O projection has `in_dim=8192`. The overflow didn't crash — it produced subtly wrong values that were hard to trace.

**Skip VMA on UMA.** Vulkan Memory Allocator requires C++ and is overkill for unified memory where every buffer is `HOST_VISIBLE | DEVICE_LOCAL`. Raw `vkAllocateMemory` is ~50 lines of C.

**io_uring is elegant.** One `io_uring_submit` + one `io_uring_wait_cqe_nr` replaces 4 separate `pread` syscalls with better batching.

## Metal Quick Start (macOS)

```bash
cd metal_infer
make
./infer --prompt "Explain quantum computing" --tokens 100
./chat    # interactive chat with tool calling
```

## What We Tried (and What Worked)

### Kept
| Approach | Impact |
|----------|--------|
| FMA dequant kernel | **+12% tok/s** |
| Trust OS page cache | **+38%** (deleted custom LRU) |
| GPU combine+norm | Eliminates CPU round-trip |
| GPU delta-net | **109x** faster than CPU BLAS |
| Deferred command buffers | **+15%** (overlap GPU + SSD I/O) |
| C BPE tokenizer | **20x** faster startup |

### Discarded (58 experiments)
| Approach | Result | Why |
|----------|--------|-----|
| LZ4 expert compression | -13% | Decompress overhead > cache savings |
| Temporal expert prediction | -18% | 25% hit rate, bandwidth waste |
| mmap expert files | -5x | Per-page fault overhead on cold data |
| dispatch_io | -70% | dispatch_data management overhead |
| Speculative routing | -38% | Cache pollution |

## License

MIT
