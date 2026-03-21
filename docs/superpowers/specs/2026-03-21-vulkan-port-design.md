# Flash-MoE Vulkan Port: Design Specification

**Date:** 2026-03-21
**Status:** Approved
**Target Hardware:** AMD Ryzen AI Max+ 395 (Minisforum MS-S1 Max)
**Goal:** Minimal viable inference of Qwen3.5-397B-A17B via Vulkan + io_uring on Linux

## Overview

Port the Flash-MoE inference engine from Apple Metal/macOS to Vulkan/Linux, targeting the AMD Ryzen AI Max+ 395 with its Radeon 8060S integrated GPU (40 RDNA 3.5 CUs, 212 GB/s unified memory bandwidth) and PCIe 4.0 x4 NVMe (~5-7 GB/s sequential read).

The architecture is I/O-bound on this hardware (vs roughly balanced on M3 Max), so the design preserves the proven SSD-streaming approach while adapting the GPU backend to Vulkan compute shaders.

**Expected performance:** ~1-3 tok/s depending on page cache warmth (vs 4.36 on M3 Max). Cold start ~1 tok/s, warm cache ~2-3 tok/s. Limited primarily by SSD throughput and 2.6x lower memory bandwidth.

## Hardware Comparison

| Dimension | M3 Max (original) | Ryzen AI Max+ 395 (target) |
|-----------|-------------------|---------------------------|
| GPU cores | 40-core Apple GPU | 40 RDNA 3.5 CUs |
| Memory BW | 546 GB/s | 212 GB/s (2.6x less) |
| SSD read | 17.5 GB/s measured | ~5-7 GB/s (PCIe 4.0 x4) |
| Unified memory | 48 GB | Up to 128 GB |
| GPU API | Metal | Vulkan 1.3 (RADV) |
| Async I/O | GCD dispatch groups | io_uring |

## Project Structure

```
vulkan_infer/
  infer.c              # Main inference engine
  vk_compute.h         # Thin Vulkan abstraction (device, pipelines, buffers, dispatch)
  vk_compute.c         # Implementation (uses volk + VMA)
  io_ring.h            # io_uring wrapper for expert streaming
  io_ring.c            # Implementation
  tokenizer.h          # Reused from metal_infer/ (unchanged)
  Makefile             # Build with glslc for SPIR-V compilation
  deps/
    volk.h             # Vulkan meta-loader (header-only, vendored)
    vk_mem_alloc.h     # Vulkan Memory Allocator (header-only, vendored)
  shaders/
    # Core compute
    dequant_matvec_4bit.comp   # FMA-optimized 4-bit dequant matvec
    rms_norm.comp              # Two-pass RMS normalization (sum-of-squares + apply)
    # Expert MoE
    fused_gate_up_swiglu.comp  # Fused gate_proj + up_proj + SwiGLU (single read of input)
    moe_combine.comp           # Weighted expert sum + shared expert + residual add
    # Full attention (15 layers)
    attention.comp             # Batched Q@K^T, softmax, scores@V
    sigmoid_gate.comp          # Element-wise sigmoid gating for full-attn output
    residual_add.comp          # Residual connection (GPU-side)
    # Linear attention (45 layers) — initial port uses CPU BLAS
    # Future: conv1d.comp, delta_net_step.comp, compute_decay_beta.comp,
    #         rms_norm_qk.comp, gated_rms_norm.comp
```

**Build dependencies:**
- Vulkan SDK (headers + `glslc` shader compiler)
- `liburing` (io_uring userspace library)
- OpenBLAS or BLIS (for CPU-side linear attention BLAS)
- volk.h and vk_mem_alloc.h vendored in `deps/`

**Model data:** Symlinked from `metal_infer/` — same binary formats, same Python preparation scripts.

## Vulkan Compute Abstraction (`vk_compute.h`)

Thin layer hiding Vulkan ceremony, exposing inference-relevant primitives:

```c
typedef struct VkCtx VkCtx;     // Device, queues, command pool
typedef struct VkBuf VkBuf;     // Buffer + VMA allocation + persistent mapping
typedef struct VkPipe VkPipe;   // Compute pipeline + descriptor set layout
typedef struct VkCmd VkCmd;     // Reusable command buffer

// Lifecycle
VkCtx* vk_create(void);
void   vk_destroy(VkCtx*);

// Buffers — HOST_VISIBLE | DEVICE_LOCAL on unified memory
VkBuf* vk_buf_create(VkCtx*, size_t bytes, VkBufferUsageFlags);
void*  vk_buf_map(VkBuf*);        // Persistent map (no unmap on UMA)
void   vk_buf_destroy(VkBuf*);

// Import: copy external data (e.g. mmap'd weights) into a VkBuf at startup
VkBuf* vk_buf_create_from_data(VkCtx*, const void* src, size_t bytes, VkBufferUsageFlags);

// Pipelines — one per SPIR-V shader
VkPipe* vk_pipe_create(VkCtx*, const char* spv_path,
                        uint32_t push_const_size, uint32_t num_buffers);

// Command buffers — map to Metal's command buffer concept
VkCmd* vk_cmd_begin(VkCtx*);
void   vk_cmd_bind(VkCmd*, VkPipe*, VkBuf** bufs, uint32_t num_bufs,
                    void* push_consts, uint32_t push_size);
void   vk_cmd_dispatch(VkCmd*, uint32_t gx, uint32_t gy, uint32_t gz);
void   vk_cmd_barrier(VkCmd*);        // Compute-to-compute memory barrier
void   vk_cmd_submit(VkCmd*);         // Submit + fence wait
void   vk_cmd_submit_no_wait(VkCmd*); // Deferred (CMD3 pattern)
void   vk_cmd_wait(VkCmd*);           // Explicit fence wait
```

**Design decisions:**
- **All buffers `HOST_VISIBLE | DEVICE_LOCAL`:** On Strix Halo UMA, CPU and GPU share LPDDR5x. VMA selects the optimal memory type. Expert weights read by io_uring into mapped buffers are immediately GPU-accessible — no staging copies.
- **Push constants** for per-dispatch parameters (dimensions, offsets) instead of uniform buffers. Faster for small data.
- **Pipeline cache** stored to disk — shader compilation only happens on first run.
- **Single compute queue** — RDNA 3.5 has one hardware compute queue.
- **Barrier semantics:** `vk_cmd_barrier()` inserts a `VkMemoryBarrier2` with src=`VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_ACCESS_SHADER_WRITE_BIT`, dst=`VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_ACCESS_SHADER_READ_BIT`. This ensures all shader writes from previous dispatches are visible to subsequent dispatches within the same command buffer.

## Non-expert Weight Loading

The original wraps mmap'd data as Metal buffers via `newBufferWithBytesNoCopy`. Vulkan cannot wrap arbitrary mmap'd memory as a VkBuffer. Two options:

- **Option A:** `VK_EXT_external_memory_host` (if supported by RADV) to import mmap'd pages directly.
- **Option B (chosen):** Allocate a 5.5GB VkBuffer and `memcpy` from mmap at startup. Adds ~1 second to startup on this hardware. Simpler, no extension dependency.

At startup:
1. `mmap()` `model_weights.bin` read-only
2. Parse `model_weights.json` manifest for tensor offsets/shapes
3. Allocate one large VkBuf (5.5GB, `HOST_VISIBLE | DEVICE_LOCAL`)
4. `memcpy` entire mmap into VkBuf mapped pointer
5. `munmap` the file
6. All weight lookups use offsets into the VkBuf

## Persistent State Buffers

Allocated at startup, maintained across tokens:

**GatedDeltaNet state (45 layers):**
- Per-layer state matrix: 64 heads x 128 x 128 x float32 = 4 MB/layer
- Total: 45 x 4 MB = ~180 MB
- Allocated as CPU-accessible buffers (used by BLAS on CPU in initial port)

**Conv1d state (45 layers):**
- Per-layer: kernel_size=3 x (q_dim + k_dim + v_dim + ...) x float32
- ~3 x 12288 x 4 = ~144 KB/layer, ~6.3 MB total
- CPU-accessible buffers

**KV cache (15 full-attention layers):**
- Per-layer: 2 (K+V) x num_kv_heads x head_dim x max_seq_len x float32
- At max_seq_len=8192: 2 x 2 x 256 x 8192 x 4 = ~32 MB/layer
- Total: 15 x 32 MB = ~480 MB
- Allocated as VkBufs (GPU-accessible for attention kernels)

**Total persistent state:** ~670 MB

## GPU Compute Shaders

### `dequant_matvec_4bit.comp` (critical path)

Same FMA optimization as Metal: `fma(nibble, scale_x, bias_x)` where `scale_x = scale * x[i]` and `bias_x = bias * x[i]` precomputed per group of 64 elements.

**Adaptations from Metal:**
- **bfloat16:** No native GLSL support. Stored as `uint16`, converted via `uintBitsToFloat(uint(bf16) << 16)`. ~1 extra instruction per conversion, amortized over group size (64).
- **Subgroup ops:** Metal SIMD reduction → GLSL `subgroupAdd()` (KHR_shader_subgroup_arithmetic). RDNA 3.5 wave32 maps directly to Metal's SIMD width.
- **Shared memory:** Metal `threadgroup` → GLSL `shared` arrays for input vector caching.
- **Workgroup size:** 256 threads (8 rows x 32 threads), matching Metal's tiling.

Used for: all projection matvecs (attention Q/K/V/O, routing gate, shared expert projections, lm_head).

### `rms_norm.comp`

Two-pass fused into single dispatch with workgroup barrier:
1. Parallel sum-of-squares with subgroup reduction
2. Barrier
3. Element-wise `x[i] * weight[i] * rsqrt(sum/n + eps)`

### `fused_gate_up_swiglu.comp` (expert forward pass)

Fuses gate_proj + up_proj + SwiGLU activation into a single kernel that reads the input vector once. For each output element: load input from shared memory, compute gate and up projections via dequant dot products, apply `silu(gate) * up`. This is critical for bandwidth efficiency — three separate dispatches would read the 4096-dim input three times.

Used for: K=4 active expert forward passes and the shared expert.

### `moe_combine.comp`

Computes `h_mid + sum(weight_k * expert_out_k) + sigmoid(shared_gate) * shared_out` where h_mid is the pre-MoE residual. Does NOT include RMS norm — that is a separate `rms_norm.comp` dispatch following the combine. Matches the original Metal kernel's decomposition.

### `attention.comp`

Three sub-dispatches for the 15 full-attention layers:
1. **Scores:** Q @ K^T with causal masking
2. **Softmax:** Per-head softmax over attention scores
3. **Values:** Scores @ V to produce attention output

KV cache is read/written during these dispatches.

### `sigmoid_gate.comp`

Element-wise `out[i] = x[i] * sigmoid(gate[i])`. Used in CMD2 for the 15 full-attention layers' output gating.

### `residual_add.comp`

Element-wise `out[i] = a[i] + b[i]`. Used in CMD2 for residual connections on GPU.

### CPU-side compute (initial port)

**GatedDeltaNet linear attention (45 layers):** The full GPU pipeline exists in the original (conv1d_step, gated_delta_net_step, compute_decay_beta, rms_norm_qk, gated_rms_norm) but for the initial port we use the CPU BLAS fallback path:
- Conv1d with SiLU activation
- Per-head QK normalization
- Decay/beta computation
- Delta-net recurrence: 64 heads x 128x128 state via `cblas_sscal`, `cblas_sgemv`, `cblas_sger`
- Gated output normalization

This is a deliberate simplification. The CPU path adds ~0.3-0.8ms per layer (measured on Apple's Accelerate; OpenBLAS on Zen5 should be comparable). The cost is a CPU-GPU sync point between CMD1 and CMD2 for linear attention layers. Porting the 5 GPU kernels is a follow-up optimization that eliminates this sync.

**RoPE (15 full-attention layers):** Applied on CPU between CMD1 wait and CMD2 encode, same as original. Partial rotary (64 of 256 head dims). Not a bottleneck.

**Top-K routing:** Softmax over 512 expert scores + top-4 selection. Trivial.

**Sampling:** Temperature scaling, top-p nucleus sampling, argmax. CPU-side after lm_head, same as original.

## I/O Pipeline (`io_ring.h`)

```c
typedef struct IoRing IoRing;

IoRing* io_ring_create(uint32_t queue_depth);   // 16-32 entries
void    io_ring_destroy(IoRing*);

// Batch-submit K=4 expert reads, wait for all completions
void io_ring_read_experts(IoRing*,
                          int layer_fd,
                          uint64_t offsets[4],
                          void*    dests[4],     // mapped VkBuf pointers
                          size_t   sizes[4],
                          int      count);
```

**Design decisions:**
- **Single batch submission:** All 4 expert reads submitted as one `io_uring_submit()`, completed via `io_uring_wait_cqe_nr(4)`. One syscall each direction vs 4 pread syscalls.
- **Page cache by default:** Regular `open()`, trusting OS LRU cache same as original. `O_DIRECT` available via `--direct-io` flag for experimentation.
- **Aligned buffers:** VMA allocations are page-aligned, satisfying `O_DIRECT` requirements when enabled.
- **Pre-opened FDs:** 60 file descriptors (one per layer) opened at startup.

**Expected I/O per layer:**
- 4 experts x 7MB = 28MB
- Cold: ~4-6ms at 5-7 GB/s
- Page cache hit (~71%): ~1.2-1.6ms
- I/O is the dominant bottleneck (vs roughly balanced on Mac)

## Inference Pipeline

### Full forward pass (per token)

```
1. Embedding lookup: dequant row fetch from [248320, 512] weight → 4096-dim hidden
2. For each of 60 layers: fused_layer_forward (see below)
3. Final RMS norm
4. lm_head projection: dequant_matvec [4096] → [248320] (largest single matvec)
5. Sampling: temperature, top-p, argmax → next token ID
6. Token decode via tokenizer
```

The lm_head matvec dispatches ~31040 threadgroups (248320 / 8 rows per TG). At 212 GB/s bandwidth this is a significant per-token cost (~several ms). This is unavoidable — it's the same bottleneck on every LLM.

### Per-layer pipeline (fused_layer_forward)

```
CMD1: attention projections (dequant_matvec)      [~3.1ms GPU]
  → CPU: RoPE (full-attn) or linear attn (BLAS)   [~0.3-0.8ms]
  → CMD2: o_proj + residual + norm + routing       [~1.4ms GPU]
      (includes sigmoid_gate + residual_add for full-attn layers)
  → CPU: softmax + topK routing                    [~0.003ms]
  → I/O: io_uring batch read K=4 experts           [~4-6ms SSD cold, ~1.5ms warm]
  → CMD3: expert fused_gate_up_swiglu + combine    [~0.1ms GPU encode, deferred]
      (moe_combine + rms_norm as separate dispatches within CMD3)
```

**Serial pipeline:** GPU → CPU → GPU → I/O → GPU strictly sequential (same as Mac). Overlap experiments are future work.

**Single-buffered expert data:** One set of K=4 expert VkBufs. Double-buffering is scaffolding for future I/O overlap — not needed for serial mode.

**Deferred CMD3:** Submitted without fence wait. Vulkan queue serializes via pipeline barriers: CMD2 → CMD3 → next CMD1.

**Per-token estimate:**
- Cold cache: 60 layers x ~9-10ms = 540-600ms → ~1.0-1.1 tok/s
- Warm cache (~71% hit): 60 layers x 6-8ms = 360-480ms → ~2.0-2.8 tok/s
- Plus lm_head + embedding + sampling overhead: ~5-10ms

## Weight Preparation (run once)

Existing Python scripts, no macOS dependencies:

```bash
# 1. Download model from HuggingFace (~200GB)
pip install huggingface_hub safetensors numpy
huggingface-cli download Qwen/Qwen3.5-397B-A17B-4bit --local-dir ./model

# 2. Extract non-expert weights (produces model_weights.bin + .json)
cd metal_infer && python extract_weights.py --model-dir ../model

# 3. Pack experts into per-layer binary files (produces packed_experts/)
cd .. && python repack_experts.py --model-dir ./model --output-dir vulkan_infer/packed_experts

# 4. Export tokenizer (produces vocab.bin + tokenizer.bin)
cd vulkan_infer && python export_tokenizer.py --model-dir ../model

# 5. Symlink shared files
ln -s ../metal_infer/model_weights.bin .
ln -s ../metal_infer/model_weights.json .
ln -s ../metal_infer/tokenizer.h .
```

~415GB temporary disk usage (safetensors + packed), settling to ~215GB after cleanup.

## Scope

**In scope:**
- Vulkan compute pipeline with 7 shader types (dequant_matvec, rms_norm, fused_gate_up_swiglu, moe_combine, attention, sigmoid_gate, residual_add)
- io_uring expert streaming
- 60-layer forward pass: embedding → layers → lm_head → sampling
- Persistent state: delta-net (CPU), conv1d (CPU), KV cache (GPU)
- CLI: `./infer --prompt "..." --tokens N [--timing]`
- 4-bit quantization only
- CPU-side GatedDeltaNet via OpenBLAS
- CPU-side RoPE
- Basic sampling (temperature, top-p)

**Out of scope (future work):**
- Chat TUI / tool calling
- 2-bit quantization
- GPU-side linear attention kernels (conv1d, delta_net_step, compute_decay_beta, rms_norm_qk, gated_rms_norm)
- GPU-side RoPE
- I/O overlap experiments / double-buffered experts
- O_DIRECT benchmarking
- Multi-token / speculative decoding
- AMD-specific optimizations
