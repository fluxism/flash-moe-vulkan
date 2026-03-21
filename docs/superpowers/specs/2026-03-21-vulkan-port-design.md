# Flash-MoE Vulkan Port: Design Specification

**Date:** 2026-03-21
**Status:** Approved
**Target Hardware:** AMD Ryzen AI Max+ 395 (Minisforum MS-S1 Max)
**Goal:** Minimal viable inference of Qwen3.5-397B-A17B via Vulkan + io_uring on Linux

## Overview

Port the Flash-MoE inference engine from Apple Metal/macOS to Vulkan/Linux, targeting the AMD Ryzen AI Max+ 395 with its Radeon 8060S integrated GPU (40 RDNA 3.5 CUs, 212 GB/s unified memory bandwidth) and PCIe 4.0 x4 NVMe (~5-7 GB/s sequential read).

The architecture is I/O-bound on this hardware (vs roughly balanced on M3 Max), so the design preserves the proven SSD-streaming approach while adapting the GPU backend to Vulkan compute shaders.

**Expected performance:** ~2-3 tok/s (vs 4.36 on M3 Max), limited primarily by SSD throughput.

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
    dequant_matvec_4bit.comp   # FMA-optimized 4-bit dequant matvec
    rms_norm.comp              # Two-pass RMS normalization
    swiglu.comp                # Fused SwiGLU activation
    moe_combine.comp           # Expert combine + residual + norm
    rope.comp                  # RoPE (fused with deinterleave)
    attention.comp             # Batched Q@K^T, softmax, scores@V
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

## GPU Compute Shaders

### `dequant_matvec_4bit.comp` (critical path)

Same FMA optimization as Metal: `fma(nibble, scale_x, bias_x)` where `scale_x = scale * x[i]` and `bias_x = bias * x[i]` precomputed per group of 64 elements.

**Adaptations from Metal:**
- **bfloat16:** No native GLSL support. Stored as `uint16`, converted via `uintBitsToFloat(uint(bf16) << 16)`. ~1 extra instruction per conversion, amortized over group size (64).
- **Subgroup ops:** Metal SIMD reduction → GLSL `subgroupAdd()` (KHR_shader_subgroup_arithmetic). RDNA 3.5 wave32 maps directly to Metal's SIMD width.
- **Shared memory:** Metal `threadgroup` → GLSL `shared` arrays for input vector caching.
- **Workgroup size:** 256 threads (8 rows x 32 threads), matching Metal's tiling.

### `rms_norm.comp`

Two-pass fused into single dispatch with workgroup barrier:
1. Parallel sum-of-squares with subgroup reduction
2. Barrier
3. Element-wise `x[i] * weight[i] * rsqrt(sum/n + eps)`

### `swiglu.comp`

`out[i] = gate[i] * sigmoid(gate[i]) * up[i]`

### `moe_combine.comp`

Weighted sum of K=4 expert outputs + shared expert output, residual addition, RMS norm. Single fused dispatch.

### `rope.comp`

Fused RoPE + Q deinterleave + K normalization. Same rotary math.

### `attention.comp`

Three dispatches for the 15 full-attention layers: Q@K^T scores, softmax, scores@V.

### CPU-side compute (not ported to GPU)

- **GatedDeltaNet recurrence:** 64 heads x 128x128 state matrix, sequential per-head. Uses OpenBLAS `cblas_sgemv`, `cblas_sger`, `cblas_sscal`. CPU-bound, not worth GPU dispatch overhead.
- **Top-K routing:** Softmax over 512 expert scores + top-4 selection. Trivial.

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

Per-layer pipeline adapted from Metal's CMD1/CMD2/CMD3:

```
CMD1: attention projections (dequant_matvec)      [~3.1ms GPU]
  → CPU: linear attention recurrence (BLAS)        [~0.3ms]
  → CMD2: o_proj + norm + routing + shared expert  [~1.4ms GPU]
  → CPU: softmax + topK routing                    [~0.003ms]
  → I/O: io_uring batch read K=4 experts           [~4-6ms SSD]
  → CMD3: expert forward + combine + norm          [~0.1ms GPU, deferred]
```

**I/O overlap strategy:**

Default is serial mode (GPU → I/O → GPU), same as Mac. The pipeline is designed to support overlapped mode (CMD3 from previous layer concurrent with io_uring reads for current layer) as a future experiment — AMD's Infinity Fabric may handle concurrent GPU compute + SSD DMA better than Apple's unified memory controller.

**Double-buffered expert data:** Two sets of K=4 expert VkBufs (A and B). GPU computes on A while io_uring fills B.

**Deferred CMD3:** Submitted without fence wait. Vulkan queue serializes via pipeline barriers: CMD2 → CMD3 → next CMD1.

**Per-token estimate:**
- 60 layers x 6-8ms = 360-480ms
- ~2.0-2.8 tok/s initial target
- With warm page cache: potentially ~3+ tok/s

## Weight Loading & Model Setup

**Non-expert weights** (`model_weights.bin`, 5.5GB): `mmap()` read-only, same as original. Manifest parsed from `model_weights.json`. Pointers used directly as shader inputs (host-visible on UMA).

**Expert weights** (`packed_experts/layer_*.bin`, 209GB): Same 4-bit binary format (7,077,888 bytes/expert). Read via io_uring into mapped VkBufs.

**Tokenizer** (`tokenizer.h`, `vocab.bin`, `tokenizer.bin`): Reused unchanged.

**Weight preparation** (existing Python scripts, run once):
```bash
huggingface-cli download Qwen/Qwen3.5-397B-A17B-4bit --local-dir ./model
cd metal_infer && python extract_weights.py --model-dir ../model
cd .. && python repack_experts.py --model-dir ./model --output-dir vulkan_infer/packed_experts
cd vulkan_infer && python export_tokenizer.py --model-dir ../model
```

~415GB temporary disk usage (safetensors + packed), settling to ~215GB after cleanup.

## Scope

**In scope:**
- Vulkan compute pipeline with all 6 shader types
- io_uring expert streaming
- 60-layer forward pass producing tokens
- CLI: `./infer --prompt "..." --tokens N [--timing]`
- 4-bit quantization only
- CPU-side GatedDeltaNet via OpenBLAS
- Basic sampling (temperature, top-p)

**Out of scope (future work):**
- Chat TUI / tool calling
- 2-bit quantization
- I/O overlap experiments
- O_DIRECT benchmarking
- Multi-token / speculative decoding
- AMD-specific optimizations
