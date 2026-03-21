# Flash-MoE Vulkan Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Minimal viable inference of Qwen3.5-397B-A17B on AMD Ryzen AI Max+ 395 via Vulkan compute shaders and io_uring expert streaming.

**Architecture:** C + Vulkan (via volk/VMA) + io_uring + OpenBLAS. Expert weights stream from NVMe SSD through io_uring into persistently mapped Vulkan buffers. GPU compute shaders handle dequant matvec, attention, and expert forward passes. CPU handles GatedDeltaNet linear attention via BLAS.

**Tech Stack:** C11, Vulkan 1.3 (RADV), GLSL 450 compute shaders, liburing, OpenBLAS, volk.h, vk_mem_alloc.h

**Spec:** `docs/superpowers/specs/2026-03-21-vulkan-port-design.md`

---

### Task 1: Project scaffold and build system

**Files:**
- Create: `vulkan_infer/Makefile`
- Create: `vulkan_infer/deps/` (directory for vendored headers)
- Create: `vulkan_infer/shaders/` (directory for compute shaders)

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p vulkan_infer/deps vulkan_infer/shaders
```

- [ ] **Step 2: Download volk.h and vk_mem_alloc.h**

```bash
cd vulkan_infer/deps
curl -LO https://raw.githubusercontent.com/zeux/volk/master/volk.h
curl -LO https://raw.githubusercontent.com/zeux/volk/master/volk.c
curl -LO https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator/master/include/vk_mem_alloc.h
```

- [ ] **Step 3: Symlink tokenizer.h from metal_infer**

```bash
cd vulkan_infer
ln -s ../metal_infer/tokenizer.h tokenizer.h
```

- [ ] **Step 4: Write Makefile**

```makefile
# Makefile for vulkan_infer — Vulkan/Linux MoE inference engine
CC = gcc
CFLAGS = -O2 -Wall -Wextra -std=c11 -D_GNU_SOURCE
LDFLAGS = -lvulkan -luring -lopenblas -lpthread -lm

GLSLC = glslc

INFER_TARGET = infer
INFER_SRC = infer.c vk_compute.c io_ring.c weights.c linear_attn.c full_attn.c deps/volk.c

# Shader sources and SPIR-V outputs
SHADER_DIR = shaders
SHADER_SRCS = $(wildcard $(SHADER_DIR)/*.comp)
SHADER_SPVS = $(SHADER_SRCS:.comp=.spv)

.PHONY: all clean shaders

all: shaders $(INFER_TARGET)

shaders: $(SHADER_SPVS)

$(SHADER_DIR)/%.spv: $(SHADER_DIR)/%.comp
	$(GLSLC) --target-env=vulkan1.3 -O $< -o $@

$(INFER_TARGET): $(INFER_SRC) vk_compute.h io_ring.h
	$(CC) $(CFLAGS) $(INFER_SRC) -o $(INFER_TARGET) $(LDFLAGS)

test_vk: test_vk.c vk_compute.c deps/volk.c shaders/test_add.spv
	$(CC) $(CFLAGS) test_vk.c vk_compute.c deps/volk.c -o test_vk $(LDFLAGS)

test_io: test_io.c io_ring.c
	$(CC) $(CFLAGS) test_io.c io_ring.c -o test_io -luring -lm

test_matvec: test_matvec.c vk_compute.c deps/volk.c shaders/dequant_matvec_4bit.spv
	$(CC) $(CFLAGS) test_matvec.c vk_compute.c deps/volk.c -o test_matvec $(LDFLAGS)

clean:
	rm -f $(INFER_TARGET) test_vk test_io test_matvec $(SHADER_DIR)/*.spv
```

- [ ] **Step 5: Verify build dependencies are available**

```bash
pkg-config --cflags --libs vulkan 2>/dev/null && echo "vulkan: OK" || echo "vulkan: MISSING"
pkg-config --cflags --libs liburing 2>/dev/null && echo "liburing: OK" || echo "liburing: MISSING"
pkg-config --cflags --libs openblas 2>/dev/null && echo "openblas: OK" || echo "openblas: MISSING"
which glslc && echo "glslc: OK" || echo "glslc: MISSING"
```

Install any missing deps. On Arch: `pacman -S vulkan-devel liburing openblas shaderc`.

- [ ] **Step 6: Commit**

```bash
git add vulkan_infer/
git commit -m "feat(vulkan): project scaffold with Makefile and vendored deps"
```

---

### Task 2: Vulkan compute abstraction (`vk_compute.h` / `vk_compute.c`)

**Files:**
- Create: `vulkan_infer/vk_compute.h`
- Create: `vulkan_infer/vk_compute.c`

- [ ] **Step 1: Write a minimal test shader to validate the abstraction**

Create `vulkan_infer/shaders/test_add.comp`:
```glsl
#version 450
layout(local_size_x = 256) in;
layout(binding = 0) buffer A { float a[]; };
layout(binding = 1) buffer B { float b[]; };
layout(binding = 2) buffer C { float c[]; };
layout(push_constant) uniform Params { uint n; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < n) c[i] = a[i] + b[i];
}
```

- [ ] **Step 2: Write `vk_compute.h`**

```c
#ifndef VK_COMPUTE_H
#define VK_COMPUTE_H

#include <stdint.h>
#include <stddef.h>

typedef struct VkCtx VkCtx;
typedef struct VkBuf VkBuf;
typedef struct VkPipe VkPipe;
typedef struct VkCmd VkCmd;

// Lifecycle
VkCtx* vk_create(void);
void   vk_destroy(VkCtx* ctx);

// Buffers — HOST_VISIBLE | DEVICE_LOCAL on unified memory
VkBuf* vk_buf_create(VkCtx* ctx, size_t bytes);
VkBuf* vk_buf_create_from_data(VkCtx* ctx, const void* src, size_t bytes);
void*  vk_buf_map(VkBuf* buf);
size_t vk_buf_size(VkBuf* buf);
void   vk_buf_destroy(VkCtx* ctx, VkBuf* buf);

// Pipelines — one per SPIR-V shader
VkPipe* vk_pipe_create(VkCtx* ctx, const char* spv_path,
                        uint32_t push_const_size, uint32_t num_bindings);
void    vk_pipe_destroy(VkCtx* ctx, VkPipe* pipe);

// Command buffers
VkCmd* vk_cmd_begin(VkCtx* ctx);
void   vk_cmd_bind(VkCmd* cmd, VkPipe* pipe, VkBuf** bufs, const size_t* offsets,
                    const size_t* ranges, uint32_t num_bufs,
                    const void* push_consts, uint32_t push_size);
void   vk_cmd_dispatch(VkCmd* cmd, uint32_t gx, uint32_t gy, uint32_t gz);
void   vk_cmd_barrier(VkCmd* cmd);
void   vk_cmd_submit(VkCmd* cmd);        // submit + fence wait
void   vk_cmd_submit_no_wait(VkCmd* cmd); // submit, no wait
void   vk_cmd_wait(VkCmd* cmd);           // wait for pending submission
void   vk_cmd_reset(VkCmd* cmd);          // reset for reuse
void   vk_cmd_destroy(VkCtx* ctx, VkCmd* cmd);

// Internal Vulkan handle access (needed for descriptor writes)
// These return opaque Vulkan handles cast to uint64_t
uint64_t vk_buf_handle(VkBuf* buf);

#endif // VK_COMPUTE_H
```

- [ ] **Step 3: Write `vk_compute.c`**

Key implementation details:
- Use `#define VOLK_IMPLEMENTATION` and `#include "deps/volk.h"` for Vulkan function loading.
- Use `#define VMA_IMPLEMENTATION`, `#define VMA_STATIC_VULKAN_FUNCTIONS 0`, `#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1` before `#include "deps/vk_mem_alloc.h"`.
- `vk_create()`: call `volkInitialize()`, create VkInstance (no validation in release), enumerate physical devices (prefer integrated GPU on UMA — `VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU`), create logical device with one compute queue, create VmaAllocator with `VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT`.
- `vk_buf_create()`: `VmaAllocationCreateInfo` with `VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT`, usage `VMA_MEMORY_USAGE_AUTO`. VMA picks HOST_VISIBLE+DEVICE_LOCAL on UMA.
- `vk_pipe_create()`: Read SPIR-V from file, create `VkShaderModule`, create `VkDescriptorSetLayout` with `num_bindings` storage buffer bindings, create `VkPipelineLayout` with push constant range, create `VkComputePipeline`. Cache in `VkPipelineCache` (saved to disk at `vk_destroy`).
- `vk_cmd_begin()`: Allocate `VkCommandBuffer` from pool, begin recording.
- `vk_cmd_bind()`: Allocate descriptor set from pool. For each buffer, create a `VkDescriptorBufferInfo` with the buffer handle, the caller-provided offset, and range (or `VK_WHOLE_SIZE` if range is 0). This enables binding sub-regions of the 5.5GB weight VkBuf as individual tensor descriptors. Write descriptor set, bind pipeline and descriptor set. Note: offsets must be multiples of `minStorageBufferOffsetAlignment` (query from physical device properties, typically 16 or 64 bytes on RADV).
- `vk_cmd_dispatch()`: `vkCmdDispatch`.
- `vk_cmd_barrier()`: `vkCmdPipelineBarrier2` with src=`VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_ACCESS_2_SHADER_WRITE_BIT`, dst=`VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_ACCESS_2_SHADER_READ_BIT`.
- `vk_cmd_submit()`: End recording, submit with fence, wait on fence.
- `vk_cmd_submit_no_wait()`: End recording, submit with fence, don't wait.
- `vk_cmd_wait()`: Wait on fence.
- Descriptor pool: Pre-allocate a large pool (e.g. 1024 sets, 8192 storage buffer descriptors). Reset between tokens.

Full implementation is ~400-500 lines of C.

- [ ] **Step 4: Write a test program to validate**

Create `vulkan_infer/test_vk.c`:
```c
#include "vk_compute.h"
#include <stdio.h>
#include <math.h>

int main(void) {
    VkCtx* ctx = vk_create();
    if (!ctx) { fprintf(stderr, "Failed to create Vulkan context\n"); return 1; }

    // Test: a + b = c for 1024 floats
    uint32_t n = 1024;
    VkBuf* a = vk_buf_create(ctx, n * sizeof(float));
    VkBuf* b = vk_buf_create(ctx, n * sizeof(float));
    VkBuf* c = vk_buf_create(ctx, n * sizeof(float));

    float* ap = vk_buf_map(a);
    float* bp = vk_buf_map(b);
    for (uint32_t i = 0; i < n; i++) { ap[i] = (float)i; bp[i] = (float)(i * 2); }

    VkPipe* pipe = vk_pipe_create(ctx, "shaders/test_add.spv", sizeof(uint32_t), 3);
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
        if (fabsf(cp[i] - expected) > 1e-5f) errors++;
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
```

Add to Makefile:
```makefile
test_vk: test_vk.c vk_compute.c vk_compute.h shaders/test_add.spv
	$(CC) $(CFLAGS) test_vk.c vk_compute.c deps/volk.c -o test_vk $(LDFLAGS)
```

- [ ] **Step 5: Build and run test**

```bash
cd vulkan_infer
make shaders
make test_vk
./test_vk
```

Expected: `test_add: 1024/1024 passed`

- [ ] **Step 6: Commit**

```bash
git add vulkan_infer/vk_compute.h vulkan_infer/vk_compute.c vulkan_infer/shaders/test_add.comp vulkan_infer/test_vk.c
git commit -m "feat(vulkan): Vulkan compute abstraction with volk+VMA"
```

---

### Task 3: io_uring wrapper (`io_ring.h` / `io_ring.c`)

**Files:**
- Create: `vulkan_infer/io_ring.h`
- Create: `vulkan_infer/io_ring.c`

- [ ] **Step 1: Write `io_ring.h`**

```c
#ifndef IO_RING_H
#define IO_RING_H

#include <stdint.h>
#include <stddef.h>

typedef struct IoRing IoRing;

IoRing* io_ring_create(uint32_t queue_depth);
void    io_ring_destroy(IoRing* ring);

// Batch-submit count reads and wait for all completions.
// offsets[i]: byte offset into the file for read i
// dests[i]: destination buffer pointer (must be valid writable memory)
// sizes[i]: number of bytes to read
// Returns 0 on success, -1 on error.
int io_ring_read_experts(IoRing* ring,
                         int fd,
                         const uint64_t* offsets,
                         void** dests,
                         const size_t* sizes,
                         int count);

#endif // IO_RING_H
```

- [ ] **Step 2: Write `io_ring.c`**

```c
#include "io_ring.h"
#include <liburing.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

struct IoRing {
    struct io_uring ring;
    uint32_t depth;
};

IoRing* io_ring_create(uint32_t queue_depth) {
    IoRing* r = calloc(1, sizeof(IoRing));
    if (!r) return NULL;
    r->depth = queue_depth;
    int ret = io_uring_queue_init(queue_depth, &r->ring, 0);
    if (ret < 0) {
        fprintf(stderr, "io_uring_queue_init failed: %s\n", strerror(-ret));
        free(r);
        return NULL;
    }
    return r;
}

void io_ring_destroy(IoRing* ring) {
    if (!ring) return;
    io_uring_queue_exit(&ring->ring);
    free(ring);
}

int io_ring_read_experts(IoRing* ring, int fd,
                         const uint64_t* offsets, void** dests,
                         const size_t* sizes, int count) {
    // Submit all reads
    for (int i = 0; i < count; i++) {
        struct io_uring_sqe* sqe = io_uring_get_sqe(&ring->ring);
        if (!sqe) {
            fprintf(stderr, "io_uring: failed to get SQE\n");
            return -1;
        }
        io_uring_prep_read(sqe, fd, dests[i], sizes[i], offsets[i]);
        io_uring_sqe_set_data(sqe, (void*)(intptr_t)i);
    }

    int ret = io_uring_submit(&ring->ring);
    if (ret < 0) {
        fprintf(stderr, "io_uring_submit failed: %s\n", strerror(-ret));
        return -1;
    }

    // Wait for all completions
    for (int i = 0; i < count; i++) {
        struct io_uring_cqe* cqe;
        ret = io_uring_wait_cqe(&ring->ring, &cqe);
        if (ret < 0) {
            fprintf(stderr, "io_uring_wait_cqe failed: %s\n", strerror(-ret));
            return -1;
        }
        if (cqe->res < 0) {
            fprintf(stderr, "io_uring read %d failed: %s\n", i, strerror(-cqe->res));
            io_uring_cqe_seen(&ring->ring, cqe);
            return -1;
        }
        // Check for short reads (cqe->res < expected size)
        int idx = (int)(intptr_t)io_uring_cqe_get_data(cqe);
        if ((size_t)cqe->res < sizes[idx]) {
            fprintf(stderr, "io_uring short read %d: got %d of %zu bytes\n",
                    idx, cqe->res, sizes[idx]);
            io_uring_cqe_seen(&ring->ring, cqe);
            return -1;  // For 7MB expert reads, short reads indicate a real problem
        }
        io_uring_cqe_seen(&ring->ring, cqe);
    }
    return 0;
}
```

- [ ] **Step 3: Write test**

Create `vulkan_infer/test_io.c` that writes a temp file with known data, reads it back via io_ring, verifies contents.

```c
#include "io_ring.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>

int main(void) {
    // Create temp file with known pattern
    char tmppath[] = "/tmp/test_io_XXXXXX";
    int fd = mkstemp(tmppath);
    size_t sz = 4 * 7077888; // 4 experts worth
    char* data = malloc(sz);
    for (size_t i = 0; i < sz; i++) data[i] = (char)(i & 0xFF);
    write(fd, data, sz);

    IoRing* ring = io_ring_create(16);
    if (!ring) { fprintf(stderr, "failed to create io_ring\n"); return 1; }

    // Read 4 "experts" in parallel
    char* bufs[4];
    uint64_t offsets[4];
    size_t sizes[4];
    for (int i = 0; i < 4; i++) {
        bufs[i] = malloc(7077888);
        offsets[i] = (uint64_t)i * 7077888;
        sizes[i] = 7077888;
    }

    int ret = io_ring_read_experts(ring, fd, offsets, (void**)bufs, sizes, 4);
    if (ret != 0) { fprintf(stderr, "read failed\n"); return 1; }

    // Verify
    int errors = 0;
    for (int i = 0; i < 4; i++) {
        if (memcmp(bufs[i], data + offsets[i], sizes[i]) != 0) {
            fprintf(stderr, "expert %d mismatch\n", i);
            errors++;
        }
    }
    printf("test_io: %d/4 experts read correctly\n", 4 - errors);

    for (int i = 0; i < 4; i++) free(bufs[i]);
    free(data);
    io_ring_destroy(ring);
    close(fd);
    unlink(tmppath);
    return errors > 0 ? 1 : 0;
}
```

- [ ] **Step 4: Build and run test**

```bash
make test_io
./test_io
```

Expected: `test_io: 4/4 experts read correctly`

- [ ] **Step 5: Commit**

```bash
git add vulkan_infer/io_ring.h vulkan_infer/io_ring.c vulkan_infer/test_io.c
git commit -m "feat(vulkan): io_uring wrapper for expert streaming"
```

---

### Task 4: dequant_matvec_4bit compute shader

**Files:**
- Create: `vulkan_infer/shaders/dequant_matvec_4bit.comp`

This is the most critical shader — used for all projection matvecs and lm_head.

- [ ] **Step 1: Write the shader**

Port `dequant_matvec_4bit_v3` from `metal_infer/shaders.metal:251-340`. Key translation:

```glsl
#version 450
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(local_size_x = 256) in; // 8 subgroups of 32

layout(binding = 0) readonly buffer WPacked { uint w_packed[]; };
layout(binding = 1) readonly buffer Scales  { uint scales_raw[]; };  // bf16 packed as uint16 pairs
layout(binding = 2) readonly buffer Biases  { uint biases_raw[]; };  // bf16 packed as uint16 pairs
layout(binding = 3) readonly buffer X       { float x[]; };
layout(binding = 4) writeonly buffer Out    { float out_data[]; };

layout(push_constant) uniform Params {
    uint out_dim;
    uint in_dim;
    uint group_size;
};

#define ROWS_PER_TG 8

// bf16 -> f32: shift left 16 bits
float bf16_to_f32(uint raw_pair, uint which) {
    uint bf16 = (which == 0u) ? (raw_pair & 0xFFFFu) : (raw_pair >> 16u);
    return uintBitsToFloat(bf16 << 16u);
}

shared float x_shared[4096];

void main() {
    uint lid = gl_LocalInvocationID.x;
    uint tgid = gl_WorkGroupID.x;
    uint simd_lane = gl_SubgroupInvocationID;
    uint simd_group = gl_SubgroupID;

    uint row = tgid * ROWS_PER_TG + simd_group;
    uint packed_cols = in_dim / 8u;
    uint num_groups = in_dim / group_size;

    // Cooperative load of input vector into shared memory
    for (uint i = lid; i < in_dim; i += 256u) {
        x_shared[i] = x[i];
    }
    barrier();

    if (row >= out_dim) return;

    // Row pointers
    uint w_row_base = row * packed_cols;
    uint s_row_base = row * num_groups;

    float acc = 0.0;

    for (uint col = simd_lane; col < packed_cols; col += 32u) {
        uint g = col / (group_size / 8u);

        // Read bf16 scale and bias (packed as pairs in uint)
        uint s_idx = s_row_base + g;
        float scale = bf16_to_f32(scales_raw[s_idx / 2u], s_idx & 1u);
        float bias  = bf16_to_f32(biases_raw[s_idx / 2u], s_idx & 1u);

        uint packed = w_packed[w_row_base + col];
        uint x_base = col * 8u;

        // FMA optimization: precompute scale*x and bias*x per group element
        float sx0 = scale * x_shared[x_base + 0u]; float bx0 = bias * x_shared[x_base + 0u];
        float sx1 = scale * x_shared[x_base + 1u]; float bx1 = bias * x_shared[x_base + 1u];
        float sx2 = scale * x_shared[x_base + 2u]; float bx2 = bias * x_shared[x_base + 2u];
        float sx3 = scale * x_shared[x_base + 3u]; float bx3 = bias * x_shared[x_base + 3u];
        float sx4 = scale * x_shared[x_base + 4u]; float bx4 = bias * x_shared[x_base + 4u];
        float sx5 = scale * x_shared[x_base + 5u]; float bx5 = bias * x_shared[x_base + 5u];
        float sx6 = scale * x_shared[x_base + 6u]; float bx6 = bias * x_shared[x_base + 6u];
        float sx7 = scale * x_shared[x_base + 7u]; float bx7 = bias * x_shared[x_base + 7u];

        acc += fma(float((packed >>  0u) & 0xFu), sx0, bx0);
        acc += fma(float((packed >>  4u) & 0xFu), sx1, bx1);
        acc += fma(float((packed >>  8u) & 0xFu), sx2, bx2);
        acc += fma(float((packed >> 12u) & 0xFu), sx3, bx3);
        acc += fma(float((packed >> 16u) & 0xFu), sx4, bx4);
        acc += fma(float((packed >> 20u) & 0xFu), sx5, bx5);
        acc += fma(float((packed >> 24u) & 0xFu), sx6, bx6);
        acc += fma(float((packed >> 28u) & 0xFu), sx7, bx7);
    }

    // Subgroup reduction
    float sum = subgroupAdd(acc);

    if (simd_lane == 0u) {
        out_data[row] = sum;
    }
}
```

**Important note on bf16 layout:** The Metal code reads `uint16_t` scales/biases directly. In GLSL we must use `uint` (32-bit) storage buffers and extract the 16-bit halves. The scales/biases arrays are stored as contiguous `uint16_t` values, so we read pairs as `uint32` and extract via bit masking. The indexing is `scales_raw[s_idx / 2]` to get the pair, then `s_idx & 1` selects low or high half.

- [ ] **Step 2: Compile the shader**

```bash
cd vulkan_infer
glslc --target-env=vulkan1.3 -O shaders/dequant_matvec_4bit.comp -o shaders/dequant_matvec_4bit.spv
```

Expected: clean compilation.

- [ ] **Step 3: Write a CPU reference and test**

Create `vulkan_infer/test_matvec.c`:

- Generate random 4-bit packed weights, bf16 scales/biases, and float input vector
- Compute reference result on CPU using the same dequant formula
- Run the GPU shader via vk_compute
- Compare results with tolerance (< 1e-3 relative error)
- Test with dimensions matching actual model: (1024, 4096, 64), (4096, 1024, 64), (248320, 4096, 64)

The CPU reference function:
```c
static void cpu_dequant_matvec_4bit(const uint32_t* W, const uint16_t* scales,
                                     const uint16_t* biases, const float* x,
                                     float* out, uint32_t out_dim, uint32_t in_dim,
                                     uint32_t group_size) {
    uint32_t packed_cols = in_dim / 8;
    uint32_t num_groups = in_dim / group_size;
    uint32_t packed_per_group = group_size / 8;
    for (uint32_t row = 0; row < out_dim; row++) {
        float acc = 0.0f;
        for (uint32_t g = 0; g < num_groups; g++) {
            // bf16 to f32
            uint16_t s_raw = scales[row * num_groups + g];
            uint16_t b_raw = biases[row * num_groups + g];
            float scale, bias;
            uint32_t tmp;
            tmp = (uint32_t)s_raw << 16; memcpy(&scale, &tmp, 4);
            tmp = (uint32_t)b_raw << 16; memcpy(&bias, &tmp, 4);

            for (uint32_t p = 0; p < packed_per_group; p++) {
                uint32_t packed = W[row * packed_cols + g * packed_per_group + p];
                for (uint32_t n = 0; n < 8; n++) {
                    uint32_t nibble = (packed >> (n * 4)) & 0xF;
                    acc += ((float)nibble * scale + bias) * x[g * group_size + p * 8 + n];
                }
            }
        }
        out[row] = acc;
    }
}
```

- [ ] **Step 4: Build and run test**

```bash
make test_matvec
./test_matvec
```

Expected: all dimension configurations pass within tolerance.

- [ ] **Step 5: Commit**

```bash
git add vulkan_infer/shaders/dequant_matvec_4bit.comp vulkan_infer/test_matvec.c
git commit -m "feat(vulkan): FMA-optimized 4-bit dequant matvec compute shader"
```

---

### Task 5: RMS norm, residual_add, sigmoid_gate shaders

**Files:**
- Create: `vulkan_infer/shaders/rms_norm.comp`
- Create: `vulkan_infer/shaders/residual_add.comp`
- Create: `vulkan_infer/shaders/sigmoid_gate.comp`

- [ ] **Step 1: Write `rms_norm.comp`**

Two-pass in a single dispatch. Port from `metal_infer/shaders.metal:745-816`.

```glsl
#version 450
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(local_size_x = 256) in;

layout(binding = 0) readonly buffer X      { float x_in[]; };
layout(binding = 1) readonly buffer Weight { uint weight_raw[]; }; // bf16 packed pairs
layout(binding = 2) buffer Out             { float out_data[]; };

layout(push_constant) uniform Params {
    uint dim;
    float eps;
};

shared float partial_sums[8]; // one per subgroup (wave32: 256/32 = 8)

float bf16_to_f32(uint raw_pair, uint which) {
    uint bf16 = (which == 0u) ? (raw_pair & 0xFFFFu) : (raw_pair >> 16u);
    return uintBitsToFloat(bf16 << 16u);
}

void main() {
    uint tid = gl_LocalInvocationID.x;

    // Pass 1: sum of squares with two-level reduction
    float local_sq = 0.0;
    for (uint i = tid; i < dim; i += 256u) {
        float v = x_in[i];
        local_sq += v * v;
    }
    float subgroup_sq = subgroupAdd(local_sq);
    if (gl_SubgroupInvocationID == 0u) partial_sums[gl_SubgroupID] = subgroup_sq;
    barrier();

    // Thread 0 sums all partial sums
    if (tid == 0u) {
        float total = 0.0;
        uint num_subgroups = (256u + gl_SubgroupSize - 1u) / gl_SubgroupSize;
        for (uint i = 0u; i < num_subgroups; i++) total += partial_sums[i];
        partial_sums[0] = total;
    }
    barrier();

    // Pass 2: normalize
    float rms = inversesqrt(partial_sums[0] / float(dim) + eps);
    for (uint i = tid; i < dim; i += 256u) {
        float w = bf16_to_f32(weight_raw[i / 2u], i & 1u);
        out_data[i] = x_in[i] * rms * w;
    }
}
```

- [ ] **Step 2: Write `residual_add.comp`**

```glsl
#version 450
layout(local_size_x = 256) in;
layout(binding = 0) readonly buffer A  { float a[]; };
layout(binding = 1) readonly buffer B  { float b[]; };
layout(binding = 2) writeonly buffer C { float c[]; };
layout(push_constant) uniform Params { uint dim; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < dim) c[i] = a[i] + b[i];
}
```

- [ ] **Step 3: Write `sigmoid_gate.comp`**

```glsl
#version 450
layout(local_size_x = 256) in;
layout(binding = 0) buffer XOut       { float x_out[]; };
layout(binding = 1) readonly buffer G { float gate[]; };
layout(push_constant) uniform Params { uint dim; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < dim) {
        float g = 1.0 / (1.0 + exp(-gate[i]));
        x_out[i] *= g;
    }
}
```

- [ ] **Step 4: Compile all shaders**

```bash
cd vulkan_infer
make shaders
```

Expected: all `.spv` files generated.

- [ ] **Step 5: Test rms_norm with CPU reference**

CPU reference:
```c
static void cpu_rms_norm(const float* x, const uint16_t* weight, float* out,
                          uint32_t dim, float eps) {
    float sq = 0;
    for (uint32_t i = 0; i < dim; i++) sq += x[i] * x[i];
    float rms = 1.0f / sqrtf(sq / dim + eps);
    for (uint32_t i = 0; i < dim; i++) {
        uint32_t tmp = (uint32_t)weight[i] << 16;
        float w; memcpy(&w, &tmp, 4);
        out[i] = x[i] * rms * w;
    }
}
```

Test with dim=4096, tolerance 1e-4.

- [ ] **Step 6: Commit**

```bash
git add vulkan_infer/shaders/rms_norm.comp vulkan_infer/shaders/residual_add.comp vulkan_infer/shaders/sigmoid_gate.comp
git commit -m "feat(vulkan): rms_norm, residual_add, sigmoid_gate shaders"
```

---

### Task 6: fused_gate_up_swiglu and moe_combine shaders

**Files:**
- Create: `vulkan_infer/shaders/fused_gate_up_swiglu.comp`
- Create: `vulkan_infer/shaders/moe_combine.comp`

- [ ] **Step 1: Write `fused_gate_up_swiglu.comp`**

Port from `metal_infer/shaders.metal:169-218`. Fuses gate_proj + up_proj + SwiGLU.

```glsl
#version 450
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(local_size_x = 256) in;

layout(binding = 0) readonly buffer GateW { uint gate_w[]; };
layout(binding = 1) readonly buffer GateS { uint gate_s_raw[]; };
layout(binding = 2) readonly buffer GateB { uint gate_b_raw[]; };
layout(binding = 3) readonly buffer UpW   { uint up_w[]; };
layout(binding = 4) readonly buffer UpS   { uint up_s_raw[]; };
layout(binding = 5) readonly buffer UpB   { uint up_b_raw[]; };
layout(binding = 6) readonly buffer X     { float x[]; };
layout(binding = 7) writeonly buffer Out  { float out_data[]; };

layout(push_constant) uniform Params {
    uint out_dim;   // 1024 (MOE_INTERMEDIATE)
    uint in_dim;    // 4096 (HIDDEN_DIM)
    uint group_size; // 64
};

float bf16_to_f32(uint raw_pair, uint which) {
    uint bf16 = (which == 0u) ? (raw_pair & 0xFFFFu) : (raw_pair >> 16u);
    return uintBitsToFloat(bf16 << 16u);
}

shared float x_shared[4096];

void main() {
    uint lid = gl_LocalInvocationID.x;
    uint tgid = gl_WorkGroupID.x;
    uint simd_lane = gl_SubgroupInvocationID;
    uint simd_group = gl_SubgroupID;
    uint row = tgid * 8u + simd_group;

    uint packed_cols = in_dim / 8u;
    uint num_groups = in_dim / group_size;

    // Cooperative load
    for (uint i = lid; i < in_dim; i += 256u) {
        x_shared[i] = x[i];
    }
    barrier();

    if (row >= out_dim) return;

    uint w_base = row * packed_cols;
    uint s_base = row * num_groups;

    float ga = 0.0, ua = 0.0;

    for (uint col = simd_lane; col < packed_cols; col += 32u) {
        uint g = col / (group_size / 8u);
        uint s_idx = s_base + g;

        float gsc = bf16_to_f32(gate_s_raw[s_idx / 2u], s_idx & 1u);
        float gbi = bf16_to_f32(gate_b_raw[s_idx / 2u], s_idx & 1u);
        float usc = bf16_to_f32(up_s_raw[s_idx / 2u], s_idx & 1u);
        float ubi = bf16_to_f32(up_b_raw[s_idx / 2u], s_idx & 1u);

        uint gp = gate_w[w_base + col];
        uint up_val = up_w[w_base + col];
        uint x_base_idx = col * 8u;

        for (uint i = 0u; i < 8u; i++) {
            float xv = x_shared[x_base_idx + i];
            ga += (float((gp >> (i * 4u)) & 0xFu) * gsc + gbi) * xv;
            ua += (float((up_val >> (i * 4u)) & 0xFu) * usc + ubi) * xv;
        }
    }

    float rg = subgroupAdd(ga);
    float ru = subgroupAdd(ua);

    if (simd_lane == 0u) {
        // SwiGLU: silu(gate) * up
        out_data[row] = (rg / (1.0 + exp(-rg))) * ru;
    }
}
```

- [ ] **Step 2: Write `moe_combine.comp`**

Port from `metal_infer/shaders.metal:1261-1296`.

```glsl
#version 450
layout(local_size_x = 256) in;

layout(binding = 0) readonly buffer HMid      { float h_mid[]; };
layout(binding = 1) readonly buffer SharedOut { float shared_out[]; };
layout(binding = 2) writeonly buffer HidOut   { float hidden_out[]; };
layout(binding = 3) readonly buffer Exp0      { float expert_out0[]; };
layout(binding = 4) readonly buffer Exp1      { float expert_out1[]; };
layout(binding = 5) readonly buffer Exp2      { float expert_out2[]; };
layout(binding = 6) readonly buffer Exp3      { float expert_out3[]; };
layout(binding = 7) readonly buffer Params    { float params[]; };
// params[0..3] = expert weights, params[4] = shared_gate_score

layout(push_constant) uniform PC {
    uint dim;
    uint K;  // number of active experts (4)
};

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= dim) return;

    float shared_gate = 1.0 / (1.0 + exp(-params[4]));

    float moe = 0.0;
    if (K > 0u) moe += params[0] * expert_out0[i];
    if (K > 1u) moe += params[1] * expert_out1[i];
    if (K > 2u) moe += params[2] * expert_out2[i];
    if (K > 3u) moe += params[3] * expert_out3[i];

    hidden_out[i] = h_mid[i] + moe + shared_gate * shared_out[i];
}
```

**Note on expert forward pass pipeline:** Each expert's forward pass is:
1. `fused_gate_up_swiglu.comp` (gate+up projections [4096->1024] + SwiGLU activation)
2. `dequant_matvec_4bit.comp` reused for down_proj [1024->4096] with `out_dim=4096, in_dim=1024, group_size=64`

The down_proj uses the same `dequant_matvec_4bit.comp` shader — no separate shader needed.

- [ ] **Step 3: Compile and test**

```bash
make shaders
```

Test `fused_gate_up_swiglu` with known inputs against CPU reference (separate gate_matvec + up_matvec + swiglu). Test `moe_combine` with random expert outputs and known weights.

- [ ] **Step 4: Commit**

```bash
git add vulkan_infer/shaders/fused_gate_up_swiglu.comp vulkan_infer/shaders/moe_combine.comp
git commit -m "feat(vulkan): fused_gate_up_swiglu and moe_combine shaders"
```

---

### Task 7: Attention shaders

**Files:**
- Create: `vulkan_infer/shaders/attn_scores.comp`
- Create: `vulkan_infer/shaders/attn_softmax.comp`
- Create: `vulkan_infer/shaders/attn_values.comp`

- [ ] **Step 1: Write `attn_scores.comp`**

Port from `metal_infer/shaders.metal:852-895`. Q @ K^T with GQA mapping.

```glsl
#version 450
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(local_size_x = 256) in;

layout(binding = 0) readonly buffer Q      { float q[]; };
layout(binding = 1) readonly buffer KCache { float k_cache[]; };
layout(binding = 2) writeonly buffer Scores { float scores[]; };

layout(push_constant) uniform Params {
    uint head_dim;      // 256
    uint kv_dim;        // 512
    uint seq_len;
    uint seq_stride;    // MAX_SEQ_LEN
    float scale;        // 1/sqrt(head_dim)
    uint heads_per_kv;  // 16
    uint num_seq_tgs;   // = seq_len
};

void main() {
    uint tgid = gl_WorkGroupID.x;
    uint lid = gl_LocalInvocationID.x;
    uint tg_size = gl_WorkGroupSize.x;

    uint pos = tgid % num_seq_tgs;
    uint h = tgid / num_seq_tgs;
    if (pos >= seq_len) return;

    uint kv_h = h / heads_per_kv;
    uint q_base = h * head_dim;
    uint k_base = pos * kv_dim + kv_h * head_dim;

    float acc = 0.0;
    for (uint d = lid; d < head_dim; d += tg_size) {
        acc += q[q_base + d] * k_cache[k_base + d];
    }

    float sum = subgroupAdd(acc);
    // Final reduction across subgroups
    shared float partial[8];
    if (gl_SubgroupInvocationID == 0u) partial[gl_SubgroupID] = sum;
    barrier();
    // All lanes of subgroup 0 participate; lanes >= num_subgroups load 0
    if (gl_SubgroupID == 0u) {
        uint num_sg = (tg_size + gl_SubgroupSize - 1u) / gl_SubgroupSize;
        float v = (gl_SubgroupInvocationID < num_sg) ? partial[gl_SubgroupInvocationID] : 0.0;
        v = subgroupAdd(v);
        if (gl_SubgroupInvocationID == 0u) {
            scores[h * seq_stride + pos] = v * scale;
        }
    }
}
```

- [ ] **Step 2: Write `attn_softmax.comp`**

Port from `metal_infer/shaders.metal:902-960`. Three-pass: max, exp+sum, normalize.

```glsl
#version 450
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(local_size_x = 256) in;

layout(binding = 0) buffer Scores { float scores[]; };

layout(push_constant) uniform Params {
    uint seq_len;
    uint seq_stride;
};

shared float shared_vals[8];
shared float broadcast_val;

void main() {
    uint head = gl_WorkGroupID.x;
    uint lid = gl_LocalInvocationID.x;
    uint tg_size = gl_WorkGroupSize.x;
    uint base = head * seq_stride;

    // Pass 1: find max
    float local_max = -1e30;
    for (uint i = lid; i < seq_len; i += tg_size) {
        local_max = max(local_max, scores[base + i]);
    }
    float sm = subgroupMax(local_max);
    if (gl_SubgroupInvocationID == 0u) shared_vals[gl_SubgroupID] = sm;
    barrier();
    if (lid == 0u) {
        float m = -1e30;
        uint ns = (tg_size + gl_SubgroupSize - 1u) / gl_SubgroupSize;
        for (uint i = 0u; i < ns; i++) m = max(m, shared_vals[i]);
        broadcast_val = m;
    }
    barrier();
    float global_max = broadcast_val;

    // Pass 2: exp and sum
    float local_sum = 0.0;
    for (uint i = lid; i < seq_len; i += tg_size) {
        float val = exp(scores[base + i] - global_max);
        scores[base + i] = val;
        local_sum += val;
    }
    float ss = subgroupAdd(local_sum);
    if (gl_SubgroupInvocationID == 0u) shared_vals[gl_SubgroupID] = ss;
    barrier();
    if (lid == 0u) {
        float s = 0.0;
        uint ns = (tg_size + gl_SubgroupSize - 1u) / gl_SubgroupSize;
        for (uint i = 0u; i < ns; i++) s += shared_vals[i];
        broadcast_val = s;
    }
    barrier();

    // Pass 3: normalize
    float inv_sum = 1.0 / broadcast_val;
    for (uint i = lid; i < seq_len; i += tg_size) {
        scores[base + i] *= inv_sum;
    }
}
```

- [ ] **Step 3: Write `attn_values.comp`**

Port from `metal_infer/shaders.metal:971-993`. Scores @ V.

```glsl
#version 450
layout(local_size_x = 256) in;

layout(binding = 0) readonly buffer Scores { float scores[]; };
layout(binding = 1) readonly buffer VCache { float v_cache[]; };
layout(binding = 2) writeonly buffer Out   { float out_data[]; };

layout(push_constant) uniform Params {
    uint head_dim;
    uint kv_dim;
    uint seq_len;
    uint seq_stride;
    uint heads_per_kv;
};

void main() {
    uint tid = gl_GlobalInvocationID.x;
    uint d = tid % head_dim;
    uint h = tid / head_dim;

    uint kv_h = h / heads_per_kv;
    uint s_base = h * seq_stride;

    float acc = 0.0;
    for (uint p = 0u; p < seq_len; p++) {
        acc += scores[s_base + p] * v_cache[p * kv_dim + kv_h * head_dim + d];
    }
    out_data[h * head_dim + d] = acc;
}
```

- [ ] **Step 4: Compile and test**

```bash
make shaders
```

Test with small dimensions (4 heads, head_dim=8, seq_len=4) against CPU reference.

- [ ] **Step 5: Commit**

```bash
git add vulkan_infer/shaders/attn_scores.comp vulkan_infer/shaders/attn_softmax.comp vulkan_infer/shaders/attn_values.comp
git commit -m "feat(vulkan): attention compute shaders (scores, softmax, values)"
```

---

### Task 8: Weight loading and manifest parsing

**Files:**
- Create: `vulkan_infer/weights.h`
- Create: `vulkan_infer/weights.c`

- [ ] **Step 1: Write `weights.h`**

```c
#ifndef WEIGHTS_H
#define WEIGHTS_H

#include "vk_compute.h"
#include <stdint.h>

typedef struct {
    char name[256];
    size_t offset;
    size_t size;
    int shape[4];
    int ndim;
} TensorInfo;

typedef struct {
    TensorInfo* tensors;
    int count;
} TensorManifest;

typedef struct {
    VkBuf* buf;           // Single large VkBuf holding all non-expert weights
    void* mapped;         // Persistently mapped pointer
    TensorManifest* manifest;
    size_t total_size;
} WeightFile;

// Load manifest from JSON, mmap weights, copy into VkBuf
WeightFile* weights_load(VkCtx* ctx, const char* bin_path, const char* json_path);
void weights_destroy(VkCtx* ctx, WeightFile* wf);

// Look up a tensor by name. Returns NULL if not found.
TensorInfo* weights_get_tensor(WeightFile* wf, const char* name);

// Get a pointer to tensor data within the VkBuf mapped region
void* weights_tensor_data(WeightFile* wf, TensorInfo* ti);

#endif
```

- [ ] **Step 2: Write `weights.c`**

Key implementation:
- Parse `model_weights.json` using a minimal JSON parser (just find `"name"`, `"offset"`, `"size"`, `"shape"` fields). The JSON is simple enough for hand-written parsing — each entry is `{"name": "...", "offset": N, "size": N, "shape": [N, ...], "dtype": "..."}`.
- `mmap()` the `.bin` file read-only
- Allocate a VkBuf of `total_size` bytes
- `memcpy` from mmap to VkBuf mapped pointer
- `munmap` the file
- Build a simple name-to-index hash table for O(1) tensor lookup

- [ ] **Step 3: Test with actual model_weights.json format**

Read `metal_infer/model_weights.json` to understand the exact format, then test parsing.

```bash
head -20 metal_infer/model_weights.json
```

- [ ] **Step 4: Commit**

```bash
git add vulkan_infer/weights.h vulkan_infer/weights.c
git commit -m "feat(vulkan): weight loading with manifest parsing"
```

---

### Task 9: CPU-side linear attention (GatedDeltaNet)

**Files:**
- Create: `vulkan_infer/linear_attn.h`
- Create: `vulkan_infer/linear_attn.c`

This handles the 45 GatedDeltaNet layers on CPU using OpenBLAS. Ported from the CPU fallback path in `metal_infer/infer.m`.

- [ ] **Step 1: Write `linear_attn.h`**

```c
#ifndef LINEAR_ATTN_H
#define LINEAR_ATTN_H

#include <stdint.h>

#define LINEAR_NUM_V_HEADS  64
#define LINEAR_NUM_K_HEADS  16
#define LINEAR_KEY_DIM      128
#define LINEAR_VALUE_DIM    128
#define LINEAR_TOTAL_KEY    (LINEAR_NUM_K_HEADS * LINEAR_KEY_DIM)
#define LINEAR_TOTAL_VALUE  (LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM)
#define LINEAR_CONV_DIM     (LINEAR_TOTAL_KEY * 2 + LINEAR_TOTAL_VALUE)
#define CONV_KERNEL_SIZE    4

typedef struct {
    float state[LINEAR_NUM_V_HEADS * LINEAR_KEY_DIM * LINEAR_VALUE_DIM]; // 64*128*128
    float conv_state[(CONV_KERNEL_SIZE - 1) * LINEAR_CONV_DIM];          // 3*12288
} LinearAttnState;

// Allocate and zero-initialize state for one layer
LinearAttnState* linear_attn_state_create(void);
void linear_attn_state_destroy(LinearAttnState* s);

// Run one token through the GatedDeltaNet pipeline:
// qkv_proj[12288], z_proj[8192], beta_proj[64], alpha_proj[64] -> output[8192]
// conv_weights: bf16 [conv_dim * 4]
// A_log: float [64], dt_bias: bf16 [64], gated_norm_w: bf16 [128]
void linear_attn_forward(
    LinearAttnState* state,
    const float* qkv_proj,      // [LINEAR_CONV_DIM] from CMD1 projection
    const float* z_proj,        // [LINEAR_TOTAL_VALUE] z gate
    const float* beta_proj,     // [LINEAR_NUM_V_HEADS]
    const float* alpha_proj,    // [LINEAR_NUM_V_HEADS]
    const uint16_t* conv_weights, // [LINEAR_CONV_DIM * CONV_KERNEL_SIZE] bf16
    const float* A_log,         // [LINEAR_NUM_V_HEADS]
    const uint16_t* dt_bias,    // [LINEAR_NUM_V_HEADS] bf16
    const uint16_t* gated_norm_w, // [LINEAR_VALUE_DIM] bf16
    float* output               // [LINEAR_TOTAL_VALUE]
);

#endif
```

- [ ] **Step 2: Write `linear_attn.c`**

Implementation steps matching `metal_infer/infer.m` CPU path:
1. **Conv1d with SiLU**: For each channel, dot product of history + new input with weights, SiLU activation, shift history.
2. **Split**: q = conv_out[0:2048], k = conv_out[2048:4096], v = conv_out[4096:12288]
3. **Per-head RMS normalize q and k**: Each of 16 k-heads normalized independently, q scaled by 1/key_dim, k scaled by 1/sqrt(key_dim).
4. **Compute decay and beta**: `g_decay = exp(-exp(A_log) * softplus(alpha + dt_bias))`, `beta_gate = sigmoid(beta)`.
5. **Delta-net recurrence** (per v-head, BLAS): `state *= g_decay`, `kv_mem = state @ k`, `delta = (v - kv_mem) * beta`, `state += outer(delta, k)`, `output = state @ q`.
6. **Gated RMS norm**: `output = rms_norm(values) * silu(z) * weight`.

Use `cblas_sscal`, `cblas_sgemv`, `cblas_sger` from OpenBLAS for the delta-net recurrence (step 5).

- [ ] **Step 3: Write unit test**

Test with known inputs and verify output against a Python reference or manual calculation for a single head.

- [ ] **Step 4: Commit**

```bash
git add vulkan_infer/linear_attn.h vulkan_infer/linear_attn.c
git commit -m "feat(vulkan): CPU-side GatedDeltaNet linear attention via OpenBLAS"
```

---

### Task 10: CPU-side full attention + RoPE

**Files:**
- Create: `vulkan_infer/full_attn.h`
- Create: `vulkan_infer/full_attn.c`

- [ ] **Step 1: Write `full_attn.h`**

```c
#ifndef FULL_ATTN_H
#define FULL_ATTN_H

#include "vk_compute.h"
#include <stdint.h>

#define NUM_ATTN_HEADS  32
#define NUM_KV_HEADS    2
#define HEAD_DIM        256
#define ROTARY_DIM      64
#define ROPE_THETA      10000000.0f
#define MAX_SEQ_LEN     8192

typedef struct {
    float* k_cache; // [MAX_SEQ_LEN * NUM_KV_HEADS * HEAD_DIM]
    float* v_cache; // [MAX_SEQ_LEN * NUM_KV_HEADS * HEAD_DIM]
    int seq_len;
} KVCache;

KVCache* kv_cache_create(void);
void kv_cache_destroy(KVCache* kv);

// Apply RoPE to Q and K projections, deinterleave Q, append K/V to cache,
// run attention (Q@K^T, softmax, scores@V), return attention output.
// q_proj: [NUM_ATTN_HEADS * HEAD_DIM * 2] (interleaved)
// k_proj: [NUM_KV_HEADS * HEAD_DIM]
// v_proj: [NUM_KV_HEADS * HEAD_DIM]
// output: [NUM_ATTN_HEADS * HEAD_DIM]
void full_attn_forward(
    KVCache* kv,
    const float* q_proj,  // interleaved Q from projection
    const float* k_proj,
    const float* v_proj,
    int pos,
    float* output         // [NUM_ATTN_HEADS * HEAD_DIM]
);

#endif
```

- [ ] **Step 2: Write `full_attn.c`**

Port from `metal_infer/infer.m:2025` (apply_rotary_emb) and the CPU attention path.

Key steps:
1. Deinterleave Q: The projection output is `[NUM_ATTN_HEADS * HEAD_DIM * 2]` with interleaved real/imaginary for RoPE. Deinterleave to `[NUM_ATTN_HEADS, HEAD_DIM]`.
2. Apply RoPE: Partial rotary (64 of 256 dims). `theta = ROPE_THETA`, freq = `1.0 / theta^(2i/rotary_dim)`.
3. Append K, V to cache at position `pos`.
4. For each attention head: compute Q@K^T scores (scaled by 1/sqrt(HEAD_DIM)), softmax, scores@V.
5. Return concatenated head outputs.

**Initial port uses CPU-side attention.** The GPU attention shaders from Task 7 are written and tested but not wired into `fused_layer_forward` in the initial port. The CPU path is simpler (avoids GPU-CPU-GPU data movement for RoPE) and the 15 full-attn layers are not the bottleneck. GPU attention is a follow-up optimization.

- [ ] **Step 3: Test with small known values**

- [ ] **Step 4: Commit**

```bash
git add vulkan_infer/full_attn.h vulkan_infer/full_attn.c
git commit -m "feat(vulkan): CPU-side full attention with RoPE and KV cache"
```

---

### Task 11: Main inference engine (`infer.c`)

**Files:**
- Create: `vulkan_infer/infer.c`

This is the main file tying everything together.

- [ ] **Step 1: Write model constants and CLI parsing**

```c
#include "vk_compute.h"
#include "io_ring.h"
#include "weights.h"
#include "linear_attn.h"
#include "full_attn.h"
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

// Model constants (from metal_infer/infer.m:72-127)
#define HIDDEN_DIM          4096
#define NUM_LAYERS          60
#define VOCAB_SIZE          248320
#define RMS_NORM_EPS        1e-6f
#define NUM_EXPERTS         512
#define MOE_INTERMEDIATE    1024
#define GROUP_SIZE          64
#define EXPERT_SIZE         7077888
#define FULL_ATTN_INTERVAL  4
#define EOS_TOKEN_1         248046
#define EOS_TOKEN_2         248044
#define K_EXPERTS           4

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

typedef struct {
    char* prompt;
    int max_tokens;
    int timing;
    float temperature;
    float top_p;
} InferArgs;

static InferArgs parse_args(int argc, char** argv) { ... }
```

- [ ] **Step 2: Write layer weight cache**

Build a per-layer cache of weight pointers (offsets into the WeightFile VkBuf) at startup, so each layer forward doesn't re-lookup tensors.

```c
typedef struct {
    // Full attention projections
    size_t q_w_off, q_s_off, q_b_off;
    size_t k_w_off, k_s_off, k_b_off;
    size_t v_w_off, v_s_off, v_b_off;
    // Linear attention projections
    size_t qkv_w_off, qkv_s_off, qkv_b_off;
    size_t z_w_off, z_s_off, z_b_off;
    size_t beta_w_off, beta_s_off, beta_b_off;
    size_t alpha_w_off, alpha_s_off, alpha_b_off;
    // Conv1d weights (bf16), A_log, dt_bias, gated_norm_w
    size_t conv1d_w_off, A_log_off, dt_bias_off, gated_norm_w_off;
    // Common: o_proj, input_norm, post_attn_norm, routing gate, shared expert
    size_t o_w_off, o_s_off, o_b_off;
    size_t input_norm_off, post_attn_norm_off;
    size_t gate_w_off, gate_s_off, gate_b_off; // routing gate (512 experts)
    size_t shared_gate_w_off, shared_up_w_off, shared_down_w_off; // + scales/biases
    // ... etc for all shared expert projections
    int is_full_attn;
} LayerWeights;

static LayerWeights g_layer_weights[NUM_LAYERS];

static void build_layer_cache(WeightFile* wf) {
    for (int l = 0; l < NUM_LAYERS; l++) {
        int is_full = ((l + 1) % FULL_ATTN_INTERVAL == 0);
        g_layer_weights[l].is_full_attn = is_full;
        // Look up each tensor by name pattern:
        //   "model.layers.{l}.self_attn.q_proj.weight" etc.
        // Store byte offsets into the VkBuf.
        ...
    }
}
```

- [ ] **Step 3: Write `embed_lookup`**

CPU-side, ported from `metal_infer/infer.m:2863-2908`:

```c
static void embed_lookup(WeightFile* wf, int token_id, float* out) {
    TensorInfo* w = weights_get_tensor(wf, "model.embed_tokens.weight");
    TensorInfo* s = weights_get_tensor(wf, "model.embed_tokens.scales");
    TensorInfo* b = weights_get_tensor(wf, "model.embed_tokens.biases");
    uint32_t* W = (uint32_t*)weights_tensor_data(wf, w);
    uint16_t* S = (uint16_t*)weights_tensor_data(wf, s);
    uint16_t* B = (uint16_t*)weights_tensor_data(wf, b);

    int packed_cols = w->shape[1]; // 512
    int num_groups = s->shape[1];  // 64
    int group_size_local = HIDDEN_DIM / num_groups;
    int packed_per_group = group_size_local / 8;

    const uint32_t* w_row = W + (size_t)token_id * packed_cols;
    const uint16_t* s_row = S + (size_t)token_id * num_groups;
    const uint16_t* b_row = B + (size_t)token_id * num_groups;

    for (int g = 0; g < num_groups; g++) {
        uint32_t tmp;
        float scale, bias;
        tmp = (uint32_t)s_row[g] << 16; memcpy(&scale, &tmp, 4);
        tmp = (uint32_t)b_row[g] << 16; memcpy(&bias, &tmp, 4);
        for (int p = 0; p < packed_per_group; p++) {
            uint32_t packed = w_row[g * packed_per_group + p];
            int base_idx = g * group_size_local + p * 8;
            for (int n = 0; n < 8; n++) {
                uint32_t nibble = (packed >> (n * 4)) & 0xF;
                out[base_idx + n] = (float)nibble * scale + bias;
            }
        }
    }
}
```

- [ ] **Step 4: Write `lm_head_forward`**

Uses GPU dequant_matvec for the [4096] -> [248320] projection:

```c
static void lm_head_forward(WeightFile* wf, VkCtx* ctx, VkBuf* hidden_buf,
                              VkBuf* logits_buf, VkPipe* matvec_pipe) {
    TensorInfo* w = weights_get_tensor(wf, "lm_head.weight");
    TensorInfo* s = weights_get_tensor(wf, "lm_head.scales");
    TensorInfo* b = weights_get_tensor(wf, "lm_head.biases");

    // Create sub-buffer views or use offsets...
    // Dispatch dequant_matvec with out_dim=248320, in_dim=4096, group_size=64
    uint32_t params[3] = {VOCAB_SIZE, HIDDEN_DIM, GROUP_SIZE};
    VkCmd* cmd = vk_cmd_begin(ctx);
    // Bind weight, scales, biases (from wf->buf at offsets), hidden_buf, logits_buf
    // ...
    vk_cmd_dispatch(cmd, (VOCAB_SIZE + 7) / 8, 1, 1); // ROWS_PER_TG=8
    vk_cmd_submit(cmd);
}
```

- [ ] **Step 5: Allocate scratch and expert buffers**

Allocate at startup, reused every layer/token:

```c
// Scratch buffers
VkBuf* buf_hidden   = vk_buf_create(ctx, HIDDEN_DIM * sizeof(float));
VkBuf* buf_normed   = vk_buf_create(ctx, HIDDEN_DIM * sizeof(float));
VkBuf* buf_residual = vk_buf_create(ctx, HIDDEN_DIM * sizeof(float));
VkBuf* buf_logits   = vk_buf_create(ctx, VOCAB_SIZE * sizeof(float));

// Attention projection outputs
VkBuf* buf_q_proj   = vk_buf_create(ctx, NUM_ATTN_HEADS * HEAD_DIM * 2 * sizeof(float));
VkBuf* buf_k_proj   = vk_buf_create(ctx, NUM_KV_HEADS * HEAD_DIM * sizeof(float));
VkBuf* buf_v_proj   = vk_buf_create(ctx, NUM_KV_HEADS * HEAD_DIM * sizeof(float));
VkBuf* buf_attn_out = vk_buf_create(ctx, NUM_ATTN_HEADS * HEAD_DIM * sizeof(float));

// Linear attention projection outputs
VkBuf* buf_qkv_proj = vk_buf_create(ctx, LINEAR_CONV_DIM * sizeof(float));
VkBuf* buf_z_proj   = vk_buf_create(ctx, LINEAR_TOTAL_VALUE * sizeof(float));
VkBuf* buf_beta     = vk_buf_create(ctx, LINEAR_NUM_V_HEADS * sizeof(float));
VkBuf* buf_alpha    = vk_buf_create(ctx, LINEAR_NUM_V_HEADS * sizeof(float));
VkBuf* buf_la_out   = vk_buf_create(ctx, LINEAR_TOTAL_VALUE * sizeof(float));

// Routing
VkBuf* buf_routing  = vk_buf_create(ctx, NUM_EXPERTS * sizeof(float));

// Expert data buffers: K=4 expert weight buffers (~7MB each)
VkBuf* buf_expert_data[K_EXPERTS];
for (int k = 0; k < K_EXPERTS; k++)
    buf_expert_data[k] = vk_buf_create(ctx, EXPERT_SIZE);

// Expert intermediate/output buffers
VkBuf* buf_expert_act[K_EXPERTS];   // gate+up SwiGLU output [1024]
VkBuf* buf_expert_out[K_EXPERTS];   // down_proj output [4096]
for (int k = 0; k < K_EXPERTS; k++) {
    buf_expert_act[k] = vk_buf_create(ctx, MOE_INTERMEDIATE * sizeof(float));
    buf_expert_out[k] = vk_buf_create(ctx, HIDDEN_DIM * sizeof(float));
}

// Shared expert buffers
VkBuf* buf_shared_act = vk_buf_create(ctx, MOE_INTERMEDIATE * sizeof(float));
VkBuf* buf_shared_out = vk_buf_create(ctx, HIDDEN_DIM * sizeof(float));

// MoE combine params buffer [K+1 floats: expert weights + shared gate score]
VkBuf* buf_moe_params = vk_buf_create(ctx, (K_EXPERTS + 1) * sizeof(float));
```

The descriptor pool must be reset between tokens to avoid exhaustion:
```c
// In the token generation loop, before processing each token:
vk_descriptor_pool_reset(ctx);  // Add this to vk_compute.h API
```

- [ ] **Step 6: Write `fused_layer_forward`**

The core per-layer function. All GPU work is encoded into a single VkCommandBuffer per phase, with barriers between phases. The pipeline is serial: GPU -> CPU -> GPU -> I/O -> GPU.

```c
static void fused_layer_forward(
    VkCtx* ctx, WeightFile* wf, IoRing* ring,
    int layer_idx, float* hidden, int pos, int packed_fd,
    /* all pipelines, scratch buffers, and state passed via a context struct */
    InferCtx* ictx
) {
    LayerWeights* lw = &g_layer_weights[layer_idx];
    int is_full = lw->is_full_attn;

    // 1. Input norm (GPU: rms_norm)
    //    memcpy hidden -> buf_hidden mapped pointer
    //    Dispatch rms_norm(buf_hidden, weight @ input_norm_off, buf_normed)
    //    Submit + wait

    // 2. CMD1: Attention projections (GPU: dequant_matvec x 3-4)
    //    Begin cmd, encode 3-4 matvec dispatches with barriers between
    //    Full attn: Q [4096->16384], K [4096->512], V [4096->512]
    //    Linear attn: QKV [4096->12288], Z [4096->8192], beta [4096->64], alpha [4096->64]
    //    Submit + wait, read results back from mapped buffers

    // 3. CPU attention
    //    if (is_full): apply_rotary_emb on CPU, KV cache append, CPU attention
    //    else: linear_attn_forward via OpenBLAS (conv1d, normalize, delta-net, gated norm)

    // 4. CMD2: o_proj + residual + norm + routing + shared expert
    //    o_proj matvec: attn_output [8192 or 8192] -> [4096]
    //    residual_add: o_proj_out + residual
    //    sigmoid_gate (if full attn only): output gating
    //    rms_norm: post-attn norm
    //    routing gate matvec: [4096] -> [512] expert scores
    //    shared expert fused_gate_up_swiglu: [4096] -> [1024]
    //    shared expert down_proj matvec: [1024] -> [4096]
    //    Submit + wait

    // 5. CPU routing: softmax over 512 scores, top-K=4 selection
    //    Read routing scores from GPU, softmax on CPU, select top-4 expert indices + weights

    // 6. I/O: io_ring_read_experts for K=4 active experts
    //    Compute byte offsets: expert_idx * EXPERT_SIZE
    //    Read into buf_expert_data[0..3] mapped pointers
    //    io_ring_read_experts(ring, layer_fds[layer_idx], offsets, dests, sizes, 4)

    // 7. CMD3: Expert forward + combine
    //    For each of K=4 experts:
    //      a) fused_gate_up_swiglu: bind expert data at GATE_W_OFF etc -> buf_expert_act[k]
    //      b) dequant_matvec (down_proj): bind expert data at DOWN_W_OFF etc -> buf_expert_out[k]
    //    moe_combine: weighted sum of expert_out[0..3] + shared_out + h_mid -> buf_hidden
    //    rms_norm: prepare input for next layer
    //    Submit (no wait — deferred for overlap with next layer's setup)
}
```
```

This is the most complex function. Implement step by step, testing each phase.

- [ ] **Step 7: Write `main()` with token generation loop**

```c
int main(int argc, char** argv) {
    InferArgs args = parse_args(argc, argv);

    // Init Vulkan
    VkCtx* ctx = vk_create();

    // Load weights
    WeightFile* wf = weights_load(ctx, "model_weights.bin", "model_weights.json");

    // Load tokenizer
    Tokenizer* tok = tokenizer_load("tokenizer.bin", "vocab.bin");

    // Create io_uring
    IoRing* ring = io_ring_create(16);

    // Open expert FDs
    int layer_fds[NUM_LAYERS];
    for (int l = 0; l < NUM_LAYERS; l++) {
        char path[256];
        snprintf(path, sizeof(path), "packed_experts/layer_%d.bin", l);
        layer_fds[l] = open(path, O_RDONLY);
    }

    // Create GPU pipelines
    VkPipe* matvec_pipe = vk_pipe_create(ctx, "shaders/dequant_matvec_4bit.spv", 12, 5);
    VkPipe* rms_pipe    = vk_pipe_create(ctx, "shaders/rms_norm.spv", 8, 3);
    // ... etc for all shaders

    // Allocate state
    LinearAttnState* la_states[45];
    KVCache* kv_caches[15];
    // ... init all

    // Allocate scratch buffers
    VkBuf* buf_hidden = vk_buf_create(ctx, HIDDEN_DIM * sizeof(float));
    VkBuf* buf_logits = vk_buf_create(ctx, VOCAB_SIZE * sizeof(float));
    // ... etc

    // Tokenize prompt
    int* prompt_ids = tokenizer_encode(tok, args.prompt, &prompt_len);

    // Prefill
    float* hidden = vk_buf_map(buf_hidden);
    for (int i = 0; i < prompt_len; i++) {
        embed_lookup(wf, prompt_ids[i], hidden);
        for (int l = 0; l < NUM_LAYERS; l++) {
            fused_layer_forward(ctx, wf, ring, l, hidden, i, layer_fds[l], ...);
        }
    }

    // Generate
    lm_head_forward(wf, ctx, buf_hidden, buf_logits, matvec_pipe);
    int next_token = sample(vk_buf_map(buf_logits), VOCAB_SIZE, args.temperature, args.top_p);

    for (int t = 0; t < args.max_tokens; t++) {
        if (next_token == EOS_TOKEN_1 || next_token == EOS_TOKEN_2) break;
        printf("%s", tokenizer_decode(tok, next_token));
        fflush(stdout);

        embed_lookup(wf, next_token, hidden);
        for (int l = 0; l < NUM_LAYERS; l++) {
            fused_layer_forward(ctx, wf, ring, l, hidden, prompt_len + t + 1, layer_fds[l], ...);
        }
        lm_head_forward(wf, ctx, buf_hidden, buf_logits, matvec_pipe);
        next_token = sample(vk_buf_map(buf_logits), VOCAB_SIZE, args.temperature, args.top_p);
    }

    // Cleanup
    ...
    return 0;
}
```

- [ ] **Step 8: Write sampling functions**

```c
static int sample_argmax(const float* logits, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (logits[i] > logits[best]) best = i;
    return best;
}

static int sample_top_p(const float* logits, int n, float temperature, float top_p) {
    // Temperature scaling
    // Softmax
    // Sort by probability
    // Nucleus sampling (cumulative probability <= top_p)
    // Random selection from nucleus
    ...
}
```

- [ ] **Step 9: Build and test with a stub (no model weights)**

```bash
make infer
./infer --help
```

Verify it compiles and the CLI parsing works.

- [ ] **Step 10: Commit**

```bash
git add vulkan_infer/infer.c vulkan_infer/weights.h vulkan_infer/weights.c
git commit -m "feat(vulkan): main inference engine with fused_layer_forward"
```

---

### Task 12: Integration testing with model weights

**Files:**
- Modify: `vulkan_infer/infer.c` (bug fixes from testing)

This task requires the actual model weights to be prepared.

- [ ] **Step 1: Prepare model weights**

```bash
pip install huggingface_hub safetensors numpy
huggingface-cli download Qwen/Qwen3.5-397B-A17B-4bit --local-dir ./model

cd metal_infer
python extract_weights.py --model-dir ../model
python repack_experts.py --model-dir ../model --output-dir ../vulkan_infer/packed_experts
python export_tokenizer.py --model-dir ../model

cd ../vulkan_infer
ln -s ../metal_infer/model_weights.bin .
ln -s ../metal_infer/model_weights.json .
ln -s ../metal_infer/vocab.bin .
ln -s ../metal_infer/tokenizer.bin .
```

- [ ] **Step 2: Test single-layer forward pass**

Add a `--debug-layer N` flag that runs only layer N and prints intermediate values. Verify shapes and numerical ranges are reasonable.

- [ ] **Step 3: Test full forward pass with short prompt**

```bash
./infer --prompt "Hello" --tokens 5
```

Check for crashes, NaN values, and reasonable output.

- [ ] **Step 4: Debug and fix issues**

Common issues to watch for:
- bf16 conversion errors (wrong byte packing in GLSL uint16 pairs)
- Buffer offset alignment (Vulkan requires minStorageBufferOffsetAlignment)
- Descriptor set exhaustion (need to reset pool between tokens)
- Wrong tensor names in weight lookup
- Expert I/O offset calculations

- [ ] **Step 5: Add --timing flag**

Print per-layer timing breakdown:
```
Layer  0: cmd1=3.2ms cpu_attn=0.4ms cmd2=1.5ms routing=0.0ms io=4.8ms cmd3=0.1ms total=10.0ms
Layer  1: ...
...
Total: 580ms (1.72 tok/s)
```

- [ ] **Step 6: Commit working inference**

```bash
git add -A vulkan_infer/
git commit -m "feat(vulkan): working end-to-end inference on AMD Vulkan"
```

---

### Task 13: Performance measurement and profiling

- [ ] **Step 1: Run inference and measure tok/s**

```bash
./infer --prompt "Explain quantum computing in simple terms" --tokens 50 --timing
```

Record: overall tok/s, per-phase breakdown, I/O hit rate.

- [ ] **Step 2: Identify bottlenecks**

Expected bottleneck order:
1. Expert I/O (~4-6ms/layer cold)
2. GPU matvec (~3-5ms/layer due to bandwidth)
3. CPU linear attention (~0.3-0.8ms/layer)

- [ ] **Step 3: Document results**

Add results to `vulkan_infer/RESULTS.md` with hardware specs, configuration, and measurements. Compare to the Mac M3 Max numbers.

- [ ] **Step 4: Commit**

```bash
git add vulkan_infer/RESULTS.md
git commit -m "docs: Vulkan port performance results on AMD Ryzen AI Max+ 395"
```
