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
void   vk_descriptor_pool_reset(VkCtx* ctx); // Reset between tokens

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
// offsets/ranges: per-buffer byte offset and range into the VkBuf.
// Pass NULL for offsets to use 0 for all. Pass NULL for ranges to use VK_WHOLE_SIZE.
void   vk_cmd_bind(VkCmd* cmd, VkPipe* pipe, VkBuf** bufs, const size_t* offsets,
                    const size_t* ranges, uint32_t num_bufs,
                    const void* push_consts, uint32_t push_size);
void   vk_cmd_dispatch(VkCmd* cmd, uint32_t gx, uint32_t gy, uint32_t gz);
void   vk_cmd_barrier(VkCmd* cmd);  // compute->compute memory barrier
void   vk_cmd_submit(VkCmd* cmd);         // submit + fence wait
void   vk_cmd_submit_no_wait(VkCmd* cmd); // submit, no wait (deferred)
void   vk_cmd_wait(VkCmd* cmd);           // wait for pending submission
void   vk_cmd_reset(VkCmd* cmd);          // reset for reuse
void   vk_cmd_destroy(VkCtx* ctx, VkCmd* cmd);

#endif // VK_COMPUTE_H
