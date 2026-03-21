/*
 * vk_compute.c — Vulkan compute abstraction for Flash-MoE inference
 *
 * Pure C, no VMA dependency. Uses raw Vulkan allocation targeting UMA
 * (unified memory architecture) where HOST_VISIBLE | DEVICE_LOCAL memory
 * is available. All buffers are persistently mapped.
 *
 * Dependencies: volk (dynamic Vulkan loader)
 */

#include "vk_compute.h"
#include "deps/volk.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// ---- Internal structs ----

struct VkCtx {
    VkInstance       instance;
    VkPhysicalDevice physical_device;
    VkDevice         device;
    VkQueue          queue;
    uint32_t         queue_family;
    VkCommandPool    cmd_pool;
    VkDescriptorPool desc_pool;
    VkPipelineCache  pipe_cache;
    uint32_t         mem_type_index;  // HOST_VISIBLE | DEVICE_LOCAL
    uint32_t         min_storage_buffer_offset_alignment;
};

struct VkBuf {
    VkBuffer       buffer;
    VkDeviceMemory memory;
    void*          mapped;
    size_t         size;
};

struct VkPipe {
    VkPipeline            pipeline;
    VkPipelineLayout      layout;
    VkDescriptorSetLayout desc_layout;
    uint32_t              push_const_size;
    uint32_t              num_bindings;
};

struct VkCmd {
    VkCommandBuffer cmd_buf;
    VkFence         fence;
    VkCtx*          ctx;
    int             submitted;
};

// ---- Helpers ----

#define VK_CHECK(call) do { \
    VkResult _r = (call); \
    if (_r != VK_SUCCESS) { \
        fprintf(stderr, "Vulkan error %d at %s:%d\n", _r, __FILE__, __LINE__); \
        abort(); \
    } \
} while (0)

static const char* PIPE_CACHE_PATH = "pipeline_cache.bin";

static uint32_t find_memory_type(VkPhysicalDevice pdev, uint32_t type_filter,
                                  VkMemoryPropertyFlags required) {
    VkPhysicalDeviceMemoryProperties props;
    vkGetPhysicalDeviceMemoryProperties(pdev, &props);

    // First pass: try to find HOST_VISIBLE | DEVICE_LOCAL | HOST_COHERENT
    for (uint32_t i = 0; i < props.memoryTypeCount; i++) {
        if ((type_filter & (1u << i)) &&
            (props.memoryTypes[i].propertyFlags & (required | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
                == (required | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            return i;
        }
    }
    // Second pass: without HOST_COHERENT
    for (uint32_t i = 0; i < props.memoryTypeCount; i++) {
        if ((type_filter & (1u << i)) &&
            (props.memoryTypes[i].propertyFlags & required) == required) {
            return i;
        }
    }
    fprintf(stderr, "Failed to find suitable memory type\n");
    abort();
}

static uint32_t find_compute_queue_family(VkPhysicalDevice pdev) {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pdev, &count, NULL);
    VkQueueFamilyProperties* props = calloc(count, sizeof(*props));
    vkGetPhysicalDeviceQueueFamilyProperties(pdev, &count, props);

    // Prefer compute-only queue
    for (uint32_t i = 0; i < count; i++) {
        if ((props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            !(props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            free(props);
            return i;
        }
    }
    // Fall back to any compute queue
    for (uint32_t i = 0; i < count; i++) {
        if (props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            free(props);
            return i;
        }
    }
    free(props);
    fprintf(stderr, "No compute queue family found\n");
    abort();
}

static void* read_file(const char* path, size_t* out_size) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return NULL; }
    void* data = malloc((size_t)sz);
    size_t rd = fread(data, 1, (size_t)sz, f);
    fclose(f);
    if (rd != (size_t)sz) { free(data); return NULL; }
    *out_size = (size_t)sz;
    return data;
}

// ---- Lifecycle ----

VkCtx* vk_create(void) {
    VK_CHECK(volkInitialize());

    VkCtx* ctx = calloc(1, sizeof(VkCtx));

    // Instance
    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "flash-moe",
        .apiVersion = VK_API_VERSION_1_3,
    };
    VkInstanceCreateInfo inst_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info,
    };
    VK_CHECK(vkCreateInstance(&inst_info, NULL, &ctx->instance));
    volkLoadInstance(ctx->instance);

    // Physical device — prefer integrated GPU (UMA)
    uint32_t dev_count = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(ctx->instance, &dev_count, NULL));
    if (dev_count == 0) {
        fprintf(stderr, "No Vulkan devices found\n");
        free(ctx);
        return NULL;
    }
    VkPhysicalDevice* devs = calloc(dev_count, sizeof(VkPhysicalDevice));
    VK_CHECK(vkEnumeratePhysicalDevices(ctx->instance, &dev_count, devs));

    ctx->physical_device = devs[0]; // default
    for (uint32_t i = 0; i < dev_count; i++) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devs[i], &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
            ctx->physical_device = devs[i];
            fprintf(stderr, "Selected GPU: %s (integrated)\n", props.deviceName);
            break;
        }
        if (i == dev_count - 1) {
            VkPhysicalDeviceProperties sel_props;
            vkGetPhysicalDeviceProperties(ctx->physical_device, &sel_props);
            fprintf(stderr, "Selected GPU: %s\n", sel_props.deviceName);
        }
    }
    free(devs);

    // Query limits
    VkPhysicalDeviceProperties dev_props;
    vkGetPhysicalDeviceProperties(ctx->physical_device, &dev_props);
    ctx->min_storage_buffer_offset_alignment =
        (uint32_t)dev_props.limits.minStorageBufferOffsetAlignment;

    // Compute queue
    ctx->queue_family = find_compute_queue_family(ctx->physical_device);

    // Logical device with Vulkan 1.3 features (synchronization2, etc.)
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = ctx->queue_family,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority,
    };

    VkPhysicalDeviceVulkan13Features features13 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .synchronization2 = VK_TRUE,
        .maintenance4 = VK_TRUE,
    };

    VkPhysicalDeviceVulkan11Features features11 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
        .pNext = &features13,
        .storageBuffer16BitAccess = VK_FALSE,
    };

    VkPhysicalDeviceFeatures2 features2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &features11,
    };

    VkDeviceCreateInfo dev_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &features2,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_info,
    };
    VK_CHECK(vkCreateDevice(ctx->physical_device, &dev_info, NULL, &ctx->device));
    volkLoadDevice(ctx->device);
    vkGetDeviceQueue(ctx->device, ctx->queue_family, 0, &ctx->queue);

    // Find UMA memory type: HOST_VISIBLE | DEVICE_LOCAL
    // Create a dummy buffer to get memory requirements
    VkBufferCreateInfo dummy_buf_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = 256,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VkBuffer dummy_buf;
    VK_CHECK(vkCreateBuffer(ctx->device, &dummy_buf_info, NULL, &dummy_buf));
    VkMemoryRequirements mem_req;
    vkGetBufferMemoryRequirements(ctx->device, dummy_buf, &mem_req);
    ctx->mem_type_index = find_memory_type(ctx->physical_device, mem_req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkDestroyBuffer(ctx->device, dummy_buf, NULL);

    // Command pool
    VkCommandPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = ctx->queue_family,
    };
    VK_CHECK(vkCreateCommandPool(ctx->device, &pool_info, NULL, &ctx->cmd_pool));

    // Descriptor pool
    VkDescriptorPoolSize pool_size = {
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 32768,
    };
    VkDescriptorPoolCreateInfo desc_pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 4096,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_size,
    };
    VK_CHECK(vkCreateDescriptorPool(ctx->device, &desc_pool_info, NULL, &ctx->desc_pool));

    // Pipeline cache — load from disk if available
    VkPipelineCacheCreateInfo cache_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
    };
    size_t cache_size = 0;
    void* cache_data = read_file(PIPE_CACHE_PATH, &cache_size);
    if (cache_data) {
        cache_info.initialDataSize = cache_size;
        cache_info.pInitialData = cache_data;
    }
    VK_CHECK(vkCreatePipelineCache(ctx->device, &cache_info, NULL, &ctx->pipe_cache));
    free(cache_data);

    return ctx;
}

void vk_destroy(VkCtx* ctx) {
    if (!ctx) return;

    vkDeviceWaitIdle(ctx->device);

    // Save pipeline cache
    size_t cache_size = 0;
    vkGetPipelineCacheData(ctx->device, ctx->pipe_cache, &cache_size, NULL);
    if (cache_size > 0) {
        void* data = malloc(cache_size);
        if (vkGetPipelineCacheData(ctx->device, ctx->pipe_cache, &cache_size, data) == VK_SUCCESS) {
            FILE* f = fopen(PIPE_CACHE_PATH, "wb");
            if (f) {
                fwrite(data, 1, cache_size, f);
                fclose(f);
            }
        }
        free(data);
    }

    vkDestroyPipelineCache(ctx->device, ctx->pipe_cache, NULL);
    vkDestroyDescriptorPool(ctx->device, ctx->desc_pool, NULL);
    vkDestroyCommandPool(ctx->device, ctx->cmd_pool, NULL);
    vkDestroyDevice(ctx->device, NULL);
    vkDestroyInstance(ctx->instance, NULL);
    free(ctx);
}

void vk_descriptor_pool_reset(VkCtx* ctx) {
    VK_CHECK(vkResetDescriptorPool(ctx->device, ctx->desc_pool, 0));
}

// ---- Buffers ----

VkBuf* vk_buf_create(VkCtx* ctx, size_t bytes) {
    VkBuf* buf = calloc(1, sizeof(VkBuf));
    buf->size = bytes;

    VkBufferCreateInfo buf_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = bytes,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VK_CHECK(vkCreateBuffer(ctx->device, &buf_info, NULL, &buf->buffer));

    VkMemoryRequirements mem_req;
    vkGetBufferMemoryRequirements(ctx->device, buf->buffer, &mem_req);

    VkMemoryAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mem_req.size,
        .memoryTypeIndex = ctx->mem_type_index,
    };
    VK_CHECK(vkAllocateMemory(ctx->device, &alloc_info, NULL, &buf->memory));
    VK_CHECK(vkBindBufferMemory(ctx->device, buf->buffer, buf->memory, 0));
    VK_CHECK(vkMapMemory(ctx->device, buf->memory, 0, bytes, 0, &buf->mapped));

    return buf;
}

VkBuf* vk_buf_create_from_data(VkCtx* ctx, const void* src, size_t bytes) {
    VkBuf* buf = vk_buf_create(ctx, bytes);
    memcpy(buf->mapped, src, bytes);
    return buf;
}

void* vk_buf_map(VkBuf* buf) {
    return buf->mapped;
}

size_t vk_buf_size(VkBuf* buf) {
    return buf->size;
}

void vk_buf_destroy(VkCtx* ctx, VkBuf* buf) {
    if (!buf) return;
    vkUnmapMemory(ctx->device, buf->memory);
    vkDestroyBuffer(ctx->device, buf->buffer, NULL);
    vkFreeMemory(ctx->device, buf->memory, NULL);
    free(buf);
}

// ---- Pipelines ----

VkPipe* vk_pipe_create(VkCtx* ctx, const char* spv_path,
                        uint32_t push_const_size, uint32_t num_bindings) {
    // Read SPIR-V
    size_t code_size = 0;
    void* code = read_file(spv_path, &code_size);
    if (!code) {
        fprintf(stderr, "Failed to read shader: %s\n", spv_path);
        return NULL;
    }

    VkPipe* pipe = calloc(1, sizeof(VkPipe));
    pipe->push_const_size = push_const_size;
    pipe->num_bindings = num_bindings;

    // Shader module
    VkShaderModuleCreateInfo mod_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code_size,
        .pCode = (const uint32_t*)code,
    };
    VkShaderModule module;
    VK_CHECK(vkCreateShaderModule(ctx->device, &mod_info, NULL, &module));
    free(code);

    // Descriptor set layout
    VkDescriptorSetLayoutBinding* bindings = calloc(num_bindings, sizeof(*bindings));
    for (uint32_t i = 0; i < num_bindings; i++) {
        bindings[i] = (VkDescriptorSetLayoutBinding){
            .binding = i,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        };
    }
    VkDescriptorSetLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = num_bindings,
        .pBindings = bindings,
    };
    VK_CHECK(vkCreateDescriptorSetLayout(ctx->device, &layout_info, NULL, &pipe->desc_layout));
    free(bindings);

    // Pipeline layout
    VkPushConstantRange push_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = push_const_size,
    };
    VkPipelineLayoutCreateInfo pipe_layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &pipe->desc_layout,
        .pushConstantRangeCount = push_const_size > 0 ? 1u : 0u,
        .pPushConstantRanges = push_const_size > 0 ? &push_range : NULL,
    };
    VK_CHECK(vkCreatePipelineLayout(ctx->device, &pipe_layout_info, NULL, &pipe->layout));

    // Compute pipeline
    VkComputePipelineCreateInfo pipe_info = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = module,
            .pName = "main",
        },
        .layout = pipe->layout,
    };
    VK_CHECK(vkCreateComputePipelines(ctx->device, ctx->pipe_cache, 1, &pipe_info, NULL,
                                       &pipe->pipeline));

    vkDestroyShaderModule(ctx->device, module, NULL);
    return pipe;
}

void vk_pipe_destroy(VkCtx* ctx, VkPipe* pipe) {
    if (!pipe) return;
    vkDestroyPipeline(ctx->device, pipe->pipeline, NULL);
    vkDestroyPipelineLayout(ctx->device, pipe->layout, NULL);
    vkDestroyDescriptorSetLayout(ctx->device, pipe->desc_layout, NULL);
    free(pipe);
}

// ---- Command buffers ----

VkCmd* vk_cmd_begin(VkCtx* ctx) {
    VkCmd* cmd = calloc(1, sizeof(VkCmd));
    cmd->ctx = ctx;
    cmd->submitted = 0;

    VkCommandBufferAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = ctx->cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VK_CHECK(vkAllocateCommandBuffers(ctx->device, &alloc_info, &cmd->cmd_buf));

    VkFenceCreateInfo fence_info = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
    };
    VK_CHECK(vkCreateFence(ctx->device, &fence_info, NULL, &cmd->fence));

    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    VK_CHECK(vkBeginCommandBuffer(cmd->cmd_buf, &begin_info));

    return cmd;
}

void vk_cmd_bind(VkCmd* cmd, VkPipe* pipe, VkBuf** bufs, const size_t* offsets,
                  const size_t* ranges, uint32_t num_bufs,
                  const void* push_consts, uint32_t push_size) {
    VkCtx* ctx = cmd->ctx;

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = ctx->desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &pipe->desc_layout,
    };
    VkDescriptorSet desc_set;
    VK_CHECK(vkAllocateDescriptorSets(ctx->device, &alloc_info, &desc_set));

    // Write descriptors
    VkDescriptorBufferInfo* buf_infos = calloc(num_bufs, sizeof(*buf_infos));
    VkWriteDescriptorSet* writes = calloc(num_bufs, sizeof(*writes));

    for (uint32_t i = 0; i < num_bufs; i++) {
        buf_infos[i] = (VkDescriptorBufferInfo){
            .buffer = bufs[i]->buffer,
            .offset = offsets ? (VkDeviceSize)offsets[i] : 0,
            .range = ranges ? (VkDeviceSize)ranges[i] : VK_WHOLE_SIZE,
        };
        writes[i] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = desc_set,
            .dstBinding = i,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &buf_infos[i],
        };
    }
    vkUpdateDescriptorSets(ctx->device, num_bufs, writes, 0, NULL);
    free(buf_infos);
    free(writes);

    // Bind pipeline and descriptor set
    vkCmdBindPipeline(cmd->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipe->pipeline);
    vkCmdBindDescriptorSets(cmd->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipe->layout, 0, 1, &desc_set, 0, NULL);

    // Push constants
    if (push_size > 0 && push_consts) {
        vkCmdPushConstants(cmd->cmd_buf, pipe->layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size, push_consts);
    }
}

void vk_cmd_dispatch(VkCmd* cmd, uint32_t gx, uint32_t gy, uint32_t gz) {
    vkCmdDispatch(cmd->cmd_buf, gx, gy, gz);
}

void vk_cmd_barrier(VkCmd* cmd) {
    VkMemoryBarrier2 barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
    };
    VkDependencyInfo dep = {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &barrier,
    };
    vkCmdPipelineBarrier2(cmd->cmd_buf, &dep);
}

static void cmd_end_and_submit(VkCmd* cmd) {
    VK_CHECK(vkEndCommandBuffer(cmd->cmd_buf));

    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd->cmd_buf,
    };
    VK_CHECK(vkQueueSubmit(cmd->ctx->queue, 1, &submit_info, cmd->fence));
    cmd->submitted = 1;
}

void vk_cmd_submit(VkCmd* cmd) {
    cmd_end_and_submit(cmd);
    VK_CHECK(vkWaitForFences(cmd->ctx->device, 1, &cmd->fence, VK_TRUE, UINT64_MAX));
}

void vk_cmd_submit_no_wait(VkCmd* cmd) {
    cmd_end_and_submit(cmd);
}

void vk_cmd_wait(VkCmd* cmd) {
    if (cmd->submitted) {
        VK_CHECK(vkWaitForFences(cmd->ctx->device, 1, &cmd->fence, VK_TRUE, UINT64_MAX));
    }
}

void vk_cmd_reset(VkCmd* cmd) {
    if (cmd->submitted) {
        VK_CHECK(vkWaitForFences(cmd->ctx->device, 1, &cmd->fence, VK_TRUE, UINT64_MAX));
    }
    VK_CHECK(vkResetFences(cmd->ctx->device, 1, &cmd->fence));
    VK_CHECK(vkResetCommandBuffer(cmd->cmd_buf, 0));
    cmd->submitted = 0;

    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    VK_CHECK(vkBeginCommandBuffer(cmd->cmd_buf, &begin_info));
}

void vk_cmd_destroy(VkCtx* ctx, VkCmd* cmd) {
    if (!cmd) return;
    if (cmd->submitted) {
        vkWaitForFences(ctx->device, 1, &cmd->fence, VK_TRUE, UINT64_MAX);
    }
    vkDestroyFence(ctx->device, cmd->fence, NULL);
    vkFreeCommandBuffers(ctx->device, ctx->cmd_pool, 1, &cmd->cmd_buf);
    free(cmd);
}
