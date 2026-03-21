#ifndef WEIGHTS_H
#define WEIGHTS_H

#include "vk_compute.h"
#include <stdint.h>
#include <stddef.h>

#define MAX_TENSOR_NAME 256
#define MAX_TENSORS 2048
#define MAX_DIMS 4

typedef struct {
    char name[MAX_TENSOR_NAME];
    size_t offset;
    size_t size;
    int shape[MAX_DIMS];
    int ndim;
} TensorInfo;

typedef struct {
    VkBuf* buf;           // Single large VkBuf holding all weights (NULL if CPU-only)
    void* mapped;         // Persistently mapped pointer (into VkBuf or malloc'd block)
    TensorInfo tensors[MAX_TENSORS];
    int num_tensors;
    size_t total_size;
} WeightFile;

// Load manifest + weights. Mmaps the .bin file, copies into VkBuf, munmaps.
// If ctx is NULL, weights are kept in malloc'd memory (for CPU-only testing).
WeightFile* weights_load(VkCtx* ctx, const char* bin_path, const char* json_path);
void weights_destroy(VkCtx* ctx, WeightFile* wf);

// Look up a tensor by name. Returns NULL if not found.
TensorInfo* weights_get_tensor(WeightFile* wf, const char* name);

// Get a pointer to tensor data within the mapped region
static inline void* weights_tensor_data(WeightFile* wf, TensorInfo* ti) {
    return (char*)wf->mapped + ti->offset;
}

#endif // WEIGHTS_H
