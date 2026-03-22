#ifndef IO_RING_H
#define IO_RING_H

#include <stdint.h>
#include <stddef.h>

typedef struct IoRing IoRing;

IoRing* io_ring_create(uint32_t queue_depth);
void    io_ring_destroy(IoRing* ring);

// Batch-submit count reads and wait for all completions.
// Returns 0 on success, -1 on error (including short reads).
int io_ring_read_experts(IoRing* ring,
                         int fd,
                         const uint64_t* offsets,
                         void** dests,
                         const size_t* sizes,
                         int count);

#endif // IO_RING_H
