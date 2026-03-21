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

    for (int i = 0; i < count; i++) {
        struct io_uring_cqe* cqe;
        ret = io_uring_wait_cqe(&ring->ring, &cqe);
        if (ret < 0) {
            fprintf(stderr, "io_uring_wait_cqe failed: %s\n", strerror(-ret));
            return -1;
        }
        if (cqe->res < 0) {
            fprintf(stderr, "io_uring read failed: %s\n", strerror(-cqe->res));
            io_uring_cqe_seen(&ring->ring, cqe);
            return -1;
        }
        int idx = (int)(intptr_t)io_uring_cqe_get_data(cqe);
        if ((size_t)cqe->res < sizes[idx]) {
            fprintf(stderr, "io_uring short read %d: got %d of %zu bytes\n",
                    idx, cqe->res, sizes[idx]);
            io_uring_cqe_seen(&ring->ring, cqe);
            return -1;
        }
        io_uring_cqe_seen(&ring->ring, cqe);
    }
    return 0;
}
