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
    if (fd < 0) { perror("mkstemp"); return 1; }

    size_t expert_size = 7077888; // 4-bit expert size
    size_t total = 4 * expert_size;
    char* data = malloc(total);
    for (size_t i = 0; i < total; i++) data[i] = (char)(i & 0xFF);
    ssize_t written = write(fd, data, total);
    if ((size_t)written != total) { perror("write"); return 1; }

    IoRing* ring = io_ring_create(16);
    if (!ring) { fprintf(stderr, "failed to create io_ring\n"); return 1; }

    // Read 4 "experts" in parallel
    char* bufs[4];
    uint64_t offsets[4];
    size_t sizes[4];
    for (int i = 0; i < 4; i++) {
        bufs[i] = malloc(expert_size);
        offsets[i] = (uint64_t)i * expert_size;
        sizes[i] = expert_size;
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
