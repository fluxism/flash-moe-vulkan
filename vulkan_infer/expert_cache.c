#include "expert_cache.h"
#include "io_ring.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <float.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

// ============================================================================
// Background backfill thread — uses posix_fadvise WILLNEED to pre-populate
// the OS page cache with experts that CAR substituted or that were loaded
// from SSD (so future accesses hit page cache).
// ============================================================================

static void* backfill_thread_func(void* arg) {
    ExpertCache* ec = (ExpertCache*)arg;

    while (1) {
        BackfillRequest req;

        pthread_mutex_lock(&ec->backfill_mutex);
        while (ec->backfill_count == 0 && !ec->backfill_shutdown) {
            pthread_cond_wait(&ec->backfill_cond, &ec->backfill_mutex);
        }
        if (ec->backfill_shutdown && ec->backfill_count == 0) {
            pthread_mutex_unlock(&ec->backfill_mutex);
            break;
        }
        req = ec->backfill_queue[ec->backfill_head];
        ec->backfill_head = (ec->backfill_head + 1) % BACKFILL_QUEUE_SIZE;
        ec->backfill_count--;
        pthread_mutex_unlock(&ec->backfill_mutex);

        // Trigger readahead via madvise WILLNEED on the mmap'd region
        if (req.layer >= 0 && req.layer < ec->num_layers && ec->layer_maps[req.layer]) {
            void* expert_addr = (char*)ec->layer_maps[req.layer] +
                                (size_t)req.expert_id * ec->expert_size;
            madvise(expert_addr, ec->expert_size, MADV_WILLNEED);
        }
    }

    return NULL;
}

// ============================================================================
// Create / Destroy
// ============================================================================

ExpertCache* expert_cache_create(int num_layers, size_t expert_size,
                                  int* layer_fds) {
    ExpertCache* ec = calloc(1, sizeof(ExpertCache));
    if (!ec) return NULL;

    ec->num_layers = num_layers;
    ec->expert_size = expert_size;
    ec->car_threshold = 1.0f; // disabled by default
    ec->car_dampen = 0;
    ec->warmup_tokens = 0;
    ec->current_token = 0;
    ec->page_size = sysconf(_SC_PAGESIZE);

    // mmap each layer's expert file
    ec->layer_maps = calloc(num_layers, sizeof(void*));
    ec->layer_sizes = calloc(num_layers, sizeof(size_t));
    if (!ec->layer_maps || !ec->layer_sizes) goto fail;

    int mapped_count = 0;
    for (int l = 0; l < num_layers; l++) {
        if (layer_fds[l] < 0) continue;

        struct stat st;
        if (fstat(layer_fds[l], &st) != 0) continue;

        void* m = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, layer_fds[l], 0);
        if (m == MAP_FAILED) {
            fprintf(stderr, "[cache] mmap failed for layer %d: %s\n", l, strerror(errno));
            continue;
        }
        // Advise sequential access pattern (helps kernel readahead)
        madvise(m, st.st_size, MADV_RANDOM);
        ec->layer_maps[l] = m;
        ec->layer_sizes[l] = st.st_size;
        mapped_count++;
    }

    // mincore scratch buffer — enough for one expert's worth of pages
    size_t pages_per_expert = (expert_size + ec->page_size - 1) / ec->page_size;
    ec->mincore_buf_len = pages_per_expert;
    ec->mincore_buf = malloc(pages_per_expert);
    if (!ec->mincore_buf) goto fail;

    // Frequency profiling counters
    ec->freq_counts = calloc((size_t)num_layers * EC_MAX_CANDIDATES, sizeof(uint32_t));

    // Start backfill thread
    pthread_mutex_init(&ec->backfill_mutex, NULL);
    pthread_cond_init(&ec->backfill_cond, NULL);
    ec->backfill_head = 0;
    ec->backfill_tail = 0;
    ec->backfill_count = 0;
    ec->backfill_shutdown = 0;
    pthread_create(&ec->backfill_thread, NULL, backfill_thread_func, ec);

    fprintf(stderr, "[cache] mmap'd %d/%d expert layer files for CAR routing\n",
            mapped_count, num_layers);

    return ec;

fail:
    expert_cache_destroy(ec);
    return NULL;
}

void expert_cache_destroy(ExpertCache* ec) {
    if (!ec) return;

    // Shutdown backfill thread
    pthread_mutex_lock(&ec->backfill_mutex);
    ec->backfill_shutdown = 1;
    pthread_cond_signal(&ec->backfill_cond);
    pthread_mutex_unlock(&ec->backfill_mutex);
    pthread_join(ec->backfill_thread, NULL);
    pthread_mutex_destroy(&ec->backfill_mutex);
    pthread_cond_destroy(&ec->backfill_cond);

    if (ec->layer_maps) {
        for (int l = 0; l < ec->num_layers; l++) {
            if (ec->layer_maps[l]) {
                munmap(ec->layer_maps[l], ec->layer_sizes[l]);
            }
        }
        free(ec->layer_maps);
    }
    free(ec->layer_sizes);
    free(ec->mincore_buf);
    free(ec->freq_counts);
    free(ec);
}

// ============================================================================
// Configuration
// ============================================================================

void expert_cache_set_car(ExpertCache* ec, float threshold, int dampen, int warmup_tokens) {
    ec->car_threshold = threshold;
    ec->car_dampen = dampen;
    ec->warmup_tokens = warmup_tokens;
}

void expert_cache_new_token(ExpertCache* ec) {
    ec->current_token++;
}

// ============================================================================
// mincore-based page cache residency check
// ============================================================================

// Check if ALL pages of an expert are resident in the OS page cache.
// Returns 1 if fully cached, 0 if any page is not resident.
int expert_cache_is_resident(ExpertCache* ec, int layer_idx, int expert_id) {
    if (!ec->layer_maps[layer_idx]) return 0;

    void* expert_addr = (char*)ec->layer_maps[layer_idx] +
                        (size_t)expert_id * ec->expert_size;

    // Align to page boundary
    uintptr_t addr = (uintptr_t)expert_addr;
    uintptr_t aligned = addr & ~((uintptr_t)ec->page_size - 1);
    size_t len = ec->expert_size + (addr - aligned);
    size_t num_pages = (len + ec->page_size - 1) / ec->page_size;

    if (num_pages > ec->mincore_buf_len) return 0; // shouldn't happen

    if (mincore((void*)aligned, len, ec->mincore_buf) != 0) return 0;

    // Check if all pages are resident
    for (size_t i = 0; i < num_pages; i++) {
        if (!(ec->mincore_buf[i] & 1)) return 0;
    }
    return 1;
}

// ============================================================================
// Core: Cache-Aware Routing + Load
// ============================================================================

int expert_cache_route_and_load(
    ExpertCache* ec,
    int layer_idx,
    int packed_fd,
    const float* all_scores,
    int* expert_indices,
    float* expert_weights,
    void** expert_dests,
    size_t expert_size,
    void* io_ring)
{
    int ssd_reads = 0;
    int in_warmup = (ec->current_token < ec->warmup_tokens);
    float threshold = in_warmup ? 1.0f : ec->car_threshold;
    int did_dampen = 0;

    // Track original indices for profiling
    int original_indices[EC_MAX_EXPERTS];
    memcpy(original_indices, expert_indices, EC_MAX_EXPERTS * sizeof(int));

    // Track which experts need SSD reads
    int need_ssd[EC_MAX_EXPERTS] = {0};

    for (int k = 0; k < EC_MAX_EXPERTS; k++) {
        int eid = expert_indices[k];
        int resident = expert_cache_is_resident(ec, layer_idx, eid);

        if (resident) {
            // Expert is in OS page cache — copy directly from mmap (fast)
            void* src = (char*)ec->layer_maps[layer_idx] +
                        (size_t)eid * expert_size;
            memcpy(expert_dests[k], src, expert_size);
            ec->cache_hits++;
        } else {
            // Expert NOT in page cache — try CAR substitution
            ec->cache_misses++;
            int substituted = 0;

            if (threshold < 1.0f && ec->layer_maps[layer_idx]) {
                // Find best resident expert not already selected
                // Strategy: check top scoring cached experts from all_scores
                float best_score = -FLT_MAX;
                int best_eid = -1;

                // Scan all experts for best cached alternative.
                // This is O(NUM_EXPERTS) but only ~2KB of scores to scan.
                for (int e = 0; e < EC_MAX_CANDIDATES; e++) {
                    if (all_scores[e] <= best_score) continue;

                    // Skip if already selected
                    int already_used = 0;
                    for (int j = 0; j < EC_MAX_EXPERTS; j++) {
                        if (expert_indices[j] == e) { already_used = 1; break; }
                    }
                    if (already_used) continue;

                    // Check if this expert is in page cache
                    if (expert_cache_is_resident(ec, layer_idx, e)) {
                        best_score = all_scores[e];
                        best_eid = e;
                    }
                }

                if (best_eid >= 0) {
                    float original_score = all_scores[eid];
                    float ratio = (original_score > 1e-10f) ?
                                  best_score / original_score : 0.0f;

                    if (ratio >= threshold) {
                        // Substitute with cached expert
                        void* src = (char*)ec->layer_maps[layer_idx] +
                                    (size_t)best_eid * expert_size;
                        memcpy(expert_dests[k], src, expert_size);
                        expert_indices[k] = best_eid;

                        // Dampening: scale weight by score ratio
                        if (ec->car_dampen) {
                            expert_weights[k] *= ratio;
                            did_dampen = 1;
                        }

                        ec->car_substitutions++;
                        substituted = 1;

                        // Queue readahead for the originally-requested expert
                        pthread_mutex_lock(&ec->backfill_mutex);
                        if (ec->backfill_count < BACKFILL_QUEUE_SIZE) {
                            BackfillRequest* req = &ec->backfill_queue[ec->backfill_tail];
                            req->layer = layer_idx;
                            req->expert_id = eid;
                            ec->backfill_tail = (ec->backfill_tail + 1) % BACKFILL_QUEUE_SIZE;
                            ec->backfill_count++;
                            pthread_cond_signal(&ec->backfill_cond);
                        }
                        pthread_mutex_unlock(&ec->backfill_mutex);
                    }
                }
            }

            if (!substituted) {
                need_ssd[k] = 1;
                ssd_reads++;
            }
        }
        ec->total_lookups++;
    }

    // Batch SSD reads for non-cached, non-substituted experts via io_uring
    if (ssd_reads > 0 && packed_fd >= 0) {
        uint64_t io_offsets[EC_MAX_EXPERTS];
        void*    io_dsts[EC_MAX_EXPERTS];
        size_t   io_sizes[EC_MAX_EXPERTS];
        int io_count = 0;

        for (int k = 0; k < EC_MAX_EXPERTS; k++) {
            if (need_ssd[k]) {
                io_offsets[io_count] = (uint64_t)expert_indices[k] * expert_size;
                io_dsts[io_count] = expert_dests[k];
                io_sizes[io_count] = expert_size;
                io_count++;
            }
        }

        int ret = io_ring_read_experts((IoRing*)io_ring, packed_fd,
                                        io_offsets, io_dsts, io_sizes, io_count);
        if (ret != 0) {
            fprintf(stderr, "WARNING: layer %d expert io_ring read failed\n", layer_idx);
        }
    }

    // Re-normalize weights only if dampening modified them
    if (did_dampen) {
        float sum = 0.0f;
        for (int k = 0; k < EC_MAX_EXPERTS; k++) sum += expert_weights[k];
        if (sum > 0.0f) {
            float inv = 1.0f / sum;
            for (int k = 0; k < EC_MAX_EXPERTS; k++) expert_weights[k] *= inv;
        }
    }

    // Record frequency for profiling (using original router selections)
    expert_cache_record_freq(ec, layer_idx, original_indices, EC_MAX_EXPERTS);

    return ssd_reads;
}

// ============================================================================
// Frequency Profiling
// ============================================================================

void expert_cache_record_freq(ExpertCache* ec, int layer, const int* experts, int k) {
    if (!ec->freq_counts) return;
    for (int i = 0; i < k; i++) {
        if (experts[i] >= 0 && experts[i] < EC_MAX_CANDIDATES) {
            ec->freq_counts[layer * EC_MAX_CANDIDATES + experts[i]]++;
        }
    }
}

int expert_cache_save_profile(ExpertCache* ec, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[profile] Failed to open %s for writing: %s\n", path, strerror(errno));
        return -1;
    }

    int32_t header[2] = { ec->num_layers, EC_MAX_CANDIDATES };
    fwrite(header, sizeof(int32_t), 2, f);
    fwrite(ec->freq_counts, sizeof(uint32_t),
           (size_t)ec->num_layers * EC_MAX_CANDIDATES, f);
    fclose(f);

    fprintf(stderr, "[profile] Saved frequency profile to %s\n", path);
    return 0;
}

int expert_cache_load_profile(ExpertCache* ec, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[profile] Failed to open %s: %s\n", path, strerror(errno));
        return -1;
    }

    int32_t header[2];
    if (fread(header, sizeof(int32_t), 2, f) != 2) {
        fclose(f);
        return -1;
    }

    if (header[0] != ec->num_layers || header[1] != EC_MAX_CANDIDATES) {
        fprintf(stderr, "[profile] Mismatch: file has %dx%d, expected %dx%d\n",
                header[0], header[1], ec->num_layers, EC_MAX_CANDIDATES);
        fclose(f);
        return -1;
    }

    size_t expected = (size_t)ec->num_layers * EC_MAX_CANDIDATES;
    size_t nread = fread(ec->freq_counts, sizeof(uint32_t), expected, f);
    fclose(f);

    if (nread != expected) {
        fprintf(stderr, "[profile] Truncated file: read %zu of %zu entries\n", nread, expected);
        return -1;
    }

    fprintf(stderr, "[profile] Loaded frequency profile from %s\n", path);
    return 0;
}

void expert_cache_warmup_from_profile(ExpertCache* ec) {
    if (!ec->freq_counts) return;

    int preloaded = 0;

    for (int l = 0; l < ec->num_layers; l++) {
        if (!ec->layer_maps[l]) continue;

        uint32_t* layer_freq = &ec->freq_counts[l * EC_MAX_CANDIDATES];

        // Find top experts by frequency and trigger readahead via madvise
        // Load up to 32 experts per layer (covers most common patterns)
        int max_warmup = 32;
        for (int s = 0; s < max_warmup; s++) {
            uint32_t best_freq = 0;
            int best_eid = -1;

            for (int e = 0; e < EC_MAX_CANDIDATES; e++) {
                if (layer_freq[e] > best_freq) {
                    best_freq = layer_freq[e];
                    best_eid = e;
                }
            }

            if (best_eid < 0 || best_freq == 0) break;

            // Trigger readahead for this expert
            void* addr = (char*)ec->layer_maps[l] +
                         (size_t)best_eid * ec->expert_size;
            madvise(addr, ec->expert_size, MADV_WILLNEED);
            preloaded++;

            // Zero out so we don't pick it again
            layer_freq[best_eid] = 0;
        }
    }

    fprintf(stderr, "[profile] Triggered readahead for %d experts from frequency profile\n",
            preloaded);
}

// ============================================================================
// Stats
// ============================================================================

void expert_cache_print_stats(ExpertCache* ec) {
    if (ec->total_lookups == 0) return;

    fprintf(stderr, "\n--- Expert Cache Stats ---\n");
    fprintf(stderr, "  Total lookups:      %lu\n", (unsigned long)ec->total_lookups);
    fprintf(stderr, "  Page cache hits:    %lu (%.1f%%)\n",
            (unsigned long)ec->cache_hits,
            100.0 * ec->cache_hits / ec->total_lookups);
    fprintf(stderr, "  Page cache misses:  %lu (%.1f%%)\n",
            (unsigned long)ec->cache_misses,
            100.0 * ec->cache_misses / ec->total_lookups);
    fprintf(stderr, "  CAR substitutions:  %lu (%.1f%% of misses)\n",
            (unsigned long)ec->car_substitutions,
            ec->cache_misses > 0 ? 100.0 * ec->car_substitutions / ec->cache_misses : 0.0);
    fprintf(stderr, "  SSD reads avoided:  %lu (%.1f%% of total)\n",
            (unsigned long)(ec->cache_hits + ec->car_substitutions),
            100.0 * (ec->cache_hits + ec->car_substitutions) / ec->total_lookups);
    fprintf(stderr, "  CAR threshold:      %.2f\n", ec->car_threshold);
    fprintf(stderr, "  Dampening:          %s\n", ec->car_dampen ? "on" : "off");
}
