#ifndef EXPERT_CACHE_H
#define EXPERT_CACHE_H

#include <stdint.h>
#include <stddef.h>
#include <pthread.h>

// ============================================================================
// Expert Cache with Cache-Aware Routing (CAR) and Background Backfill
//
// Uses mmap + mincore() to leverage the OS page cache for CAR decisions.
// No separate DRAM cache — works WITH the OS page cache instead of fighting it.
//
// Inspired by fomoe (github.com/pmerolla/fomoe):
// - CAR: substitute cached experts when uncached ones are selected
// - Dampening: scale substituted expert weights by score ratio
// - Background backfill: readahead originally-requested experts
// - Frequency profiling: pre-seed OS page cache with hot experts at startup
// ============================================================================

#define EC_MAX_EXPERTS    4    // K_EXPERTS
#define EC_MAX_CANDIDATES 512  // NUM_EXPERTS

// Backfill request: readahead an expert that CAR substituted
typedef struct {
    int layer;
    int expert_id;
} BackfillRequest;

#define BACKFILL_QUEUE_SIZE 256

typedef struct {
    // Per-layer mmap'd expert files
    void** layer_maps;   // [num_layers] mmap'd regions (NULL if not available)
    size_t* layer_sizes; // [num_layers] mmap size per layer
    int num_layers;
    size_t expert_size;  // bytes per expert

    // CAR parameters
    float car_threshold; // τ — minimum score ratio for substitution (0=always, 1=disabled)
    int   car_dampen;    // scale substituted weights by score ratio
    int   warmup_tokens; // force CAR=1.0 for first N tokens
    int   current_token; // track token count for warmup

    // Stats
    uint64_t cache_hits;       // expert was in page cache
    uint64_t cache_misses;     // expert was NOT in page cache
    uint64_t car_substitutions;
    uint64_t total_lookups;

    // Background backfill (readahead)
    pthread_t backfill_thread;
    pthread_mutex_t backfill_mutex;
    pthread_cond_t  backfill_cond;
    BackfillRequest backfill_queue[BACKFILL_QUEUE_SIZE];
    int backfill_head;
    int backfill_tail;
    int backfill_count;
    int backfill_shutdown;

    // mincore scratch buffer (reused per call, page_size aligned)
    unsigned char* mincore_buf;
    size_t mincore_buf_len;
    long page_size;

    // Frequency profiling
    uint32_t* freq_counts; // [num_layers * EC_MAX_CANDIDATES] expert usage counts
} ExpertCache;

// Create/destroy
// layer_fds: file descriptors for each layer's packed_experts file
ExpertCache* expert_cache_create(int num_layers, size_t expert_size,
                                  int* layer_fds);
void expert_cache_destroy(ExpertCache* ec);

// Configure CAR
void expert_cache_set_car(ExpertCache* ec, float threshold, int dampen, int warmup_tokens);

// Check if an expert is resident in OS page cache via mincore()
int expert_cache_is_resident(ExpertCache* ec, int layer_idx, int expert_id);

// Core: given router scores and topk results, apply CAR.
// expert_indices/expert_weights are modified in-place (CAR substitution + dampening).
// For cached experts, copies directly from mmap. For uncached, uses io_uring.
// expert_dests[K] are filled with expert data.
// Returns number of SSD reads actually performed.
int expert_cache_route_and_load(
    ExpertCache* ec,
    int layer_idx,
    int packed_fd,
    const float* all_scores,     // [NUM_EXPERTS] softmax'd routing scores
    int* expert_indices,         // [K] in/out — may be modified by CAR
    float* expert_weights,       // [K] in/out — may be dampened
    void** expert_dests,         // [K] destination buffers to fill
    size_t expert_size,          // bytes per expert
    void* io_ring                // IoRing* for SSD reads
);

// Signal that a new token is starting (for warmup tracking)
void expert_cache_new_token(ExpertCache* ec);

// Frequency profiling
void expert_cache_record_freq(ExpertCache* ec, int layer, const int* experts, int k);
int  expert_cache_save_profile(ExpertCache* ec, const char* path);
int  expert_cache_load_profile(ExpertCache* ec, const char* path);
void expert_cache_warmup_from_profile(ExpertCache* ec);

// Print stats
void expert_cache_print_stats(ExpertCache* ec);

#endif // EXPERT_CACHE_H
