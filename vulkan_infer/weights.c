#include "weights.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

// ---------------------------------------------------------------------------
// Minimal hand-written JSON parser
// ---------------------------------------------------------------------------
// Expected format (produced by extract_weights.py):
//
//   {
//     "tensors": {
//       "tensor.name": { "offset": N, "size": N, "shape": [d0, d1, ...], "dtype": "..." },
//       ...
//     },
//     ...
//   }
//
// We only care about the "tensors" object and ignore everything else.
// ---------------------------------------------------------------------------

// Advance p past whitespace.
static const char* skip_ws(const char* p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')
        p++;
    return p;
}

// Copy the next JSON string (after the opening '"') into buf (max len-1 chars).
// Returns pointer just past the closing '"', or NULL on error.
static const char* parse_string(const char* p, char* buf, int len) {
    // p should point at the opening '"'
    p = skip_ws(p);
    if (*p != '"') return NULL;
    p++;
    int i = 0;
    while (*p && *p != '"') {
        if (*p == '\\') {
            p++; // skip escape prefix
            if (!*p) return NULL;
            if (*p == '"' || *p == '\\' || *p == '/') {
                if (i < len - 1) buf[i++] = *p;
            } else if (*p == 'n') {
                if (i < len - 1) buf[i++] = '\n';
            } else if (*p == 't') {
                if (i < len - 1) buf[i++] = '\t';
            } else {
                // other escape sequences: just copy the char
                if (i < len - 1) buf[i++] = *p;
            }
        } else {
            if (i < len - 1) buf[i++] = *p;
        }
        p++;
    }
    if (*p != '"') return NULL;
    buf[i] = '\0';
    return p + 1; // past closing '"'
}

// Parse a JSON integer (no leading whitespace assumed consumed by caller).
// Returns pointer just past the last digit, or NULL on error.
static const char* parse_int(const char* p, long long* out) {
    p = skip_ws(p);
    char* end;
    errno = 0;
    *out = strtoll(p, &end, 10);
    if (end == p || errno != 0) return NULL;
    return end;
}

// Scan forward in p for the next occurrence of needle, return pointer to it
// or NULL if not found before end of string.
static const char* find_str(const char* p, const char* needle) {
    p = strstr(p, needle);
    return p;
}

// Parse the contents of the "tensors" JSON object into wf->tensors[].
// p points just after the opening '{' of the tensors object.
static int parse_tensors_object(const char* p, WeightFile* wf) {
    int count = 0;
    while (count < MAX_TENSORS) {
        p = skip_ws(p);
        // Each entry starts with a quoted tensor name or we hit '}'
        if (*p == '}') break;
        if (*p != '"') {
            // Skip comma between entries
            if (*p == ',') { p++; continue; }
            // Unexpected char — try to recover by skipping
            p++;
            continue;
        }

        TensorInfo* ti = &wf->tensors[count];
        memset(ti, 0, sizeof(*ti));

        // Parse tensor name
        const char* after = parse_string(p, ti->name, MAX_TENSOR_NAME);
        if (!after) break;
        p = after;

        // Expect ':'
        p = skip_ws(p);
        if (*p != ':') break;
        p++;

        // Expect '{'
        p = skip_ws(p);
        if (*p != '{') break;
        p++;

        // Parse fields: "offset", "size", "shape" (in any order)
        // We look for each field key inside the object.
        // Find the closing '}' of this tensor object.
        // Strategy: scan for each key within the next closing '}'.
        // Since the objects are simple and non-nested (shape is an array of ints),
        // we scan for each known key.

        // Find end of this tensor object to bound our search.
        // We need to find the matching '}' — shape array contains only ints so
        // nesting depth is at most 1 extra '['.
        const char* obj_start = p;
        int depth = 1;
        const char* obj_end = p;
        while (*obj_end && depth > 0) {
            if (*obj_end == '{') depth++;
            else if (*obj_end == '}') depth--;
            else if (*obj_end == '[') depth++; // treat array as nested for safety
            else if (*obj_end == ']') depth--;
            if (depth > 0) obj_end++;
        }
        // obj_end now points at the closing '}' of the tensor object.
        // Make a temporary copy so we can null-terminate.
        int obj_len = (int)(obj_end - obj_start);
        char* obj = (char*)malloc(obj_len + 1);
        if (!obj) return -1;
        memcpy(obj, obj_start, obj_len);
        obj[obj_len] = '\0';

        // Extract "offset"
        const char* kp = find_str(obj, "\"offset\"");
        if (kp) {
            kp += strlen("\"offset\"");
            kp = skip_ws(kp);
            if (*kp == ':') kp++;
            long long v;
            if (parse_int(kp, &v)) ti->offset = (size_t)v;
        }

        // Extract "size"
        kp = find_str(obj, "\"size\"");
        if (kp) {
            kp += strlen("\"size\"");
            kp = skip_ws(kp);
            if (*kp == ':') kp++;
            long long v;
            if (parse_int(kp, &v)) ti->size = (size_t)v;
        }

        // Extract "shape": [d0, d1, ...]
        kp = find_str(obj, "\"shape\"");
        if (kp) {
            kp += strlen("\"shape\"");
            kp = skip_ws(kp);
            if (*kp == ':') kp++;
            kp = skip_ws(kp);
            if (*kp == '[') {
                kp++;
                ti->ndim = 0;
                while (ti->ndim < MAX_DIMS) {
                    kp = skip_ws(kp);
                    if (*kp == ']') break;
                    if (*kp == ',') { kp++; continue; }
                    long long d;
                    const char* after_d = parse_int(kp, &d);
                    if (!after_d) break;
                    ti->shape[ti->ndim++] = (int)d;
                    kp = after_d;
                }
            }
        }

        free(obj);

        // Advance p past the closing '}' of this tensor
        p = obj_end;
        if (*p == '}') p++;

        count++;
    }
    return count;
}

// Parse the JSON manifest and populate wf->tensors / wf->num_tensors.
// Returns 0 on success, -1 on error.
static int parse_manifest(const char* json, size_t json_len, WeightFile* wf) {
    (void)json_len;

    // Find the "tensors" key at the top level
    const char* p = find_str(json, "\"tensors\"");
    if (!p) {
        fprintf(stderr, "weights: 'tensors' key not found in manifest\n");
        return -1;
    }
    p += strlen("\"tensors\"");
    p = skip_ws(p);
    if (*p != ':') {
        fprintf(stderr, "weights: expected ':' after 'tensors'\n");
        return -1;
    }
    p++;
    p = skip_ws(p);
    if (*p != '{') {
        fprintf(stderr, "weights: expected '{' for tensors object\n");
        return -1;
    }
    p++; // past opening '{'

    int n = parse_tensors_object(p, wf);
    if (n < 0) {
        fprintf(stderr, "weights: error parsing tensors object\n");
        return -1;
    }
    wf->num_tensors = n;
    return 0;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

WeightFile* weights_load(VkCtx* ctx, const char* bin_path, const char* json_path) {
    WeightFile* wf = (WeightFile*)calloc(1, sizeof(WeightFile));
    if (!wf) { perror("calloc WeightFile"); return NULL; }

    // --- Load JSON manifest ---
    FILE* jf = fopen(json_path, "rb");
    if (!jf) {
        fprintf(stderr, "weights: cannot open manifest '%s': %s\n", json_path, strerror(errno));
        free(wf);
        return NULL;
    }
    fseek(jf, 0, SEEK_END);
    long json_len = ftell(jf);
    fseek(jf, 0, SEEK_SET);
    char* json_buf = (char*)malloc((size_t)json_len + 1);
    if (!json_buf) {
        perror("malloc json_buf");
        fclose(jf);
        free(wf);
        return NULL;
    }
    if (fread(json_buf, 1, (size_t)json_len, jf) != (size_t)json_len) {
        fprintf(stderr, "weights: short read on manifest\n");
        fclose(jf);
        free(json_buf);
        free(wf);
        return NULL;
    }
    fclose(jf);
    json_buf[json_len] = '\0';

    if (parse_manifest(json_buf, (size_t)json_len, wf) != 0) {
        free(json_buf);
        free(wf);
        return NULL;
    }
    free(json_buf);

    fprintf(stderr, "weights: parsed %d tensors from manifest\n", wf->num_tensors);

    // --- Mmap the .bin file ---
    int fd = open(bin_path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "weights: cannot open bin '%s': %s\n", bin_path, strerror(errno));
        free(wf);
        return NULL;
    }
    struct stat st;
    if (fstat(fd, &st) != 0) {
        perror("fstat");
        close(fd);
        free(wf);
        return NULL;
    }
    size_t file_size = (size_t)st.st_size;
    wf->total_size = file_size;

    void* mapped_file = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapped_file == MAP_FAILED) {
        fprintf(stderr, "weights: mmap failed: %s\n", strerror(errno));
        free(wf);
        return NULL;
    }

    fprintf(stderr, "weights: loading %.2f GB from '%s'\n",
            (double)file_size / (1024.0 * 1024.0 * 1024.0), bin_path);

    // --- Copy into VkBuf or malloc ---
    if (ctx != NULL) {
        wf->buf = vk_buf_create(ctx, file_size);
        if (!wf->buf) {
            fprintf(stderr, "weights: vk_buf_create failed\n");
            munmap(mapped_file, file_size);
            free(wf);
            return NULL;
        }
        wf->mapped = vk_buf_map(wf->buf);
        if (!wf->mapped) {
            fprintf(stderr, "weights: vk_buf_map returned NULL\n");
            vk_buf_destroy(ctx, wf->buf);
            munmap(mapped_file, file_size);
            free(wf);
            return NULL;
        }
        memcpy(wf->mapped, mapped_file, file_size);
    } else {
        // CPU-only path
        wf->buf = NULL;
        wf->mapped = malloc(file_size);
        if (!wf->mapped) {
            fprintf(stderr, "weights: malloc(%zu) failed\n", file_size);
            munmap(mapped_file, file_size);
            free(wf);
            return NULL;
        }
        memcpy(wf->mapped, mapped_file, file_size);
    }

    munmap(mapped_file, file_size);

    fprintf(stderr, "weights: loaded %.2f GB OK, %d tensors\n",
            (double)file_size / (1024.0 * 1024.0 * 1024.0), wf->num_tensors);
    return wf;
}

void weights_destroy(VkCtx* ctx, WeightFile* wf) {
    if (!wf) return;
    if (wf->buf) {
        // VkBuf owns the mapped memory — no separate free needed
        if (ctx) vk_buf_destroy(ctx, wf->buf);
    } else if (wf->mapped) {
        free(wf->mapped);
    }
    free(wf);
}

TensorInfo* weights_get_tensor(WeightFile* wf, const char* name) {
    for (int i = 0; i < wf->num_tensors; i++) {
        if (strcmp(wf->tensors[i].name, name) == 0)
            return &wf->tensors[i];
    }
    return NULL;
}
