/*
 * index_dir.c — Recursively tokenize and index a directory of source files.
 *
 * Chunks files at 2048 bytes, tokenizes each chunk, builds a corpus.
 * Processes files in batches to avoid holding all token strings in memory.
 *
 * Build:  cc -O2 -std=c11 -Isrc index_dir.c -Lbuild -lstatic_nomic -lpthread -lm -o index_dir
 * Run:    ./index_dir /path/to/source [chunk_size] [batch_size]
 */
#include "semantic/semantic.h"
#include "foundation/platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <time.h>
#include <fnmatch.h>

#define DEFAULT_CHUNK_SIZE 2048
#define DEFAULT_BATCH_SIZE 5000
#define MAX_PATH_LEN 1024
#define MAX_TOKENS_PER_CHUNK 256

/* File extensions to index */
static const char *INCLUDE_EXTS[] = { ".c", ".h", ".cpp", ".hpp", ".java", ".py", ".rs", ".go", ".js", ".ts", NULL };

static int should_include(const char *path) {
    size_t len = strlen(path);
    for (const char **ext = INCLUDE_EXTS; *ext; ext++) {
        size_t extlen = strlen(*ext);
        if (len >= extlen && strcmp(path + len - extlen, *ext) == 0) return 1;
    }
    return 0;
}

/* ── File reading ──────────────────────────────────────────────── */

#define MAX_FILE_SIZE (64 * 1024 * 1024)  /* 64 MB sanity limit */

static char *read_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }

    /* Use fstat for portable file size — avoids 32-bit ftell overflow. */
    struct stat st;
    if (fstat(fileno(f), &st) != 0 || st.st_size <= 0) {
        fprintf(stderr, "%s: cannot determine file size\n", path);
        fclose(f);
        return NULL;
    }
    if (st.st_size > MAX_FILE_SIZE) {
        fprintf(stderr, "%s: file too large (%ld bytes, max %d)\n",
                path, (long)st.st_size, MAX_FILE_SIZE);
        fclose(f);
        return NULL;
    }

    size_t len = (size_t)st.st_size;
    char *buf = (char *)malloc(len);
    if (!buf) { perror("malloc"); fclose(f); return NULL; }

    size_t nread = fread(buf, 1, len, f);
    fclose(f);
    *out_len = nread;
    return buf;
}

/* ── Recursive directory walk ──────────────────────────────────── */

typedef struct {
    char **paths;
    int count;
    int capacity;
} file_list_t;

static void file_list_add(file_list_t *list, const char *path) {
    if (list->count >= list->capacity) {
        int new_cap = list->capacity ? list->capacity * 2 : 10000;
        char **grown = (char **)realloc(list->paths, (size_t)new_cap * sizeof(char *));
        if (!grown) return; /* OOM: skip this file */
        list->paths = grown;
        list->capacity = new_cap;
    }
    list->paths[list->count] = strdup(path);
    if (!list->paths[list->count]) return; /* OOM: skip this file */
    list->count++;
}

/* Iterative directory walk using an explicit stack — prevents stack overflow
 * on deep directory trees. */
static void walk_dir(const char *root, file_list_t *list) {
    /* dir_stack stores directory paths to process. */
    int stack_cap = 256;
    int stack_len = 0;
    char **dir_stack = (char **)malloc((size_t)stack_cap * sizeof(char *));
    if (!dir_stack) return;
    dir_stack[stack_len++] = strdup(root);

    while (stack_len > 0) {
        char *dirpath = dir_stack[--stack_len];
        DIR *dir = opendir(dirpath);
        if (!dir) { free(dirpath); continue; }
        struct dirent *entry;
        while ((entry = readdir(dir)) != NULL) {
            if (entry->d_name[0] == '.') continue;
            char fullpath[MAX_PATH_LEN];
            int written = snprintf(fullpath, sizeof(fullpath), "%s/%s", dirpath, entry->d_name);
            if (written < 0 || (size_t)written >= sizeof(fullpath)) continue;

            struct stat st;
            if (lstat(fullpath, &st) != 0) continue;
            if (S_ISLNK(st.st_mode)) continue;
            if (S_ISDIR(st.st_mode)) {
                if (stack_len == stack_cap) {
                    stack_cap *= 2;
                    char **grown = (char **)realloc(dir_stack, (size_t)stack_cap * sizeof(char *));
                    if (!grown) continue;
                    dir_stack = grown;
                }
                dir_stack[stack_len++] = strdup(fullpath);
            } else if (S_ISREG(st.st_mode) && should_include(fullpath)) {
                file_list_add(list, fullpath);
            }
        }
        closedir(dir);
        free(dirpath);
    }
    free(dir_stack);
}

/* ── Timing helper ─────────────────────────────────────────────── */

static double ms_since(struct timespec start) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (now.tv_sec - start.tv_sec) * 1000.0 +
           (now.tv_nsec - start.tv_nsec) / 1e6;
}

/* ── Main ──────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <directory> [chunk_size] [batch_size]\n", argv[0]);
        return 1;
    }
    const char *root_dir = argv[1];
    int chunk_size = argc > 2 ? atoi(argv[2]) : DEFAULT_CHUNK_SIZE;
    int batch_size = argc > 3 ? atoi(argv[3]) : DEFAULT_BATCH_SIZE;
    if (chunk_size <= 0) chunk_size = DEFAULT_CHUNK_SIZE;
    if (batch_size <= 0) batch_size = DEFAULT_BATCH_SIZE;

    struct timespec t_total, t0;
    clock_gettime(CLOCK_MONOTONIC, &t_total);

    printf("fast-code-embed directory indexer\n");
    printf("=====================================\n");
    printf("Directory: %s\n", root_dir);
    printf("Chunk size: %d bytes\n", chunk_size);
    printf("Batch size: %d chunks\n\n", batch_size);

    /* ── 1. Walk directory ────────────────────────────────────── */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    file_list_t files = {0};
    walk_dir(root_dir, &files);
    double walk_ms = ms_since(t0);
    printf("  Walk directory:           %8.1f ms  (%d files)\n", walk_ms, files.count);

    if (files.count == 0) {
        printf("No source files found.\n");
        free(files.paths);
        return 0;
    }

    /* ── 2. Stream files to corpus in batches ─────────────────── */
    clock_gettime(CLOCK_MONOTONIC, &t0);

    /* Allocate batch buffers */
    char **all_tokens = (char **)malloc((size_t)batch_size * MAX_TOKENS_PER_CHUNK * sizeof(char *));
    int *token_counts = (int *)malloc((size_t)batch_size * sizeof(int));
    if (!all_tokens || !token_counts) {
        fprintf(stderr, "ERROR: Failed to allocate batch buffers\n");
        free(all_tokens);
        free(token_counts);
        free(files.paths);
        return 1;
    }
    int batch_used = 0;
    int total_chunks = 0;
    int files_processed = 0;

    fce_sem_corpus_t *corp = fce_sem_corpus_new();

    for (int f = 0; f < files.count; f++) {
        size_t len;
        char *content = read_file(files.paths[f], &len);
        if (!content) continue;
        files_processed++;

        /* Chunk the file at } boundaries */
        for (size_t offset = 0; offset < len; ) {
            /* Find next } at or after offset+chunk_size for semantic split. */
            size_t end = offset + (size_t)chunk_size;
            if (end < len) {
                size_t found = 0;
                for (size_t i = end; i < len; i++) {
                    if (content[i] == '}') { found = i + 1; break; }
                }
                end = found ? found : len; /* no } left → take rest of file */
            } else {
                end = len; /* last chunk */
            }
            size_t chunk_len = end - offset;

            /* Null-terminate the chunk for tokenization */
            char *chunk = (char *)malloc(chunk_len + 1);
            if (!chunk) continue;  /* Skip chunk on malloc failure */
            memcpy(chunk, content + offset, chunk_len);
            chunk[chunk_len] = '\0';

            /* Tokenize the chunk as a single "name" */
            char *tok_buf[MAX_TOKENS_PER_CHUNK];
            int ntok = fce_sem_tokenize(chunk, tok_buf, MAX_TOKENS_PER_CHUNK);
            free(chunk);

            /* Store tokens in batch buffer */
            int base = batch_used * MAX_TOKENS_PER_CHUNK;
            for (int t = 0; t < ntok; t++) {
                all_tokens[base + t] = tok_buf[t];
            }
            token_counts[batch_used] = ntok;
            batch_used++;
            total_chunks++;
            offset = end;

            /* Flush batch when full */
            if (batch_used >= batch_size) {
                fce_sem_corpus_add_docs_batch(corp, all_tokens, token_counts, batch_used, MAX_TOKENS_PER_CHUNK);
                /* Free token strings (add_docs_batch doesn't take ownership) */
                for (int i = 0; i < batch_used; i++) {
                    int base2 = i * MAX_TOKENS_PER_CHUNK;
                    for (int t = 0; t < token_counts[i]; t++) {
                        free(all_tokens[base2 + t]);
                    }
                }
                batch_used = 0;
            }
        }
        free(content);
    }

    /* Flush remaining batch */
    if (batch_used > 0) {
        fce_sem_corpus_add_docs_batch(corp, all_tokens, token_counts, batch_used, MAX_TOKENS_PER_CHUNK);
        for (int i = 0; i < batch_used; i++) {
            int base = i * MAX_TOKENS_PER_CHUNK;
            for (int t = 0; t < token_counts[i]; t++) {
                free(all_tokens[base + t]);
            }
        }
        batch_used = 0;
    }

    free(all_tokens);
    free(token_counts);

    double chunk_ms = ms_since(t0);
    printf("  Read + chunk + tokenize:  %8.1f ms  (%d chunks from %d files)\n",
           chunk_ms, total_chunks, files_processed);

    /* ── 3. Finalize corpus ──────────────────────────────────── */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (fce_sem_corpus_finalize(corp) != 0) {
        fprintf(stderr, "Error: corpus finalization failed (out of memory)\n");
        fce_sem_corpus_free(corp);
        return 1;
    }
    double build_ms = ms_since(t0);
    printf("  Corpus finalize:          %8.1f ms\n", build_ms);

    /* ── 4. Stats ─────────────────────────────────────────────── */
    double total_ms = ms_since(t_total);
    printf("\n  ── Summary ──────────────────────────────────\n");
    printf("  Total chunks:    %d\n", total_chunks);
    printf("  Vocabulary:      %d tokens\n", fce_sem_corpus_token_count(corp));
    printf("  Documents:       %d\n", fce_sem_corpus_doc_count(corp));
    printf("  Total time:      %.1f ms\n", total_ms);
    printf("  Throughput:      %.0f chunks/sec\n", total_chunks / (total_ms / 1000.0));

    /* ── Cleanup ──────────────────────────────────────────────── */
    for (int i = 0; i < files.count; i++) free(files.paths[i]);
    free(files.paths);
    fce_sem_corpus_free(corp);

    return 0;
}
