# fast-code-embed — Standalone library for algorithmic code embeddings
#
# Targets:
#   make            — Build static library (libfast_code_embed.a)
#   make test       — Build and run tests
#   make test-asan  — Build and run tests with ASan/UBSan
#   make clean      — Remove build artifacts
#   make extract    — Regenerate nomic vectors (requires Python + torch)
#
CC      ?= cc
# -Wgnu-zero-variadic-macro-arguments is suppressed because fce_log_* macros
# rely on the GNU `, ##__VA_ARGS__` extension (C-2 review 0002 §5.8:
# __VA_OPT__ is C23 and rejected by -std=c11). All other pedantic warnings
# remain on.
CFLAGS  ?= -O2 -Wall -Wextra -Wpedantic -Wno-gnu-zero-variadic-macro-arguments -std=c11 -mtune=native
AR      ?= ar
ARFLAGS ?= rcs

# ── Paths ────────────────────────────────────────────────────────
SRCDIR   = src
BUILDDIR = build
INCDIR   = src

# ── Sources ──────────────────────────────────────────────────────
SRCS = \
	$(SRCDIR)/version.c \
	$(SRCDIR)/semantic/semantic.c \
	$(SRCDIR)/foundation/hash_table.c \
	$(SRCDIR)/foundation/platform.c \
	$(SRCDIR)/foundation/system_info.c \
	$(SRCDIR)/foundation/compat_thread.c \
	$(SRCDIR)/foundation/log.c \
	$(SRCDIR)/foundation/profile.c \
	$(SRCDIR)/pipeline/worker_pool.c \
	$(SRCDIR)/xxhash/xxhash.c \
	$(SRCDIR)/embed/code_vectors_blob.S

OBJS = $(SRCS:$(SRCDIR)/%.c=$(BUILDDIR)/%.o)
OBJS := $(OBJS:$(SRCDIR)/%.S=$(BUILDDIR)/%.o)

# ── Library ──────────────────────────────────────────────────────
LIB = $(BUILDDIR)/libfast_code_embed.a

# ── Test ─────────────────────────────────────────────────────────
TEST_SRC  = tests/test_semantic.c
TEST_BIN  = $(BUILDDIR)/test_semantic

.PHONY: all lib test test-asan clean extract

all: lib

lib: $(LIB)

$(LIB): $(OBJS)
	@mkdir -p $(dir $@)
	$(AR) $(ARFLAGS) $@ $^
	@echo "Built $@"

# ── Object compilation ──────────────────────────────────────────
$(BUILDDIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -MMD -MP -I$(INCDIR) -c $< -o $@

$(BUILDDIR)/%.o: $(SRCDIR)/%.S
	@mkdir -p $(dir $@)
	$(CC) -c $< -o $@

# ── Tests ────────────────────────────────────────────────────────
test: $(TEST_BIN)
	$(TEST_BIN)

$(TEST_BIN): $(TEST_SRC) $(LIB)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -I$(INCDIR) $< -L$(BUILDDIR) -lfast_code_embed -lpthread -lm -o $@

# ── Benchmark ────────────────────────────────────────────────────
BENCH_SRC  = bench_mem_query.c
BENCH_BIN  = $(BUILDDIR)/bench_mem_query

bench: $(BENCH_BIN)

$(BENCH_BIN): $(BENCH_SRC) $(LIB)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -I$(INCDIR) $< -L$(BUILDDIR) -lfast_code_embed -lpthread -lm -o $@

# ── Reduced-dimension build (256 dims, ~640 MB less memory) ─────
LIB256    = $(BUILDDIR)/libfast_code_embed_256.a
OBJS256   = $(SRCS:$(SRCDIR)/%.c=$(BUILDDIR)/dim256/%.o)
OBJS256   := $(OBJS256:$(SRCDIR)/%.S=$(BUILDDIR)/dim256/%.o)
BENCH256  = $(BUILDDIR)/bench_mem_query_256

$(BUILDDIR)/dim256/%.o: $(SRCDIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -DFCE_SEM_DIM_256 -I$(INCDIR) -c $< -o $@

$(BUILDDIR)/dim256/%.o: $(SRCDIR)/%.S
	@mkdir -p $(dir $@)
	$(CC) -c $< -o $@

$(LIB256): $(OBJS256)
	@mkdir -p $(dir $@)
	$(AR) $(ARFLAGS) $@ $^
	@echo "Built $@ (256-dim)"

$(BENCH256): $(BENCH_SRC) $(LIB256)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -DFCE_SEM_DIM_256 -I$(INCDIR) $< -L$(BUILDDIR) -lfast_code_embed_256 -lpthread -lm -o $@

bench-256: $(BENCH256)

# ── Auto-generated dependency files ─────────────────────────────
-include $(OBJS:.o=.d)

# ── Extract nomic vectors ───────────────────────────────────────
extract:
	python3 scripts/extract_nomic_vectors.py --output-dir $(SRCDIR)/embed

# ── Clean ────────────────────────────────────────────────────────
clean:
	rm -rf $(BUILDDIR)
