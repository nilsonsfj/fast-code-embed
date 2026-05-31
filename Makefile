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
CFLAGS  ?= -O2 -Wall -Wextra -Wpedantic -std=c11 -mtune=native
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

# ASan/UBSan build — reproducible from Makefile, no committed binary needed
ASAN_CFLAGS = -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -Wall -Wextra -Wpedantic -std=c11
ASAN_BIN = $(BUILDDIR)/test_asan
LIB_ASAN = $(BUILDDIR)/libfast_code_embed_asan.a
OBJS_ASAN = $(SRCS:$(SRCDIR)/%.c=$(BUILDDIR)/asan/%.o)
OBJS_ASAN := $(OBJS_ASAN:$(SRCDIR)/%.S=$(BUILDDIR)/asan/%.o)

test-asan: $(ASAN_BIN)
	$(ASAN_BIN)

$(BUILDDIR)/asan/%.o: $(SRCDIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(ASAN_CFLAGS) -I$(INCDIR) -c $< -o $@

$(BUILDDIR)/asan/%.o: $(SRCDIR)/%.S
	@mkdir -p $(dir $@)
	$(CC) $(ASAN_CFLAGS) -c $< -o $@

$(LIB_ASAN): $(OBJS_ASAN)
	@mkdir -p $(dir $@)
	$(AR) $(ARFLAGS) $@ $^
	@echo "Built $@ (ASan)"

$(ASAN_BIN): $(TEST_SRC) $(LIB_ASAN)
	@mkdir -p $(dir $@)
	$(CC) $(ASAN_CFLAGS) -I$(INCDIR) $< -L$(BUILDDIR) -lfast_code_embed_asan -lpthread -lm -o $@

# ── Auto-generated dependency files ─────────────────────────────
-include $(OBJS:.o=.d)

# ── Extract nomic vectors ───────────────────────────────────────
extract:
	python3 scripts/extract_nomic_vectors.py --output-dir $(SRCDIR)/embed

# ── Clean ────────────────────────────────────────────────────────
clean:
	rm -rf $(BUILDDIR)
