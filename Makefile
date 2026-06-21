# fast-code-embed — Standalone library for algorithmic code embeddings
#
# Targets:
#   make            — Build static library (libfast_code_embed.a)
#   make test       — Build and run tests
#   make test-asan  — Build and run tests with ASan/UBSan
#   make install    — Install lib + headers + pkg-config (PREFIX=/usr/local)
#   make uninstall  — Remove installed files
#   make clean      — Remove build artifacts
#   make extract    — Regenerate nomic vectors (requires Python + torch)
#
CC      ?= cc
# -mtune controls instruction scheduling for a target microarchitecture (it does
# NOT change the instruction set — the AVX2 path is selected at runtime via
# __builtin_cpu_supports, so correctness is independent of this flag). Default to
# -mtune=generic so redistributed artifacts (manylinux wheel / Maven JAR) are not
# tuned for whatever CI runner happened to build them, which can pessimize on
# unrelated deployment CPUs. Source builders targeting their own machine can pass
# NATIVE=1 to recover -mtune=native scheduling.
NATIVE ?= 0
ifeq ($(NATIVE),1)
TUNE ?= -mtune=native
else
TUNE ?= -mtune=generic
endif
# -fPIC ensures the static library objects can be linked into a shared library
# (e.g., the JNI dylib) on all platforms, including macOS ARM64 where the linker
# requires PIC for dylib content. The tree builds warning-clean under
# -Wall -Wextra -Wpedantic on both GCC and Clang (-std=c11, no C23 features).
CFLAGS       ?= -O2 -Wall -Wextra -Wpedantic -std=c11 $(TUNE) -DNDEBUG -fPIC
CFLAGS_DEBUG ?= -O0 -Wall -Wextra -Wpedantic -std=c11 -g -fsanitize=address,undefined -fno-sanitize-recover=undefined -fno-omit-frame-pointer
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

.PHONY: all lib test test-asan bench bench-256 loadquery install uninstall clean extract

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
	$(CC) -fPIC -c $< -o $@

# ── Tests ────────────────────────────────────────────────────────
test: $(TEST_BIN)
	$(TEST_BIN)

$(TEST_BIN): $(TEST_SRC) $(LIB)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -I$(INCDIR) $< -L$(BUILDDIR) -lfast_code_embed -lpthread -lm -o $@

# ── ASan/UBSan build (no NDEBUG — asserts remain live) ──────────
LIB_ASAN   = $(BUILDDIR)/libfast_code_embed_asan.a
OBJS_ASAN  = $(SRCS:$(SRCDIR)/%.c=$(BUILDDIR)/asan/%.o)
OBJS_ASAN  := $(OBJS_ASAN:$(SRCDIR)/%.S=$(BUILDDIR)/asan/%.o)
TEST_ASAN  = $(BUILDDIR)/test_semantic_asan

$(BUILDDIR)/asan/%.o: $(SRCDIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS_DEBUG) -MMD -MP -I$(INCDIR) -c $< -o $@

$(BUILDDIR)/asan/%.o: $(SRCDIR)/%.S
	@mkdir -p $(dir $@)
	$(CC) -c $< -o $@

$(LIB_ASAN): $(OBJS_ASAN)
	@mkdir -p $(dir $@)
	$(AR) $(ARFLAGS) $@ $^
	@echo "Built $@ (ASan)"

$(TEST_ASAN): $(TEST_SRC) $(LIB_ASAN)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS_DEBUG) -I$(INCDIR) $< -L$(BUILDDIR) -lfast_code_embed_asan -lpthread -lm -o $@

test-asan: $(TEST_ASAN)
	$(TEST_ASAN)

# ── Benchmark ────────────────────────────────────────────────────
BENCH_SRC  = bench_mem_query.c
BENCH_BIN  = $(BUILDDIR)/bench_mem_query

bench: $(BENCH_BIN)

$(BENCH_BIN): $(BENCH_SRC) $(LIB)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -I$(INCDIR) $< -L$(BUILDDIR) -lfast_code_embed -lpthread -lm -o $@

# ── Example: load a saved corpus cache and query it ──────────────
LOADQUERY_SRC = examples/loadquery.c
LOADQUERY_BIN = $(BUILDDIR)/loadquery

loadquery: $(LOADQUERY_BIN)

$(LOADQUERY_BIN): $(LOADQUERY_SRC) $(LIB)
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

# ── Install ─────────────────────────────────────────────────────
# Installs the static library, the public headers (semantic/semantic.h and
# version.h — all that's needed to compile against the documented C API), and a
# pkg-config file. Honors PREFIX and DESTDIR (staged installs). Uses cp/mkdir
# only, so it works the same on Linux and macOS.
PREFIX     ?= /usr/local
DESTDIR    ?=
LIBDIR     ?= $(PREFIX)/lib
INCLUDEDIR ?= $(PREFIX)/include/fast-code-embed
PCDIR      ?= $(LIBDIR)/pkgconfig
VERSION    := $(shell awk '/define FCE_VERSION_MAJOR/{a=$$3}/define FCE_VERSION_MINOR/{b=$$3}/define FCE_VERSION_PATCH/{c=$$3}END{print a"."b"."c}' $(SRCDIR)/version.h)

install: $(LIB)
	@mkdir -p "$(DESTDIR)$(LIBDIR)" "$(DESTDIR)$(INCLUDEDIR)/semantic" "$(DESTDIR)$(PCDIR)"
	cp -f $(LIB) "$(DESTDIR)$(LIBDIR)/"
	cp -f $(SRCDIR)/version.h "$(DESTDIR)$(INCLUDEDIR)/version.h"
	cp -f $(SRCDIR)/semantic/semantic.h "$(DESTDIR)$(INCLUDEDIR)/semantic/semantic.h"
	@printf 'prefix=%s\nlibdir=%s\nincludedir=%s\n\nName: fast-code-embed\nDescription: Algorithmic code embeddings (TF-IDF + Random Indexing, no GPU)\nURL: https://github.com/nilsonsfj/fast-code-embed\nVersion: %s\nLibs: -L$${libdir} -lfast_code_embed -lpthread -lm\nCflags: -I$${includedir}\n' \
		"$(PREFIX)" "$(LIBDIR)" "$(INCLUDEDIR)" "$(VERSION)" \
		> "$(DESTDIR)$(PCDIR)/fast-code-embed.pc"
	@echo "Installed fast-code-embed $(VERSION) to $(DESTDIR)$(PREFIX)"

uninstall:
	rm -f "$(DESTDIR)$(LIBDIR)/$(notdir $(LIB))"
	rm -f "$(DESTDIR)$(PCDIR)/fast-code-embed.pc"
	rm -rf "$(DESTDIR)$(INCLUDEDIR)"
	@echo "Uninstalled fast-code-embed from $(DESTDIR)$(PREFIX)"

# ── Clean ────────────────────────────────────────────────────────
clean:
	rm -rf $(BUILDDIR)
