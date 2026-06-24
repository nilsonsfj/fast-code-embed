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

# On macOS, pin the deployment target so the shipped library's minimum-OS floor
# is decoupled from whichever macOS the build host runs. 11.0 (Big Sur) is the
# first Apple-Silicon release, so it is the lowest floor an arm64 build can have.
# Without this, the compiler stamps the floor to the build host's OS version,
# which would stop the redistributed .a (and any dylib linking it) from loading
# on older macOS and emit "built for newer macOS" link warnings.
ifeq ($(shell uname -s),Darwin)
MACOS_MIN := -mmacosx-version-min=11.0
CFLAGS += $(MACOS_MIN)
endif
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

.PHONY: all lib test test-asan bench bench-256 loadquery windows-cross fuzz install uninstall clean extract

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
	$(CC) $(MACOS_MIN) -fPIC -c $< -o $@

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
	$(CC) $(MACOS_MIN) -c $< -o $@

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
	$(CC) $(MACOS_MIN) -c $< -o $@

$(LIB256): $(OBJS256)
	@mkdir -p $(dir $@)
	$(AR) $(ARFLAGS) $@ $^
	@echo "Built $@ (256-dim)"

$(BENCH256): $(BENCH_SRC) $(LIB256)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -DFCE_SEM_DIM_256 -I$(INCDIR) $< -L$(BUILDDIR) -lfast_code_embed_256 -lpthread -lm -o $@

bench-256: $(BENCH256)

# ── Windows cross-compile validation (mingw-w64) ────────────────
# Compiles and links the library for a Windows target using a mingw-w64 cross
# toolchain on a Linux/macOS host, exercising the _WIN32 code paths (native
# Win32 threads, QueryPerformanceCounter timing, the COFF vector blob). Nothing
# is executed — the artifacts are PE binaries; the value is that compile+link
# succeed, so a broken Windows path is caught in CI without a Windows runner.
# This is a mingw (GCC) build, not MSVC; a clean build here does not by itself
# guarantee an MSVC build. Override CROSS to select the toolchain prefix.
# -fPIC is omitted: all code is position-independent on Windows by default.
CROSS        ?= x86_64-w64-mingw32-
WIN_CC       = $(CROSS)gcc
WIN_AR       = $(CROSS)ar
WIN_CFLAGS   = -O2 -Wall -Wextra -Wpedantic -std=c11 $(TUNE) -DNDEBUG
WIN_BUILDDIR = $(BUILDDIR)/windows
WIN_OBJS     = $(SRCS:$(SRCDIR)/%.c=$(WIN_BUILDDIR)/%.o)
WIN_OBJS     := $(WIN_OBJS:$(SRCDIR)/%.S=$(WIN_BUILDDIR)/%.o)
WIN_LIB      = $(WIN_BUILDDIR)/libfast_code_embed.a
WIN_SMOKE    = $(WIN_BUILDDIR)/smoke.exe

$(WIN_BUILDDIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p $(dir $@)
	$(WIN_CC) $(WIN_CFLAGS) -I$(INCDIR) -c $< -o $@

$(WIN_BUILDDIR)/%.o: $(SRCDIR)/%.S
	@mkdir -p $(dir $@)
	$(WIN_CC) -c $< -o $@

$(WIN_LIB): $(WIN_OBJS)
	@mkdir -p $(dir $@)
	$(WIN_AR) $(ARFLAGS) $@ $^
	@echo "Built $@ (Windows cross)"

windows-cross: $(WIN_LIB)
	$(WIN_CC) $(WIN_CFLAGS) -I$(INCDIR) tests/smoke_windows.c $(WIN_LIB) -lm -o $(WIN_SMOKE)
	@echo "Linked $(WIN_SMOKE) — Windows _WIN32 paths compile and link"

# ── Fuzzing: cache loader (libFuzzer + ASan/UBSan) ──────────────
# Continuously exercises fce_sem_corpus_load (the untrusted-cache parser) with
# mutated bytes. Requires clang (libFuzzer is bundled with clang/LLVM). The
# library sources are recompiled with coverage instrumentation into a separate
# build dir; the harness links in the libFuzzer runtime (which provides main).
# The seed generator links the normal (uninstrumented) static library, so it
# avoids the fuzzer runtime and provides its own main.
#   make fuzz
#   ./build/fuzz/fuzz_corpus_load build/fuzz/corpus      # run until Ctrl-C
#   ./build/fuzz/fuzz_corpus_load -max_total_time=60 build/fuzz/corpus
FUZZ_CC       ?= clang
FUZZ_SAN       = -fsanitize=address,undefined -fno-sanitize-recover=undefined
FUZZ_CFLAGS    = -O1 -g -std=c11 -fsanitize=fuzzer-no-link $(FUZZ_SAN) -I$(INCDIR)
FUZZ_BUILDDIR  = $(BUILDDIR)/fuzz
FUZZ_OBJS      = $(SRCS:$(SRCDIR)/%.c=$(FUZZ_BUILDDIR)/%.o)
FUZZ_OBJS      := $(FUZZ_OBJS:$(SRCDIR)/%.S=$(FUZZ_BUILDDIR)/%.o)
FUZZ_BIN       = $(FUZZ_BUILDDIR)/fuzz_corpus_load
FUZZ_SEEDGEN   = $(FUZZ_BUILDDIR)/gen_seed
FUZZ_CORPUSDIR = $(FUZZ_BUILDDIR)/corpus

$(FUZZ_BUILDDIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p $(dir $@)
	$(FUZZ_CC) $(FUZZ_CFLAGS) -c $< -o $@

$(FUZZ_BUILDDIR)/%.o: $(SRCDIR)/%.S
	@mkdir -p $(dir $@)
	$(FUZZ_CC) -c $< -o $@

$(FUZZ_BIN): fuzz/fuzz_corpus_load.c $(FUZZ_OBJS)
	@mkdir -p $(dir $@)
	$(FUZZ_CC) -O1 -g -std=c11 -fsanitize=fuzzer $(FUZZ_SAN) -I$(INCDIR) $^ -lpthread -lm -o $@

$(FUZZ_SEEDGEN): fuzz/gen_seed.c $(LIB)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -I$(INCDIR) $< -L$(BUILDDIR) -lfast_code_embed -lpthread -lm -o $@

fuzz: $(FUZZ_BIN) $(FUZZ_SEEDGEN)
	@mkdir -p $(FUZZ_CORPUSDIR)
	$(FUZZ_SEEDGEN) $(FUZZ_CORPUSDIR)/seed_valid.cache
	@echo "Fuzzer built. Run:"
	@echo "  $(FUZZ_BIN) $(FUZZ_CORPUSDIR)"

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
