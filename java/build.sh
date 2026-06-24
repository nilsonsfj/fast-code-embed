#!/usr/bin/env bash
# build.sh — Build and test the Java JNI binding for fast-code-embed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Detect JAVA_HOME
if [ -z "${JAVA_HOME:-}" ]; then
    JAVA_HOME="$(dirname "$(dirname "$(readlink -f "$(which java)")")")"
fi
JNI_INCLUDE="$JAVA_HOME/include"

# Compiler (honor $CC; default to cc).
CC="${CC:-cc}"

# Detect platform: macOS vs Linux (controls JNI headers + library suffix).
OS="$(uname -s)"
if [ "$OS" = "Darwin" ]; then
    JNI_PLATFORM_INCLUDE="$JNI_INCLUDE/darwin"
    LIB_SUFFIX=".dylib"
    # Pin the deployment target so the dylib's minimum-OS load requirement is
    # decoupled from whichever macOS the build host runs. 11.0 (Big Sur) is the
    # first Apple-Silicon release, so it is the lowest floor an arm64 dylib can
    # have. Without this, clang stamps minos to the build host's OS version,
    # which would stop the shipped dylib from loading on older macOS.
    PLATFORM_FLAGS="-mmacosx-version-min=11.0"
else
    JNI_PLATFORM_INCLUDE="$JNI_INCLUDE/linux"
    LIB_SUFFIX=".so"
    PLATFORM_FLAGS=""
fi

# The uninitialized-variable warning has a different name on GCC vs Clang;
# pick it by compiler family rather than by OS (Linux can use either).
if "$CC" --version 2>/dev/null | grep -qiE 'clang|llvm'; then
    UNINIT_FLAG="-Werror=sometimes-uninitialized"
else
    UNINIT_FLAG="-Werror=maybe-uninitialized"
fi

# Output dirs
CLASSES_DIR="$SCRIPT_DIR/lib/classes"
NATIVE_DIR="$SCRIPT_DIR/lib/native"
mkdir -p "$CLASSES_DIR" "$NATIVE_DIR"

echo "=== Compiling C library ==="
cd "$PROJECT_ROOT"
make -j4 lib CC="$CC"

echo ""
echo "=== Compiling JNI native code ==="
"$CC" -shared -fPIC -O2 -Wall -Wextra -Werror \
   -Werror=uninitialized "$UNINIT_FLAG" \
   $PLATFORM_FLAGS \
   -DNDEBUG \
   -I"$PROJECT_ROOT/src" \
   -I"$JNI_INCLUDE" -I"$JNI_PLATFORM_INCLUDE" \
   "$SCRIPT_DIR/src/main/native/fast_code_embed_jni.c" \
   -L"$PROJECT_ROOT/build" -lfast_code_embed \
   -lpthread -lm \
   -o "$NATIVE_DIR/libfast_code_embed_jni${LIB_SUFFIX}"

echo ""
echo "=== Compiling Java sources ==="
find "$SCRIPT_DIR/src/main/java" -name '*.java' > /tmp/java_sources.txt
find "$SCRIPT_DIR/src/test/java" -name '*.java' >> /tmp/java_sources.txt
javac -d "$CLASSES_DIR" @/tmp/java_sources.txt

echo ""
echo "=== Running tests ==="
java -cp "$CLASSES_DIR" -Djava.library.path="$NATIVE_DIR" \
    -Xcheck:jni \
    io.github.nilsonsfj.fastcodeembed.FastCodeEmbedTest

if [ "${1:-}" = "bench" ]; then
    echo ""
    echo "=== Running benchmark ==="
    java -cp "$CLASSES_DIR" -Djava.library.path="$NATIVE_DIR" \
        io.github.nilsonsfj.fastcodeembed.BenchJava
fi

if [ "${1:-}" = "memquery" ]; then
    echo ""
    echo "=== Running memory + query benchmark ==="
    shift
    java -cp "$CLASSES_DIR" -Djava.library.path="$NATIVE_DIR" \
        io.github.nilsonsfj.fastcodeembed.BenchMemQuery "$@"
fi

if [ "${1:-}" = "index" ]; then
    echo ""
    echo "=== Running indexer ==="
    java -cp "$CLASSES_DIR" -Djava.library.path="$NATIVE_DIR" \
        io.github.nilsonsfj.fastcodeembed.IndexDir "${2:-}" "${3:-}"
fi

echo ""
echo "=== Done ==="