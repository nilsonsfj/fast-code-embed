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
JNI_DARWIN="$JAVA_HOME/include/darwin"

# Output dirs
CLASSES_DIR="$SCRIPT_DIR/lib/classes"
NATIVE_DIR="$SCRIPT_DIR/lib/native"
mkdir -p "$CLASSES_DIR" "$NATIVE_DIR"

echo "=== Compiling C library ==="
cd "$PROJECT_ROOT"
make -j4 lib

echo ""
echo "=== Compiling JNI native code ==="
cc -shared -fPIC -O2 -Wall -Wextra -Werror \
   -Werror=uninitialized -Werror=sometimes-uninitialized \
   -DNDEBUG \
   -I"$PROJECT_ROOT/src" \
   -I"$JNI_INCLUDE" -I"$JNI_DARWIN" \
   "$SCRIPT_DIR/src/main/native/fast_code_embed_jni.c" \
   -L"$PROJECT_ROOT/build" -lfast_code_embed \
   -lpthread -lm \
   -o "$NATIVE_DIR/libfast_code_embed_jni.dylib"

echo ""
echo "=== Compiling Java sources ==="
find "$SCRIPT_DIR/src/main/java" -name '*.java' > /tmp/java_sources.txt
find "$SCRIPT_DIR/src/test/java" -name '*.java' >> /tmp/java_sources.txt
javac -d "$CLASSES_DIR" @/tmp/java_sources.txt

echo ""
echo "=== Running tests ==="
java -cp "$CLASSES_DIR" -Djava.library.path="$NATIVE_DIR" \
    com.github.nilsonsfj.fastcodeembed.FastCodeEmbedTest

if [ "${1:-}" = "bench" ]; then
    echo ""
    echo "=== Running benchmark ==="
    java -cp "$CLASSES_DIR" -Djava.library.path="$NATIVE_DIR" \
        com.github.nilsonsfj.fastcodeembed.BenchJava
fi

if [ "${1:-}" = "memquery" ]; then
    echo ""
    echo "=== Running memory + query benchmark ==="
    shift
    java -cp "$CLASSES_DIR" -Djava.library.path="$NATIVE_DIR" \
        com.github.nilsonsfj.fastcodeembed.BenchMemQuery "$@"
fi

if [ "${1:-}" = "index" ]; then
    echo ""
    echo "=== Running indexer ==="
    java -cp "$CLASSES_DIR" -Djava.library.path="$NATIVE_DIR" \
        com.github.nilsonsfj.fastcodeembed.IndexDir "${2:-}" "${3:-}"
fi

echo ""
echo "=== Done ==="