#!/usr/bin/env bash
# bump_version.sh — Bump the project version across all tracked files.
#
# Usage:
#   ./scripts/bump_version.sh --to 1.2.3              # bump + commit
#   ./scripts/bump_version.sh                          # auto-bump patch + commit
#   ./scripts/bump_version.sh --to 1.0.0 --create-tag  # bump + commit + tag
#   ./scripts/bump_version.sh --create-tag             # auto-bump patch + commit + tag
#   ./scripts/bump_version.sh --no-commit              # bump files only, no commit
#   ./scripts/bump_version.sh --create-tag --push      # bump + commit + tag + push
#
# The script updates all version strings checked by check_version.sh,
# plus Maven dependency snippets in README files.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# --- Parse arguments ---
TARGET_VERSION=""
CREATE_TAG=false
DO_COMMIT=true
DO_PUSH=false

while [ $# -gt 0 ]; do
    case "$1" in
        --to)
            TARGET_VERSION="${2:-}"
            if [ -z "$TARGET_VERSION" ]; then
                echo "ERROR: --to requires a version argument (X.Y.Z)"
                exit 1
            fi
            shift 2
            ;;
        --create-tag)
            CREATE_TAG=true
            shift
            ;;
        --no-commit)
            DO_COMMIT=false
            shift
            ;;
        --push)
            DO_PUSH=true
            shift
            ;;
        -h|--help)
            sed -n '2,/^set /{ /^set /d; s/^# \?//; p; }' "$0"
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument '$1'"
            echo "Usage: $0 [--to X.Y.Z] [--create-tag] [--no-commit] [--push]"
            exit 1
            ;;
    esac
done

# --- Derive current version from source files ---
CURRENT_VERSION="$(grep -oE 'FCE_VERSION_MAJOR [0-9]+' "$REPO_ROOT/src/version.h" | awk '{print $2}').$(grep -oE 'FCE_VERSION_MINOR [0-9]+' "$REPO_ROOT/src/version.h" | awk '{print $2}').$(grep -oE 'FCE_VERSION_PATCH [0-9]+' "$REPO_ROOT/src/version.h" | awk '{print $2}')"

if ! echo "$CURRENT_VERSION" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+$'; then
    echo "ERROR: could not read current version from src/version.h"
    exit 1
fi

CURRENT_TAG="$(git -C "$REPO_ROOT" describe --tags --abbrev=0 2>/dev/null || true)"
TAG_VERSION="${CURRENT_TAG#v}"
if [ -n "$TAG_VERSION" ] && [ "$CURRENT_VERSION" != "$TAG_VERSION" ]; then
    echo "WARNING: source files ($CURRENT_VERSION) and git tag ($TAG_VERSION) differ."
    echo "         Using source file version as the base."
fi

# --- Determine target version ---
if [ -z "$TARGET_VERSION" ]; then
    IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT_VERSION"
    PATCH=$((PATCH + 1))
    TARGET_VERSION="$MAJOR.$MINOR.$PATCH"
    echo "Auto-bumping patch: $CURRENT_VERSION → $TARGET_VERSION"
else
    if ! echo "$TARGET_VERSION" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+$'; then
        echo "ERROR: '$TARGET_VERSION' is not a valid semver (expected X.Y.Z)"
        exit 1
    fi
    echo "Setting version: $CURRENT_VERSION → $TARGET_VERSION"
fi

if [ "$TARGET_VERSION" = "$CURRENT_VERSION" ]; then
    echo "ERROR: target version $TARGET_VERSION is the same as current version."
    exit 1
fi

IFS='.' read -r NEW_MAJOR NEW_MINOR NEW_PATCH <<< "$TARGET_VERSION"
IFS='.' read -r OLD_MAJOR OLD_MINOR OLD_PATCH <<< "$CURRENT_VERSION"

TODAY="$(date +%Y-%m-%d)"

# --- Apply replacements ---
echo ""
echo "Updating files ..."

# 1. src/version.h — #define FCE_VERSION_{MAJOR,MINOR,PATCH}
sed -i '' "s/FCE_VERSION_MAJOR $OLD_MAJOR/FCE_VERSION_MAJOR $NEW_MAJOR/" "$REPO_ROOT/src/version.h"
sed -i '' "s/FCE_VERSION_MINOR $OLD_MINOR/FCE_VERSION_MINOR $NEW_MINOR/" "$REPO_ROOT/src/version.h"
sed -i '' "s/FCE_VERSION_PATCH $OLD_PATCH/FCE_VERSION_PATCH $NEW_PATCH/" "$REPO_ROOT/src/version.h"
echo "  src/version.h"

# 2. README.md — header "**Version X.Y.Z**" and Maven <version> snippet
sed -i '' "s/Version $CURRENT_VERSION/Version $TARGET_VERSION/" "$REPO_ROOT/README.md"
sed -i '' "/<artifactId>fast-code-embed<\/artifactId>/{n;s|<version>$CURRENT_VERSION</version>|<version>$TARGET_VERSION</version>|;}" "$REPO_ROOT/README.md"
echo "  README.md"

# 3. java/pom.xml — <revision>X.Y.Z</revision>
sed -i '' "s|<revision>$CURRENT_VERSION</revision>|<revision>$TARGET_VERSION</revision>|" "$REPO_ROOT/java/pom.xml"
echo "  java/pom.xml"

# 4. java/README.md — header "**Version: X.Y.Z**" and Maven <version> snippet
sed -i '' "s/Version: $CURRENT_VERSION/Version: $TARGET_VERSION/" "$REPO_ROOT/java/README.md"
sed -i '' "/<artifactId>fast-code-embed<\/artifactId>/{n;s|<version>$CURRENT_VERSION</version>|<version>$TARGET_VERSION</version>|;}" "$REPO_ROOT/java/README.md"
echo "  java/README.md"

# 5. FastCodeEmbed.java — public static final String VERSION = "X.Y.Z"
sed -i '' "s/\"$CURRENT_VERSION\"/\"$TARGET_VERSION\"/" "$REPO_ROOT/java/src/main/java/io/github/nilsonsfj/fastcodeembed/FastCodeEmbed.java"
echo "  FastCodeEmbed.java"

# 6. CHANGELOG.md — insert new section after the header preamble
if ! grep -qF "[$TARGET_VERSION]" "$REPO_ROOT/CHANGELOG.md"; then
    sed -i '' "/^## \[$CURRENT_VERSION\]/i\\
## [$TARGET_VERSION] — $TODAY\\
\\
" "$REPO_ROOT/CHANGELOG.md"
    echo "  CHANGELOG.md (added [$TARGET_VERSION] section)"
else
    echo "  CHANGELOG.md (section already exists, skipping)"
fi

# --- Validate with check_version.sh ---
echo ""
echo "Running version validation ..."
if "$REPO_ROOT/scripts/check_version.sh" "v$TARGET_VERSION"; then
    echo ""
else
    echo ""
    echo "ERROR: version validation failed. Some files may not have been updated."
    exit 1
fi

# --- Commit ---
if [ "$DO_COMMIT" = true ]; then
    echo "Committing version bump ..."
    git -C "$REPO_ROOT" add \
        src/version.h \
        README.md \
        java/pom.xml \
        java/README.md \
        java/src/main/java/io/github/nilsonsfj/fastcodeembed/FastCodeEmbed.java \
        CHANGELOG.md
    git -C "$REPO_ROOT" commit -m "bump version to $TARGET_VERSION"
    echo ""
fi

# --- Create tag (after commit so it points to the right code) ---
if [ "$CREATE_TAG" = true ]; then
    echo "Creating git tag v$TARGET_VERSION ..."
    git -C "$REPO_ROOT" tag "v$TARGET_VERSION"
    echo "Tag v$TARGET_VERSION created."
    if [ "$DO_PUSH" = true ]; then
        echo ""
        echo "Pushing commit and tag ..."
        git -C "$REPO_ROOT" push origin main
        git -C "$REPO_ROOT" push origin "v$TARGET_VERSION"
    else
        echo ""
        echo "To push: git push origin main && git push origin v$TARGET_VERSION"
    fi
fi

# --- Push commit (when no tag, --push just pushes the commit) ---
if [ "$DO_PUSH" = true ] && [ "$CREATE_TAG" = false ]; then
    echo ""
    echo "Pushing commit ..."
    git -C "$REPO_ROOT" push origin main
fi

echo "Version bumped to $TARGET_VERSION successfully."
