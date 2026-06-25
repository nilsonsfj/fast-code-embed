#!/usr/bin/env bash
# check_version.sh — Verify all version strings in the codebase match the given tag.
#
# Usage:
#   ./scripts/check_version.sh v0.0.5   # check against tag
#   ./scripts/check_version.sh           # auto-detect from git describe
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ $# -ge 1 ]; then
    TAG="$1"
else
    TAG="$(git -C "$REPO_ROOT" describe --tags --exact-match 2>/dev/null || true)"
    if [ -z "$TAG" ]; then
        echo "ERROR: no tag provided and HEAD is not a tagged commit."
        echo "Usage: $0 v0.0.X"
        exit 1
    fi
fi

# Extract version from tag (strip leading v)
VERSION="${TAG#v}"
if ! echo "$VERSION" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+$'; then
    echo "ERROR: '$TAG' is not a valid semver tag (expected vX.Y.Z)"
    exit 1
fi

IFS='.' read -r MAJOR MINOR PATCH <<< "$VERSION"

echo "Checking version strings match $VERSION ..."
ERRORS=0

check() {
    local file="$1"
    local pattern="$2"
    local label="$3"
    if ! grep -qE "$pattern" "$REPO_ROOT/$file" 2>/dev/null; then
        echo "  FAIL: $label"
        ERRORS=$((ERRORS + 1))
    else
        echo "  OK:   $label"
    fi
}

check "src/version.h" \
    "FCE_VERSION_MAJOR $MAJOR" \
    "src/version.h MAJOR"

check "src/version.h" \
    "FCE_VERSION_MINOR $MINOR" \
    "src/version.h MINOR"

check "src/version.h" \
    "FCE_VERSION_PATCH $PATCH" \
    "src/version.h PATCH"

check "README.md" \
    "Version $VERSION" \
    "README.md"

check "java/pom.xml" \
    "<revision>$VERSION</revision>" \
    "java/pom.xml"

check "java/README.md" \
    "Version: $VERSION" \
    "java/README.md"

check "java/src/main/java/io/github/nilsonsfj/fastcodeembed/FastCodeEmbed.java" \
    "\"$VERSION\"" \
    "FastCodeEmbed.java VERSION"

check "python/pyproject.toml" \
    "^version = \"$VERSION\"" \
    "python/pyproject.toml"

check "README.md" \
    "<version>$VERSION</version>" \
    "README.md (Maven snippet)"

check "java/README.md" \
    "<version>$VERSION</version>" \
    "java/README.md (Maven snippet)"

check "CHANGELOG.md" \
    "\[$VERSION\]" \
    "CHANGELOG.md"

# The release workflow uses the body of the matching changelog section as the
# GitHub release notes, so an empty section ships a blank release. Verify the
# section for this version exists AND has non-whitespace content.
CHANGELOG_BODY="$(awk -v ver="$VERSION" '
    $0 ~ "^## \\[" ver "\\]" { found=1; next }
    found && /^## \[/        { exit }
    found                    { print }
' "$REPO_ROOT/CHANGELOG.md" 2>/dev/null)"
if [ -z "$(echo "$CHANGELOG_BODY" | tr -d '[:space:]')" ]; then
    echo "  FAIL: CHANGELOG.md ([$VERSION] section is empty)"
    ERRORS=$((ERRORS + 1))
else
    echo "  OK:   CHANGELOG.md ([$VERSION] section has content)"
fi

echo ""
if [ "$ERRORS" -gt 0 ]; then
    echo "FAILED: $ERRORS version mismatch(es) found."
    echo "Bump all version strings to $VERSION before tagging."
    exit 1
else
    echo "OK: all version strings match $VERSION."
fi
