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

# --- Require a clean tree when committing, so the post-commit guard below can
# reliably tell that every bumped file was actually committed (any leftover
# tracked modification then means a file was edited but not staged). ---
if [ "$DO_COMMIT" = true ] && \
   ! { git -C "$REPO_ROOT" diff --quiet && git -C "$REPO_ROOT" diff --cached --quiet; }; then
    echo "ERROR: working tree has uncommitted changes to tracked files."
    echo "       Commit or stash them first, or run with --no-commit."
    git -C "$REPO_ROOT" status --porcelain | grep -vE '^\?\?' | sed 's/^/         /'
    exit 1
fi

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

# Portable in-place sed: GNU sed uses `-i`, BSD/macOS sed requires `-i ''`.
if sed --version >/dev/null 2>&1; then
    SED_INPLACE=(sed -i)
else
    SED_INPLACE=(sed -i '')
fi

# 1. src/version.h — #define FCE_VERSION_{MAJOR,MINOR,PATCH}
"${SED_INPLACE[@]}" "s/FCE_VERSION_MAJOR $OLD_MAJOR/FCE_VERSION_MAJOR $NEW_MAJOR/" "$REPO_ROOT/src/version.h"
"${SED_INPLACE[@]}" "s/FCE_VERSION_MINOR $OLD_MINOR/FCE_VERSION_MINOR $NEW_MINOR/" "$REPO_ROOT/src/version.h"
"${SED_INPLACE[@]}" "s/FCE_VERSION_PATCH $OLD_PATCH/FCE_VERSION_PATCH $NEW_PATCH/" "$REPO_ROOT/src/version.h"
echo "  src/version.h"

# 2. README.md — header "**Version X.Y.Z**" and Maven <version> snippet
"${SED_INPLACE[@]}" "s/Version $CURRENT_VERSION/Version $TARGET_VERSION/" "$REPO_ROOT/README.md"
"${SED_INPLACE[@]}" "/<artifactId>fast-code-embed<\/artifactId>/{n;s|<version>$CURRENT_VERSION</version>|<version>$TARGET_VERSION</version>|;}" "$REPO_ROOT/README.md"
echo "  README.md"

# 3. java/pom.xml — <revision>X.Y.Z</revision>
"${SED_INPLACE[@]}" "s|<revision>$CURRENT_VERSION</revision>|<revision>$TARGET_VERSION</revision>|" "$REPO_ROOT/java/pom.xml"
echo "  java/pom.xml"

# 4. java/README.md — header "**Version: X.Y.Z**" and Maven <version> snippet
"${SED_INPLACE[@]}" "s/Version: $CURRENT_VERSION/Version: $TARGET_VERSION/" "$REPO_ROOT/java/README.md"
"${SED_INPLACE[@]}" "/<artifactId>fast-code-embed<\/artifactId>/{n;s|<version>$CURRENT_VERSION</version>|<version>$TARGET_VERSION</version>|;}" "$REPO_ROOT/java/README.md"
echo "  java/README.md"

# 5. FastCodeEmbed.java — public static final String VERSION = "X.Y.Z"
"${SED_INPLACE[@]}" "s/\"$CURRENT_VERSION\"/\"$TARGET_VERSION\"/" "$REPO_ROOT/java/src/main/java/io/github/nilsonsfj/fastcodeembed/FastCodeEmbed.java"
echo "  FastCodeEmbed.java"

# 5b. python/pyproject.toml — version = "X.Y.Z"
"${SED_INPLACE[@]}" "s/^version = \"$CURRENT_VERSION\"/version = \"$TARGET_VERSION\"/" "$REPO_ROOT/python/pyproject.toml"
echo "  python/pyproject.toml"

# 6. CHANGELOG.md — auto-generate the new version section from commit history.
# The release notes are the commit subjects since the previous release tag, so no
# manual changelog editing is ever required. The generated section is inserted
# right after the "## [Unreleased]" placeholder; release.yml later extracts it
# verbatim as the GitHub release body.
if grep -qF "[$TARGET_VERSION]" "$REPO_ROOT/CHANGELOG.md"; then
    echo "  CHANGELOG.md ([$TARGET_VERSION] section already exists, skipping)"
else
    # Range start: the previous release tag if it exists locally, else full history.
    RANGE=""
    if [ -n "$CURRENT_TAG" ] && \
       git -C "$REPO_ROOT" rev-parse -q --verify "refs/tags/$CURRENT_TAG^{commit}" >/dev/null 2>&1; then
        RANGE="$CURRENT_TAG..HEAD"
    fi

    # Commit subjects since the last release, minus merges and prior bump commits.
    # shellcheck disable=SC2086
    NOTES="$(git -C "$REPO_ROOT" log --no-merges --pretty='- %s' $RANGE \
        | grep -vE '^- bump version to ' || true)"
    if [ -z "$NOTES" ]; then
        NOTES="- Maintenance release (no notable changes)"
    fi

    # Build the section in a temp file, then splice it in after "## [Unreleased]".
    # Using getline (not awk -v) keeps commit subjects byte-exact regardless of
    # any backslashes they may contain.
    SECTION_TMP="$(mktemp)"
    {
        printf '\n## [%s] — %s\n\n' "$TARGET_VERSION" "$TODAY"
        printf '%s\n' "$NOTES"
    } > "$SECTION_TMP"

    CHANGELOG_TMP="$(mktemp)"
    awk -v sf="$SECTION_TMP" '
        { print }
        !done && index($0, "## [Unreleased]") == 1 {
            while ((getline line < sf) > 0) print line
            close(sf)
            done = 1
        }
    ' "$REPO_ROOT/CHANGELOG.md" > "$CHANGELOG_TMP"
    mv "$CHANGELOG_TMP" "$REPO_ROOT/CHANGELOG.md"
    rm -f "$SECTION_TMP"

    N_NOTES="$(printf '%s\n' "$NOTES" | grep -c '^- ')"
    echo "  CHANGELOG.md (generated [$TARGET_VERSION] section from $N_NOTES commit(s))"
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
        python/pyproject.toml \
        CHANGELOG.md
    git -C "$REPO_ROOT" commit -m "bump version to $TARGET_VERSION"

    # Guard: every file the sed steps edited must have been committed. A file
    # that was bumped but missing from the `git add` list above stays modified
    # in the working tree after the commit — fail here, before tagging/pushing,
    # instead of shipping a half-bumped release that only CI would catch.
    if ! { git -C "$REPO_ROOT" diff --quiet && git -C "$REPO_ROOT" diff --cached --quiet; }; then
        echo "ERROR: the version bump left tracked files uncommitted (a bumped"
        echo "       file is probably missing from the add list). NOT tagging."
        git -C "$REPO_ROOT" status --porcelain | grep -vE '^\?\?' | sed 's/^/         /'
        exit 1
    fi
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
