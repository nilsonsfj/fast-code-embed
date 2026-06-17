/* * version.h — Library version for fast-code-embed.
 *
 * Semantic versioning: https://semver.org/
 * BUMP PATCH: bug fixes only.
 * BUMP MINOR: new features, backward-compatible.
 * BUMP MAJOR: breaking changes. */
#ifndef FCE_VERSION_H
#define FCE_VERSION_H

#define FCE_VERSION_MAJOR 0
#define FCE_VERSION_MINOR 0
#define FCE_VERSION_PATCH 11

/* stringify helpers */
#define FCE_VERSION_STR_(major, minor, patch) #major "." #minor "." #patch
#define FCE_VERSION_STR FCE_VERSION_STR_(FCE_VERSION_MAJOR, FCE_VERSION_MINOR, FCE_VERSION_PATCH)

/* Full version string. */
const char *fce_version(void);

#endif /* FCE_VERSION_H */
