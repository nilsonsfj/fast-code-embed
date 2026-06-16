# Security Policy

## Supported versions

fast-code-embed is an early-stage (0.0.x) library. Security fixes are applied to
the latest released version on the `main` branch. There is no long-term support
for older 0.0.x releases — please upgrade to the most recent version.

## Reporting a vulnerability

Please report suspected security vulnerabilities privately. Do **not** open a
public GitHub issue for security problems.

- Preferred: use GitHub's [private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
  ("Report a vulnerability" under the repository's **Security** tab).
- Alternatively, email the maintainer at **nilson.santos@gmail.com** with the
  details and, if possible, a minimal reproduction.

Please include:

- The affected version (or commit) and platform/compiler.
- A description of the issue and its impact.
- Steps to reproduce, ideally with a minimal test case.

## What to expect

- Acknowledgement of your report within a reasonable timeframe.
- An assessment of severity and, if confirmed, a fix on `main` followed by a
  patch release.
- Credit for the report if you would like it, once a fix is available.

## Scope notes

This library performs no network I/O and loads no untrusted code at runtime. Its
main attack surface is parsing/handling of caller-supplied corpus data and the
pretrained vector blob. The hash table is seeded per-instance to mitigate
hash-flooding from adversarial token inputs.
