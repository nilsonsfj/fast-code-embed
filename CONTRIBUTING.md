# Contributing to fast-code-embed

Thanks for your interest in contributing!

## Building

```bash
make            # build build/libfast_code_embed.a
make test       # run the test suite
make test-asan  # run tests with AddressSanitizer
```

For the Java binding:

```bash
make            # build C library first
cd java && ./build.sh
```

## Testing

C tests: `make test` (64 tests)
Java tests: `cd java && ./build.sh` (21 tests)
Sanitizers: `make test-asan`

All tests must pass before submitting a pull request.

## Code style

- C11 (`-std=c11`), no C23 features
- Compiler warnings are errors (`-Wall -Wextra -Wpedantic`)
- No internal review/task ID prefixes in comments (those were used during development)
- Keep comments explaining *why*, not *what*

## Pull requests

1. Fork the repo and create a feature branch
2. Make your changes, ensuring tests pass
3. Open a PR with a clear description of the change

## Reporting issues

Open a GitHub issue with:
- What you expected
- What actually happened
- Steps to reproduce
