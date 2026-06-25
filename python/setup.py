"""Thin setup.py — metadata lives in pyproject.toml.

Its only job is to register the out-of-line cffi extension. cffi has no
pyproject table, so the `cffi_modules` hook must be declared here. It points at
the `ffibuilder` object in fast_code_embed/_build.py, which links the prebuilt
static archive (build/libfast_code_embed.a, blob included) into the extension.
"""

from setuptools import setup

setup(
    packages=["fast_code_embed"],
    cffi_modules=["fast_code_embed/_build.py:ffibuilder"],
    # The extension is built against the stable ABI (see _build.py), so tag the
    # wheel cp39-abi3 — one wheel per platform covers every Python >= 3.9.
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
