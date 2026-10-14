"""Pytest configuration and Windows DLL initialization."""

# Fix for Windows C++ CRT conflict:
# sentencepiece must be imported before pyarrow to avoid access violation crash on Windows.
try:
    import sentencepiece  # noqa: F401
except ImportError:
    pass
