"""
lzma_mt - Multi-threaded LZMA/XZ compression

A drop-in replacement for Python's lzma module with multi-threading support.
"""

from lzma_mt.lzma_mt import (
    compress,
    decompress,
    LZMACompressor,
    LZMADecompressor,
    CHECK_NONE,
    CHECK_CRC32,
    CHECK_CRC64,
    CHECK_SHA256,
    PRESET_DEFAULT,
    PRESET_EXTREME,
)

try:
    from lzma_mt._version import __version__
except ImportError:
    __version__ = "0.0.0.dev0"

__all__ = [
    "compress",
    "decompress",
    "LZMACompressor",
    "LZMADecompressor",
    "CHECK_NONE",
    "CHECK_CRC32",
    "CHECK_CRC64",
    "CHECK_SHA256",
    "PRESET_DEFAULT",
    "PRESET_EXTREME",
    "__version__",
]
