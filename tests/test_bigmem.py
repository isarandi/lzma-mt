"""Tests for large memory (>4GB) handling - adapted from CPython's test_lzma.py

These tests verify correct handling of data larger than 4GB, which is
important for ensuring no integer overflow issues in size calculations.

These tests are skipped by default because they require:
- ~10GB+ of RAM
- Several minutes to complete

Run with: pytest tests/test_bigmem.py -v --run-bigmem
"""

import random
import pytest

import lzma_mt


# 4GB boundary - important for detecting 32-bit overflow issues
_4G = 4 * 1024 * 1024 * 1024


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "bigmem: mark test as requiring large memory (>4GB)"
    )


# Check if bigmem tests should run
import os
def _should_skip_bigmem():
    """Skip bigmem tests unless explicitly requested via env var."""
    return os.environ.get("RUN_BIGMEM", "0") != "1"


bigmem = pytest.mark.skipif(
    _should_skip_bigmem(),
    reason="Bigmem tests skipped by default. Use --run-bigmem to run."
)


class TestBigmem:
    """Test handling of large (>4GB) data.

    Adapted from CPython Lib/test/test_lzma.py bigmem tests.
    """

    @bigmem
    def test_compressor_bigmem(self):
        """Test compressing >4GB of data.

        Adapted from CPython test_compressor_bigmem.
        Uses memuse=2 (needs ~2x the data size in memory).
        """
        size = _4G + 100

        lzc = lzma_mt.LZMACompressor()
        cdata = lzc.compress(b"x" * size) + lzc.flush()
        ddata = lzma_mt.decompress(cdata)
        try:
            assert len(ddata) == size
            assert len(ddata.strip(b"x")) == 0
        finally:
            ddata = None

    @bigmem
    def test_decompressor_bigmem(self):
        """Test decompressing >4GB of data.

        Adapted from CPython test_decompressor_bigmem.
        Uses memuse=3 (needs ~3x the data size in memory).
        """
        size = _4G + 100

        lzd = lzma_mt.LZMADecompressor()
        blocksize = min(10 * 1024 * 1024, size)  # 10MB blocks
        block = random.randbytes(blocksize)
        try:
            input_data = block * ((size - 1) // blocksize + 1)
            input_data = input_data[:size]  # Trim to exact size
            cdata = lzma_mt.compress(input_data)
            ddata = lzd.decompress(cdata)
            assert ddata == input_data
        finally:
            input_data = cdata = ddata = None

    @bigmem
    def test_compress_function_bigmem(self):
        """Test one-shot compress/decompress of >4GB data."""
        size = _4G + 100

        try:
            input_data = b"y" * size
            cdata = lzma_mt.compress(input_data)
            ddata = lzma_mt.decompress(cdata)
            assert len(ddata) == size
            assert ddata == input_data
        finally:
            input_data = cdata = ddata = None


class TestLargishMem:
    """Test moderately large data that doesn't require bigmem skip.

    These tests use ~100MB-1GB of data and should run in normal test suites.
    """

    def test_compress_100mb(self):
        """Test compressing 100MB of data."""
        size = 100 * 1024 * 1024
        input_data = b"z" * size
        cdata = lzma_mt.compress(input_data, threads=0)
        ddata = lzma_mt.decompress(cdata, threads=0)
        assert len(ddata) == size
        assert ddata == input_data

    def test_streaming_100mb(self):
        """Test streaming 100MB through compressor."""
        size = 100 * 1024 * 1024
        chunk_size = 1024 * 1024  # 1MB chunks

        compressor = lzma_mt.LZMACompressor(threads=0)
        chunks = []

        # Compress in chunks
        data = b"w" * size
        for i in range(0, size, chunk_size):
            chunk = data[i:i + chunk_size]
            chunks.append(compressor.compress(chunk))
        chunks.append(compressor.flush())

        compressed = b''.join(chunks)
        decompressed = lzma_mt.decompress(compressed)
        assert decompressed == data

    def test_random_data_50mb(self):
        """Test compressing random (incompressible) data."""
        size = 50 * 1024 * 1024
        input_data = random.randbytes(size)
        cdata = lzma_mt.compress(input_data, threads=0)
        ddata = lzma_mt.decompress(cdata, threads=0)
        assert ddata == input_data
