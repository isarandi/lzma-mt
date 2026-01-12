"""Tests for decompressor input buffer handling - adapted from CPython's test_lzma.py

These tests verify correct behavior when the decompressor's internal input
buffer is reused, extended, or has data moved within it.
"""

import pytest

import lzma_mt


# Generate test data - needs to be large enough that compressed size > 300 bytes
# for the input buffer tests to work properly
INPUT = (b"The quick brown fox jumps over the lazy dog. " * 500 +
         bytes(range(256)) * 20)  # Add some incompressible data
COMPRESSED_XZ = lzma_mt.compress(INPUT)
assert len(COMPRESSED_XZ) > 350, f"Compressed size {len(COMPRESSED_XZ)} too small for tests"


class TestDecompressorInputBuffer:
    """Test decompressor input buffer edge cases.

    Adapted from CPython Lib/test/test_lzma.py::CompressorDecompressorTestCase
    """

    def test_decompressor_inputbuf_1(self):
        """Test reusing input buffer after moving existing contents to beginning.

        Adapted from CPython test_decompressor_inputbuf_1
        """
        lzd = lzma_mt.LZMADecompressor()
        out = []

        # Create input buffer and fill it
        assert lzd.decompress(COMPRESSED_XZ[:100], max_length=0) == b''

        # Retrieve some results, freeing capacity at beginning of input buffer
        out.append(lzd.decompress(b'', max_length=2))

        # Add more data that fits into input buffer after moving existing
        # data to beginning
        out.append(lzd.decompress(COMPRESSED_XZ[100:105], max_length=15))

        # Decompress rest of data
        out.append(lzd.decompress(COMPRESSED_XZ[105:]))
        assert b''.join(out) == INPUT

    def test_decompressor_inputbuf_2(self):
        """Test reusing input buffer by appending data at the end right away.

        Adapted from CPython test_decompressor_inputbuf_2
        """
        lzd = lzma_mt.LZMADecompressor()
        out = []

        # Create input buffer and empty it
        assert lzd.decompress(COMPRESSED_XZ[:200], max_length=0) == b''
        out.append(lzd.decompress(b''))

        # Fill buffer with new data
        out.append(lzd.decompress(COMPRESSED_XZ[200:280], max_length=2))

        # Append some more data, not enough to require resize
        out.append(lzd.decompress(COMPRESSED_XZ[280:300], max_length=2))

        # Decompress rest of data
        out.append(lzd.decompress(COMPRESSED_XZ[300:]))
        assert b''.join(out) == INPUT

    def test_decompressor_inputbuf_3(self):
        """Test reusing input buffer after extending it.

        Adapted from CPython test_decompressor_inputbuf_3
        """
        lzd = lzma_mt.LZMADecompressor()
        out = []

        # Create almost full input buffer
        out.append(lzd.decompress(COMPRESSED_XZ[:200], max_length=5))

        # Add even more data to it, requiring resize
        out.append(lzd.decompress(COMPRESSED_XZ[200:300], max_length=5))

        # Decompress rest of data
        out.append(lzd.decompress(COMPRESSED_XZ[300:]))
        assert b''.join(out) == INPUT

    def test_decompressor_inputbuf_empty_then_data(self):
        """Test feeding empty bytes then real data."""
        lzd = lzma_mt.LZMADecompressor()
        out = []

        # Feed empty data multiple times
        out.append(lzd.decompress(b''))
        out.append(lzd.decompress(b''))

        # Now feed real data
        out.append(lzd.decompress(COMPRESSED_XZ))
        assert b''.join(out) == INPUT

    def test_decompressor_inputbuf_byte_by_byte(self):
        """Test decompressing one byte at a time."""
        lzd = lzma_mt.LZMADecompressor()
        out = []

        for i in range(len(COMPRESSED_XZ)):
            chunk = lzd.decompress(COMPRESSED_XZ[i:i+1])
            out.append(chunk)
            if lzd.eof:
                break

        assert b''.join(out) == INPUT

    def test_decompressor_inputbuf_max_length_zero(self):
        """Test max_length=0 buffers input without producing output."""
        lzd = lzma_mt.LZMADecompressor()

        # Feed all data with max_length=0
        result = lzd.decompress(COMPRESSED_XZ, max_length=0)
        assert result == b''
        assert not lzd.eof
        assert not lzd.needs_input

        # Now decompress everything
        result = lzd.decompress(b'')
        assert result == INPUT
        assert lzd.eof
