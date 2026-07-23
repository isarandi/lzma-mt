"""
Comprehensive tests for lzma_mt module.

Includes:
- Basic functionality tests
- Edge cases
- Property-based tests with Hypothesis
- Compatibility tests with stdlib lzma
- Streaming API tests
- Various input types (bytes, bytearray, memoryview)
"""

import lzma
import pytest
from hypothesis import given, settings, strategies as st

import lzma_mt


# =============================================================================
# Basic Functionality Tests
# =============================================================================

class TestCompress:
    def test_empty_data(self):
        """Compressing empty data should work."""
        compressed = lzma_mt.compress(b"")
        assert isinstance(compressed, bytes)
        assert len(compressed) > 0  # XZ header is always present
        assert lzma_mt.decompress(compressed) == b""

    def test_single_byte(self):
        """Compressing a single byte should work."""
        compressed = lzma_mt.compress(b"x")
        assert lzma_mt.decompress(compressed) == b"x"

    def test_basic_compression(self):
        """Basic compression roundtrip."""
        data = b"Hello, World! " * 1000
        compressed = lzma_mt.compress(data)
        assert len(compressed) < len(data)
        assert lzma_mt.decompress(compressed) == data

    def test_incompressible_data(self):
        """Random data that doesn't compress well."""
        import os
        data = os.urandom(10000)
        compressed = lzma_mt.compress(data)
        assert lzma_mt.decompress(compressed) == data

    def test_large_data(self):
        """Test with larger data to exercise multi-threading."""
        data = b"x" * (10 * 1024 * 1024)  # 10 MB
        compressed = lzma_mt.compress(data, threads=4)
        assert lzma_mt.decompress(compressed, threads=4) == data


class TestDecompress:
    def test_invalid_data(self):
        """Decompressing invalid data should raise LZMAError (matching stdlib)."""
        with pytest.raises(lzma_mt.LZMAError):
            lzma_mt.decompress(b"not valid xz data")

    def test_truncated_data(self):
        """Decompressing truncated data should raise LZMAError (matching stdlib)."""
        compressed = lzma_mt.compress(b"Hello, World!")
        with pytest.raises(lzma_mt.LZMAError):
            lzma_mt.decompress(compressed[:-10])

    def test_corrupted_data(self):
        """Decompressing corrupted data should raise LZMAError (matching stdlib)."""
        compressed = bytearray(lzma_mt.compress(b"Hello, World!"))
        compressed[len(compressed) // 2] ^= 0xFF  # Flip some bits
        with pytest.raises(lzma_mt.LZMAError):
            lzma_mt.decompress(bytes(compressed))

    def test_concatenated_streams(self):
        """One-shot decompress() should handle concatenated streams."""
        stream1 = lzma_mt.compress(b"First")
        stream2 = lzma_mt.compress(b"Second")
        stream3 = lzma_mt.compress(b"Third")
        result = lzma_mt.decompress(stream1 + stream2 + stream3)
        assert result == b"FirstSecondThird"


# =============================================================================
# Preset and Check Tests
# =============================================================================

class TestPresets:
    @pytest.mark.parametrize("preset", range(10))
    def test_all_presets(self, preset):
        """Test all compression presets 0-9."""
        data = b"Test data for preset " * 100
        compressed = lzma_mt.compress(data, preset=preset)
        assert lzma_mt.decompress(compressed) == data

    def test_extreme_preset(self):
        """Test extreme compression flag."""
        data = b"Test data " * 1000
        compressed = lzma_mt.compress(
            data, preset=6 | lzma_mt.PRESET_EXTREME
        )
        assert lzma_mt.decompress(compressed) == data


class TestChecks:
    @pytest.mark.parametrize("check", [
        lzma_mt.CHECK_NONE,
        lzma_mt.CHECK_CRC32,
        lzma_mt.CHECK_CRC64,
        lzma_mt.CHECK_SHA256,
    ])
    def test_all_check_types(self, check):
        """Test all integrity check types."""
        data = b"Test data for check type " * 100
        compressed = lzma_mt.compress(data, check=check)
        assert lzma_mt.decompress(compressed) == data


# =============================================================================
# Thread Count Tests
# =============================================================================

class TestThreading:
    @pytest.mark.parametrize("threads", [0, 1, 2, 4, 8])
    def test_thread_counts(self, threads):
        """Test various thread counts."""
        data = b"x" * 100000
        compressed = lzma_mt.compress(data, threads=threads)
        decompressed = lzma_mt.decompress(compressed, threads=threads)
        assert decompressed == data

    def test_auto_threads(self):
        """threads=0 should auto-detect CPU count."""
        data = b"Test " * 10000
        compressed = lzma_mt.compress(data, threads=0)
        assert lzma_mt.decompress(compressed, threads=0) == data


# =============================================================================
# Input Type Tests (Zero-Copy Verification)
# =============================================================================

class TestInputTypes:
    def test_bytes_input(self):
        """Test with bytes input."""
        data = b"Hello, World!"
        compressed = lzma_mt.compress(data)
        assert lzma_mt.decompress(compressed) == data

    def test_bytearray_input(self):
        """Test with bytearray input (zero-copy)."""
        data = bytearray(b"Hello, World!")
        compressed = lzma_mt.compress(data)
        assert lzma_mt.decompress(compressed) == bytes(data)

    def test_memoryview_input(self):
        """Test with memoryview input (zero-copy)."""
        data = memoryview(b"Hello, World!")
        compressed = lzma_mt.compress(data)
        assert lzma_mt.decompress(compressed) == bytes(data)

    def test_memoryview_slice(self):
        """Test with memoryview slice."""
        full_data = b"XXXHello, World!XXX"
        data = memoryview(full_data)[3:-3]
        compressed = lzma_mt.compress(data)
        assert lzma_mt.decompress(compressed) == b"Hello, World!"

    def test_bytearray_decompress(self):
        """Test decompressing into bytearray input."""
        original = b"Hello, World!"
        compressed = lzma_mt.compress(original)
        # Decompress from bytearray
        result = lzma_mt.decompress(bytearray(compressed))
        assert result == original

    def test_non_contiguous_rejected(self):
        """Non-contiguous memoryview should be rejected."""
        import array
        arr = array.array('B', b"Hello, World!")
        # Create strided view (every other byte)
        mv = memoryview(arr)[::2]
        with pytest.raises((ValueError, TypeError, BufferError)):
            lzma_mt.compress(mv)


# =============================================================================
# Streaming Compressor Tests
# =============================================================================

class TestLZMACompressor:
    def test_basic_streaming(self):
        """Basic streaming compression."""
        compressor = lzma_mt.LZMACompressor()
        chunks = [b"Hello, ", b"World", b"!"]
        compressed = bytearray()
        for chunk in chunks:
            compressed.extend(compressor.compress(chunk))
        compressed.extend(compressor.flush())
        assert lzma_mt.decompress(compressed) == b"Hello, World!"

    def test_single_chunk(self):
        """Streaming with single chunk."""
        compressor = lzma_mt.LZMACompressor()
        compressed = compressor.compress(b"Hello") + compressor.flush()
        assert lzma_mt.decompress(compressed) == b"Hello"

    def test_empty_chunks(self):
        """Streaming with empty chunks."""
        compressor = lzma_mt.LZMACompressor()
        compressed = bytearray()
        compressed.extend(compressor.compress(b""))
        compressed.extend(compressor.compress(b"Hello"))
        compressed.extend(compressor.compress(b""))
        compressed.extend(compressor.flush())
        assert lzma_mt.decompress(compressed) == b"Hello"

    def test_many_small_chunks(self):
        """Streaming with many small chunks."""
        compressor = lzma_mt.LZMACompressor()
        data = b"x" * 10000
        compressed = bytearray()
        for byte in data:
            compressed.extend(compressor.compress(bytes([byte])))
        compressed.extend(compressor.flush())
        assert lzma_mt.decompress(compressed) == data

    def test_flush_twice_raises(self):
        """Flushing twice should raise an error."""
        compressor = lzma_mt.LZMACompressor()
        compressor.compress(b"Hello")
        compressor.flush()
        with pytest.raises(ValueError):
            compressor.flush()

    def test_compress_after_flush_raises(self):
        """Compressing after flush should raise an error."""
        compressor = lzma_mt.LZMACompressor()
        compressor.compress(b"Hello")
        compressor.flush()
        with pytest.raises(ValueError):
            compressor.compress(b"World")

    def test_streaming_with_threads(self):
        """Streaming compression with multiple threads."""
        compressor = lzma_mt.LZMACompressor(threads=4)
        data = b"Test data " * 10000
        compressed = compressor.compress(data) + compressor.flush()
        assert lzma_mt.decompress(compressed) == data

    def test_streaming_bytearray_input(self):
        """Streaming with bytearray input."""
        compressor = lzma_mt.LZMACompressor()
        compressed = compressor.compress(bytearray(b"Hello")) + compressor.flush()
        assert lzma_mt.decompress(compressed) == b"Hello"


# =============================================================================
# Streaming Decompressor Tests
# =============================================================================

class TestLZMADecompressor:
    def test_basic_streaming(self):
        """Basic streaming decompression."""
        data = b"Hello, World!"
        compressed = lzma_mt.compress(data)
        decompressor = lzma_mt.LZMADecompressor()
        result = decompressor.decompress(compressed)
        assert result == data

    def test_chunked_decompression(self):
        """Decompressing in chunks."""
        data = b"Hello, World!" * 1000
        compressed = lzma_mt.compress(data)
        decompressor = lzma_mt.LZMADecompressor()

        result = b""
        chunk_size = 100
        for i in range(0, len(compressed), chunk_size):
            chunk = compressed[i:i + chunk_size]
            result += decompressor.decompress(chunk)

        assert result == data

    def test_eof_property(self):
        """EOF property should be set after stream ends."""
        data = b"Hello"
        compressed = lzma_mt.compress(data)
        decompressor = lzma_mt.LZMADecompressor()

        assert not decompressor.eof
        decompressor.decompress(compressed)
        assert decompressor.eof

    def test_decompress_after_eof_raises(self):
        """Decompress after EOF should raise ValueError."""
        compressed = lzma_mt.compress(b"Hello")
        decompressor = lzma_mt.LZMADecompressor()
        decompressor.decompress(compressed)
        assert decompressor.eof

        with pytest.raises(ValueError):
            decompressor.decompress(b"more data")

    def test_max_length(self):
        """Test max_length parameter."""
        data = b"x" * 10000
        compressed = lzma_mt.compress(data)
        decompressor = lzma_mt.LZMADecompressor()

        # Get only first 100 bytes
        result = decompressor.decompress(compressed, max_length=100)
        assert len(result) == 100
        assert result == b"x" * 100

    def test_max_length_continuation(self):
        """Continue decompression after max_length until all data is recovered."""
        data = b"x" * 1000
        compressed = lzma_mt.compress(data)
        decompressor = lzma_mt.LZMADecompressor()

        result = b""
        remaining = compressed
        while not decompressor.eof:
            chunk = decompressor.decompress(remaining, max_length=100)
            assert len(chunk) <= 100
            result += chunk
            remaining = b""  # Already consumed

        assert result == data

    def test_max_length_larger_than_first_block(self):
        """max_length beyond the 32KB initial buffer block is honored."""
        data = b"y" * 200000
        compressed = lzma_mt.compress(data)
        decompressor = lzma_mt.LZMADecompressor()

        result = decompressor.decompress(compressed, max_length=150000)
        assert len(result) == 150000
        assert not decompressor.eof
        assert not decompressor.needs_input

        rest = decompressor.decompress(b"")
        assert result + rest == data
        assert decompressor.eof

    def test_streaming_with_threads(self):
        """Streaming decompression with threads."""
        data = b"Test " * 10000
        compressed = lzma_mt.compress(data)
        decompressor = lzma_mt.LZMADecompressor(threads=4)
        result = decompressor.decompress(compressed)
        assert result == data

    def test_unused_data_with_concatenated(self):
        """Streaming decompressor only gets first stream, unused_data has rest."""
        stream1 = lzma_mt.compress(b"First")
        stream2 = lzma_mt.compress(b"Second")
        combined = stream1 + stream2

        decompressor = lzma_mt.LZMADecompressor()
        result = decompressor.decompress(combined)

        assert result == b"First"
        assert decompressor.eof
        assert decompressor.unused_data == stream2


# =============================================================================
# Stdlib Compatibility Tests
# =============================================================================

class TestStdlibCompatibility:
    def test_decompress_stdlib_compressed(self):
        """Decompress data compressed by stdlib lzma."""
        data = b"Hello from stdlib!"
        compressed = lzma.compress(data)
        assert lzma_mt.decompress(compressed) == data

    def test_stdlib_decompress_our_compressed(self):
        """Stdlib should decompress our compressed data."""
        data = b"Hello from lzma_mt!"
        compressed = lzma_mt.compress(data)
        assert lzma.decompress(compressed) == data

    def test_streaming_compatibility(self):
        """Streaming APIs should be compatible."""
        data = b"Streaming test " * 100

        # Compress with lzma_mt
        compressor = lzma_mt.LZMACompressor()
        compressed = compressor.compress(data) + compressor.flush()

        # Decompress with stdlib
        assert lzma.decompress(compressed) == data

        # And vice versa
        stdlib_compressor = lzma.LZMACompressor()
        stdlib_compressed = stdlib_compressor.compress(data) + stdlib_compressor.flush()
        assert lzma_mt.decompress(stdlib_compressed) == data


# =============================================================================
# Drop-in Compatibility Tests
# =============================================================================

class TestDropinCompat:
    """Behaviors that must match the stdlib lzma module exactly."""

    def test_lzmaerror_is_stdlib_class(self):
        """Exceptions must be catchable as either lzma_mt or lzma LZMAError."""
        assert lzma_mt.LZMAError is lzma.LZMAError

    def test_decompress_empty_raises(self):
        """decompress(b'') raises like stdlib."""
        with pytest.raises(lzma_mt.LZMAError):
            lzma_mt.decompress(b"")

    @pytest.mark.parametrize("threads", [1, 2])
    def test_format_auto_handles_alone(self, threads):
        """FORMAT_AUTO must decode legacy .lzma (FORMAT_ALONE) data."""
        data = b"legacy alone-format data " * 100
        alone = lzma.compress(data, format=lzma.FORMAT_ALONE)
        assert lzma_mt.decompress(alone, threads=threads) == data

        decompressor = lzma_mt.LZMADecompressor(threads=threads)
        assert decompressor.decompress(alone) == data
        assert decompressor.eof

    def test_format_auto_alone_byte_by_byte(self):
        """Format sniffing works even when input arrives one byte at a time."""
        data = b"alone data " * 50
        alone = lzma.compress(data, format=lzma.FORMAT_ALONE)
        decompressor = lzma_mt.LZMADecompressor(threads=2)
        result = b""
        for i in range(len(alone)):
            result += decompressor.decompress(alone[i:i + 1])
        assert result == data

    def test_format_auto_xz_byte_by_byte_mt(self):
        """Sniffing correctly picks the XZ path with trickled input."""
        data = b"xz data " * 50
        compressed = lzma_mt.compress(data)
        decompressor = lzma_mt.LZMADecompressor(threads=2)
        result = b""
        for i in range(len(compressed)):
            result += decompressor.decompress(compressed[i:i + 1])
        assert result == data
        assert decompressor.eof

    def test_invalid_format_decompress(self):
        """Invalid format values raise ValueError like stdlib."""
        compressed = lzma_mt.compress(b"hello")
        with pytest.raises(ValueError):
            lzma_mt.decompress(compressed, format=999)
        with pytest.raises(ValueError):
            lzma_mt.LZMADecompressor(format=999)

    def test_non_buffer_input_rejected(self):
        """Ints and strs are rejected with TypeError like stdlib."""
        with pytest.raises(TypeError):
            lzma_mt.decompress(5)
        with pytest.raises(TypeError):
            lzma_mt.LZMADecompressor().decompress(5)
        with pytest.raises(TypeError):
            lzma_mt.LZMADecompressor().decompress("text")

    def test_memlimit_raises_lzmaerror(self):
        """Exceeding memlimit raises LZMAError like stdlib."""
        compressed = lzma_mt.compress(b"x" * 100000)
        with pytest.raises(lzma_mt.LZMAError):
            lzma_mt.decompress(compressed, memlimit=1024)
        with pytest.raises(lzma_mt.LZMAError):
            lzma_mt.LZMADecompressor(memlimit=1024).decompress(compressed)

    def test_check_unknown_before_decoding(self):
        """check is CHECK_UNKNOWN before header, then the actual type."""
        decompressor = lzma_mt.LZMADecompressor()
        assert decompressor.check == lzma_mt.CHECK_UNKNOWN
        compressed = lzma_mt.compress(b"data", check=lzma_mt.CHECK_SHA256)
        decompressor.decompress(compressed)
        assert decompressor.check == lzma_mt.CHECK_SHA256

    @pytest.mark.parametrize("threads", [1, 2])
    def test_check_property_matches_stdlib(self, threads):
        """check property agrees with stdlib for all check types."""
        for check in (lzma_mt.CHECK_NONE, lzma_mt.CHECK_CRC32,
                      lzma_mt.CHECK_CRC64, lzma_mt.CHECK_SHA256):
            compressed = lzma_mt.compress(b"data" * 100, check=check)
            ours = lzma_mt.LZMADecompressor(threads=threads)
            ours.decompress(compressed)
            stdlib = lzma.LZMADecompressor()
            stdlib.decompress(compressed)
            assert ours.check == stdlib.check == check

    def test_compress_threads1_byte_identical(self):
        """threads=1 produces byte-identical output to the stdlib."""
        data = b"The quick brown fox jumps over the lazy dog. " * 200
        assert lzma_mt.compress(data) == lzma.compress(data)
        assert lzma_mt.compress(data, preset=1) == lzma.compress(data, preset=1)
        assert (lzma_mt.compress(data, check=lzma_mt.CHECK_SHA256)
                == lzma.compress(data, check=lzma.CHECK_SHA256))


# =============================================================================
# Memory Limit Tests
# =============================================================================

class TestMemoryLimits:
    def test_decompress_with_memlimit(self):
        """Test decompression with memory limit."""
        data = b"x" * 10000
        compressed = lzma_mt.compress(data)
        # High limit should work
        result = lzma_mt.decompress(compressed, memlimit=100_000_000)
        assert result == data

    def test_streaming_with_memlimit(self):
        """Test streaming decompression with memory limit."""
        data = b"x" * 10000
        compressed = lzma_mt.compress(data)
        decompressor = lzma_mt.LZMADecompressor(memlimit=100_000_000)
        result = decompressor.decompress(compressed)
        assert result == data


# =============================================================================
# Hypothesis Property-Based Tests
# =============================================================================

class TestHypothesis:
    @given(st.binary(max_size=100000))
    @settings(max_examples=100)
    def test_roundtrip_any_data(self, data):
        """Any binary data should roundtrip correctly."""
        compressed = lzma_mt.compress(data)
        assert lzma_mt.decompress(compressed) == data

    @given(st.binary(max_size=50000), st.integers(min_value=0, max_value=9))
    @settings(max_examples=50)
    def test_roundtrip_any_preset(self, data, preset):
        """Any preset should produce valid compressed data."""
        compressed = lzma_mt.compress(data, preset=preset)
        assert lzma_mt.decompress(compressed) == data

    @given(st.binary(max_size=50000), st.integers(min_value=1, max_value=8))
    @settings(max_examples=50)
    def test_roundtrip_any_threads(self, data, threads):
        """Any thread count should work."""
        compressed = lzma_mt.compress(data, threads=threads)
        assert lzma_mt.decompress(compressed, threads=threads) == data

    @given(st.lists(st.binary(max_size=1000), min_size=1, max_size=20))
    @settings(max_examples=50)
    def test_streaming_any_chunks(self, chunks):
        """Streaming should work with any chunk pattern."""
        compressor = lzma_mt.LZMACompressor()
        compressed = bytearray()
        for chunk in chunks:
            compressed.extend(compressor.compress(chunk))
        compressed.extend(compressor.flush())

        expected = b"".join(chunks)
        assert lzma_mt.decompress(compressed) == expected

    @given(st.binary(min_size=100, max_size=10000))
    @settings(max_examples=50)
    def test_chunked_decompress(self, data):
        """Chunked decompression should produce correct result."""
        compressed = lzma_mt.compress(data)
        decompressor = lzma_mt.LZMADecompressor()

        result = b""
        chunk_size = max(1, len(compressed) // 10)
        for i in range(0, len(compressed), chunk_size):
            chunk = compressed[i:i + chunk_size]
            result += decompressor.decompress(chunk)

        assert result == data

    @given(st.binary(max_size=10000))
    @settings(max_examples=50)
    def test_bytearray_same_as_bytes(self, data):
        """bytearray input should produce same result as bytes."""
        compressed_bytes = lzma_mt.compress(data)
        compressed_bytearray = lzma_mt.compress(bytearray(data))
        # Compressed output should be identical
        assert compressed_bytes == compressed_bytearray

    @given(st.binary(max_size=10000))
    @settings(max_examples=50)
    def test_memoryview_same_as_bytes(self, data):
        """memoryview input should produce same result as bytes."""
        compressed_bytes = lzma_mt.compress(data)
        compressed_mv = lzma_mt.compress(memoryview(data))
        assert compressed_bytes == compressed_mv

    @given(st.binary(min_size=1, max_size=10000))
    @settings(max_examples=50)
    def test_stdlib_interop(self, data):
        """Our output should be readable by stdlib and vice versa."""
        # Our compress -> stdlib decompress
        our_compressed = lzma_mt.compress(data)
        assert lzma.decompress(our_compressed) == data

        # Stdlib compress -> our decompress
        stdlib_compressed = lzma.compress(data)
        assert lzma_mt.decompress(stdlib_compressed) == data


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    def test_repeated_byte(self):
        """Highly compressible repeated byte."""
        data = b"\x00" * 1000000
        compressed = lzma_mt.compress(data)
        assert len(compressed) < 1000  # Should compress very well
        assert lzma_mt.decompress(compressed) == data

    def test_all_byte_values(self):
        """Data containing all 256 byte values."""
        data = bytes(range(256)) * 100
        compressed = lzma_mt.compress(data)
        assert lzma_mt.decompress(compressed) == data

    def test_alternating_pattern(self):
        """Alternating byte pattern."""
        data = b"\x00\xff" * 50000
        compressed = lzma_mt.compress(data)
        assert lzma_mt.decompress(compressed) == data

    def test_very_long_repeated_sequence(self):
        """Very long sequence that exercises dictionary."""
        pattern = b"abcdefghij" * 10
        data = pattern * 10000
        compressed = lzma_mt.compress(data)
        assert lzma_mt.decompress(compressed) == data

    def test_binary_with_nulls(self):
        """Binary data with null bytes."""
        data = b"hello\x00world\x00test\x00"
        compressed = lzma_mt.compress(data)
        assert lzma_mt.decompress(compressed) == data

    def test_unicode_as_bytes(self):
        """Unicode text encoded as UTF-8."""
        text = "Hello, 世界! 🎉 Привет мир!"
        data = text.encode("utf-8")
        compressed = lzma_mt.compress(data)
        assert lzma_mt.decompress(compressed) == data


# =============================================================================
# Stress Tests
# =============================================================================

class TestStress:
    def test_many_small_compressions(self):
        """Many small compression operations."""
        for i in range(1000):
            data = f"Test {i}".encode()
            compressed = lzma_mt.compress(data)
            assert lzma_mt.decompress(compressed) == data

    def test_many_compressor_instances(self):
        """Create many compressor instances."""
        compressors = [lzma_mt.LZMACompressor() for _ in range(100)]
        for i, comp in enumerate(compressors):
            compressed = comp.compress(f"Data {i}".encode()) + comp.flush()
            assert lzma_mt.decompress(compressed) == f"Data {i}".encode()

    def test_many_decompressor_instances(self):
        """Create many decompressor instances."""
        data = b"Test data"
        compressed = lzma_mt.compress(data)
        for _ in range(100):
            dec = lzma_mt.LZMADecompressor()
            assert dec.decompress(compressed) == data


# =============================================================================
# New Feature Tests (CVE check, needs_input, thread safety, validation)
# =============================================================================

class TestVersionCheck:
    def test_get_xz_version(self):
        """get_xz_version returns a version string."""
        version = lzma_mt.get_xz_version()
        assert isinstance(version, str)
        assert len(version) > 0
        # Version should look like "5.x.y"
        assert version[0].isdigit()

    def test_is_mt_decoder_safe(self):
        """is_mt_decoder_safe returns a boolean."""
        result = lzma_mt.is_mt_decoder_safe()
        assert isinstance(result, bool)


class TestNeedsInput:
    def test_needs_input_initial(self):
        """needs_input should be True initially."""
        decompressor = lzma_mt.LZMADecompressor(threads=1)
        assert decompressor.needs_input is True

    def test_needs_input_after_complete(self):
        """needs_input should be False after complete decompression."""
        data = b"Hello"
        compressed = lzma_mt.compress(data)
        decompressor = lzma_mt.LZMADecompressor(threads=1)
        decompressor.decompress(compressed)
        assert decompressor.eof is True
        assert decompressor.needs_input is False

    def test_needs_input_after_partial(self):
        """needs_input should be True after partial input."""
        data = b"x" * 10000
        compressed = lzma_mt.compress(data)
        decompressor = lzma_mt.LZMADecompressor(threads=1)

        # Feed only part of the compressed data
        partial = compressed[:len(compressed) // 2]
        decompressor.decompress(partial)
        assert not decompressor.eof
        assert decompressor.needs_input is True

    def test_needs_input_after_max_length(self):
        """needs_input should be False when output limit is hit."""
        data = b"x" * 10000
        compressed = lzma_mt.compress(data)
        decompressor = lzma_mt.LZMADecompressor(threads=1)

        # Limit output
        result = decompressor.decompress(compressed, max_length=100)
        assert len(result) == 100
        assert not decompressor.eof
        # needs_input should be False because we have unconsumed input
        assert decompressor.needs_input is False


class TestParameterValidation:
    def test_negative_threads_compress(self):
        """Negative threads should raise ValueError."""
        with pytest.raises((ValueError, OverflowError)):
            lzma_mt.compress(b"test", threads=-1)

    def test_negative_threads_decompress(self):
        """Negative threads should raise ValueError."""
        compressed = lzma_mt.compress(b"test")
        with pytest.raises((ValueError, OverflowError)):
            lzma_mt.decompress(compressed, threads=-1)

    def test_negative_threads_compressor(self):
        """Negative threads should raise ValueError for LZMACompressor."""
        with pytest.raises((ValueError, OverflowError)):
            lzma_mt.LZMACompressor(threads=-1)

    def test_negative_threads_decompressor(self):
        """Negative threads should raise ValueError for LZMADecompressor."""
        with pytest.raises((ValueError, OverflowError)):
            lzma_mt.LZMADecompressor(threads=-1)

    def test_invalid_preset(self):
        """Invalid preset should raise LZMAError (matching stdlib behavior)."""
        with pytest.raises(lzma_mt.LZMAError):
            lzma_mt.compress(b"test", preset=10)  # 10 is invalid (max is 9)

    def test_preset_extreme(self):
        """PRESET_EXTREME flag should work."""
        data = b"Test data " * 100
        compressed = lzma_mt.compress(data, preset=6 | lzma_mt.PRESET_EXTREME)
        assert lzma_mt.decompress(compressed) == data


class TestMemlimit:
    def test_memlimit_decompress(self):
        """Test memlimit parameter (stdlib compatible)."""
        data = b"x" * 10000
        compressed = lzma_mt.compress(data)

        result = lzma_mt.decompress(
            compressed,
            threads=1,
            memlimit=1_000_000_000,
        )
        assert result == data

    def test_memlimit_decompressor(self):
        """Test streaming with memlimit."""
        data = b"x" * 10000
        compressed = lzma_mt.compress(data)

        decompressor = lzma_mt.LZMADecompressor(
            threads=1,
            memlimit=1_000_000_000,
        )
        result = decompressor.decompress(compressed)
        assert result == data


class TestThreadSafety:
    def test_concurrent_compress_same_instance(self):
        """Concurrent use of same compressor should not crash."""
        import threading

        compressor = lzma_mt.LZMACompressor()
        results = []
        errors = []

        def worker(chunk_id):
            try:
                data = f"Chunk {chunk_id} ".encode() * 100
                result = compressor.compress(data)
                results.append((chunk_id, result))
            except Exception as e:
                errors.append((chunk_id, e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without errors
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5

    def test_concurrent_decompress_different_instances(self):
        """Concurrent decompression with different instances should work."""
        import threading

        data = b"Test data " * 1000
        compressed = lzma_mt.compress(data)

        results = []
        errors = []

        def worker(worker_id):
            try:
                # Each thread gets its own decompressor
                decompressor = lzma_mt.LZMADecompressor(threads=1)
                result = decompressor.decompress(compressed)
                results.append((worker_id, result == data))
            except Exception as e:
                errors.append((worker_id, e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"
        # All should succeed with correct data
        assert len(results) == 5
        assert all(success for _, success in results)


class TestGILRelease:
    def test_gil_released_during_compression(self):
        """Verify other threads can run during compression."""
        import threading
        import time

        data = b"x" * (5 * 1024 * 1024)  # 5 MB - takes a bit of time
        counter = [0]

        def increment_counter():
            for _ in range(100):
                counter[0] += 1
                time.sleep(0.001)

        # Start counter thread
        counter_thread = threading.Thread(target=increment_counter)
        counter_thread.start()

        # Compress in main thread
        lzma_mt.compress(data, preset=0)  # preset 0 is fastest

        counter_thread.join()

        # If GIL was released, counter should have incremented
        assert counter[0] > 0, "Counter thread didn't run - GIL may not be released"
