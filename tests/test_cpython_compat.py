"""Tests adapted from CPython's Lib/test/test_lzma.py

These tests verify compatibility with CPython's lzma module behavior.
lzma_mt only supports XZ format with multi-threading, so FORMAT_ALONE
and FORMAT_RAW tests are skipped.

Adapted from:
- CPython main branch Lib/test/test_lzma.py
- PR #114954 (multithreaded XZ compression)
"""

import lzma
import pytest

import lzma_mt


# =============================================================================
# Test Data from CPython (Hamlet excerpt from Shakespeare)
# =============================================================================

INPUT = b"""
LAERTES

       O, fear me not.
       I stay too long: but here my father comes.

       Enter POLONIUS

       A double blessing is a double grace,
       Occasion smiles upon a second leave.

LORD POLONIUS

       Yet here, Laertes! aboard, aboard, for shame!
       The wind sits in the shoulder of your sail,
       And you are stay'd for. There; my blessing with thee!
       And these few precepts in thy memory
       See thou character. Give thy thoughts no tongue,
       Nor any unproportioned thought his act.
       Be thou familiar, but by no means vulgar.
       Those friends thou hast, and their adoption tried,
       Grapple them to thy soul with hoops of steel;
       But do not dull thy palm with entertainment
       Of each new-hatch'd, unfledged comrade. Beware
       Of entrance to a quarrel, but being in,
       Bear't that the opposed may beware of thee.
       Give every man thy ear, but few thy voice;
       Take each man's censure, but reserve thy judgment.
       Costly thy habit as thy purse can buy,
       But not express'd in fancy; rich, not gaudy;
       For the apparel oft proclaims the man,
       And they in France of the best rank and station
       Are of a most select and generous chief in that.
       Neither a borrower nor a lender be;
       For loan oft loses both itself and friend,
       And borrowing dulls the edge of husbandry.
       This above all: to thine ownself be true,
       And it must follow, as the night the day,
       Thou canst not then be false to any man.
       Farewell: my blessing season this in thee!

LAERTES

       Most humbly do I take my leave, my lord.

LORD POLONIUS

       The time invites you; go; your servants tend.

LAERTES

       Farewell, Ophelia; and remember well
       What I have said to you.

OPHELIA

       'Tis in my memory lock'd,
       And you yourself shall keep the key of it.

LAERTES

       Farewell.
"""

# XZ-compressed version of INPUT (from CPython test suite)
COMPRESSED_XZ = (
    b"\xfd7zXZ\x00\x00\x04\xe6\xd6\xb4F\x02\x00!\x01\x16\x00\x00\x00t/\xe5\xa3"
    b"\xe0\x07\x80\x03\xdf]\x00\x05\x14\x07bX\x19\xcd\xddn\x98\x15\xe4\xb4\x9d"
    b"o\x1d\xc4\xe5\n\x03\xcc2h\xc7\\\x86\xff\xf8\xe2\xfc\xe7\xd9\xfe6\xb8("
    b"\xa8wd\xc2\"u.n\x1e\xc3\xf2\x8e\x8d\x8f\x02\x17/\xa6=\xf0\xa2\xdf/M\x89"
    b"\xbe\xde\xa7\x1cz\x18-]\xd5\xef\x13\x8frZ\x15\x80\x8c\xf8\x8do\xfa\x12"
    b"\x9b#z/\xef\xf0\xfaF\x01\x82\xa3M\x8e\xa1t\xca6 BF$\xe5Q\xa4\x98\xee\xde"
    b"l\xe8\x7f\xf0\x9d,bn\x0b\x13\xd4\xa8\x81\xe4N\xc8\x86\x153\xf5x2\xa2O"
    b"\x13@Q\xa1\x00/\xa5\xd0O\x97\xdco\xae\xf7z\xc4\xcdS\xb6t<\x16\xf2\x9cI#"
    b"\x89ud\xc66Y\xd9\xee\xe6\xce\x12]\xe5\xf0\xaa\x96-Pe\xade:\x04\t\x1b\xf7"
    b"\xdb7\n\x86\x1fp\xc8J\xba\xf4\xf0V\xa9\xdc\xf0\x02%G\xf9\xdf=?\x15\x1b"
    b"\xe1(\xce\x82=\xd6I\xac3\x12\x0cR\xb7\xae\r\xb1i\x03\x95\x01\xbd\xbe\xfa"
    b"\x02s\x01P\x9d\x96X\xb12j\xc8L\xa8\x84b\xf6\xc3\xd4c-H\x93oJl\xd0iQ\xe4k"
    b"\x84\x0b\xc1\xb7\xbc\xb1\x17\x88\xb1\xca?@\xf6\x07\xea\xe6x\xf1H12P\x0f"
    b"\x8a\xc9\xeauw\xe3\xbe\xaai\xa9W\xd0\x80\xcd#cb5\x99\xd8]\xa9d\x0c\xbd"
    b"\xa2\xdcWl\xedUG\xbf\x89yF\xf77\x81v\xbd5\x98\xbeh8\x18W\x08\xf0\x1b\x99"
    b"5:\x1a?rD\x96\xa1\x04\x0f\xae\xba\x85\xeb\x9d5@\xf5\x83\xd37\x83\x8ac"
    b"\x06\xd4\x97i\xcdt\x16S\x82k\xf6K\x01vy\x88\x91\x9b6T\xdae\r\xfd]:k\xbal"
    b"\xa9\xbba\xc34\xf9r\xeb}r\xdb\xc7\xdb*\x8f\x03z\xdc8h\xcc\xc9\xd3\xbcl"
    b"\xa5-\xcb\xeaK\xa2\xc5\x15\xc0\xe3\xc1\x86Z\xfb\xebL\xe13\xcf\x9c\xe3"
    b"\x1d\xc9\xed\xc2\x06\xcc\xce!\x92\xe5\xfe\x9c^\xa59w \x9bP\xa3PK\x08d"
    b"\xf9\xe2Z}\xa7\xbf\xed\xeb%$\x0c\x82\xb8/\xb0\x01\xa9&,\xf7qh{Q\x96)\xf2"
    b"q\x96\xc3\x80\xb4\x12\xb0\xba\xe6o\xf4!\xb4[\xd4\x8aw\x10\xf7t\x0c\xb3"
    b"\xd9\xd5\xc3`^\x81\x11??\\\xa4\x99\x85R\xd4\x8e\x83\xc9\x1eX\xbfa\xf1"
    b"\xac\xb0\xea\xea\xd7\xd0\xab\x18\xe2\xf2\xed\xe1\xb7\xc9\x18\xcbS\xe4>"
    b"\xc9\x95H\xe8\xcb\t\r%\xeb\xc7$.o\xf1\xf3R\x17\x1db\xbb\xd8U\xa5^\xccS"
    b"\x16\x01\x87\xf3/\x93\xd1\xf0v\xc0r\xd7\xcc\xa2Gkz\xca\x80\x0e\xfd\xd0"
    b"\x8b\xbb\xd2Ix\xb3\x1ey\xca-0\xe3z^\xd6\xd6\x8f_\xf1\x9dP\x9fi\xa7\xd1"
    b"\xe8\x90\x84\xdc\xbf\xcdky\x8e\xdc\x81\x7f\xa3\xb2+\xbf\x04\xef\xd8\\"
    b"\xc4\xdf\xe1\xb0\x01\xe9\x93\xe3Y\xf1\x1dY\xe8h\x81\xcf\xf1w\xcc\xb4\xef"
    b" \x8b|\x04\xea\x83ej\xbe\x1f\xd4z\x9c`\xd3\x1a\x92A\x06\xe5\x8f\xa9\x13"
    b"\t\x9e=\xfa\x1c\xe5_\x9f%v\x1bo\x11ZO\xd8\xf4\t\xddM\x16-\x04\xfc\x18<\""
    b"CM\xddg~b\xf6\xef\x8e\x0c\xd0\xde|\xa0'\x8a\x0c\xd6x\xae!J\xa6F\x88\x15u"
    b"\x008\x17\xbc7y\xb3\xd8u\xac_\x85\x8d\xe7\xc1@\x9c\xecqc\xa3#\xad\xf1"
    b"\x935\xb5)_\r\xec3]\x0fo]5\xd0my\x07\x9b\xee\x81\xb5\x0f\xcfK+\x00\xc0"
    b"\xe4b\x10\xe4\x0c\x1a \x9b\xe0\x97t\xf6\xa1\x9e\x850\xba\x0c\x9a\x8d\xc8"
    b"\x8f\x07\xd7\xae\xc8\xf9+i\xdc\xb9k\xb0>f\x19\xb8\r\xa8\xf8\x1f$\xa5{p"
    b"\xc6\x880\xce\xdb\xcf\xca_\x86\xac\x88h6\x8bZ%'\xd0\n\xbf\x0f\x9c\"\xba"
    b"\xe5\x86\x9f\x0f7X=mNX[\xcc\x19FU\xc9\x860\xbc\x90a+* \xae_$\x03\x1e\xd3"
    b"\xcd_\xa0\x9c\xde\xaf46q\xa5\xc9\x92\xd7\xca\xe3`\x9d\x85}\xb4\xff\xb3"
    b"\x83\xfb\xb6\xca\xae`\x0bw\x7f\xfc\xd8\xacVe\x19\xc8\x17\x0bZ\xad\x88"
    b"\xeb#\x97\x03\x13\xb1d\x0f{\x0c\x04w\x07\r\x97\xbd\xd6\xc1\xc3B:\x95\x08"
    b"^\x10V\xaeaH\x02\xd9\xe3\n\\\x01X\xf6\x9c\x8a\x06u#%\xbe*\xa1\x18v\x85"
    b"\xec!\t4\x00\x00\x00\x00Vj?uLU\xf3\xa6\x00\x01\xfb\x07\x81\x0f\x00\x00tw"
    b"\x99P\xb1\xc4g\xfb\x02\x00\x00\x00\x00\x04YZ"
)

COMPRESSED_BOGUS = b"this is not a valid lzma stream"


# =============================================================================
# Helper to check decompressor state
# =============================================================================

def _check_decompressor(lzd, data, unused_data=b""):
    """Helper to verify decompressor behavior."""
    assert not lzd.eof
    out = lzd.decompress(data)
    assert out == INPUT
    assert lzd.eof
    assert lzd.unused_data == unused_data


# =============================================================================
# CompressorDecompressorTestCase - Streaming tests
# =============================================================================

class TestStreamingCompression:
    """Tests for LZMACompressor/LZMADecompressor streaming.

    Adapted from CPython CompressorDecompressorTestCase.
    """

    def test_decompressor_eof_property(self):
        """Test that eof property works correctly."""
        lzd = lzma_mt.LZMADecompressor()
        assert not lzd.eof
        lzd.decompress(COMPRESSED_XZ)
        assert lzd.eof

    def test_decompressor_after_eof(self):
        """After EOF, decompress should raise."""
        lzd = lzma_mt.LZMADecompressor()
        lzd.decompress(COMPRESSED_XZ)
        with pytest.raises(ValueError):
            lzd.decompress(b"nyan")

    def test_decompressor_memlimit(self):
        """Test memory limit enforcement."""
        lzd = lzma_mt.LZMADecompressor(memlimit=1024)
        with pytest.raises(MemoryError):
            lzd.decompress(COMPRESSED_XZ)

    def test_decompressor_xz(self):
        """Test decompressing XZ format."""
        lzd = lzma_mt.LZMADecompressor()
        _check_decompressor(lzd, COMPRESSED_XZ)

    def test_decompressor_chunks(self):
        """Test decompressing in chunks."""
        lzd = lzma_mt.LZMADecompressor()
        out = []
        for i in range(0, len(COMPRESSED_XZ), 10):
            assert not lzd.eof
            out.append(lzd.decompress(COMPRESSED_XZ[i:i+10]))
        out = b"".join(out)
        assert out == INPUT
        assert lzd.eof
        assert lzd.unused_data == b""

    def test_decompressor_chunks_empty(self):
        """Test that empty chunks are handled correctly."""
        lzd = lzma_mt.LZMADecompressor()
        out = []
        for i in range(0, len(COMPRESSED_XZ), 10):
            assert not lzd.eof
            out.append(lzd.decompress(b''))
            out.append(lzd.decompress(b''))
            out.append(lzd.decompress(b''))
            out.append(lzd.decompress(COMPRESSED_XZ[i:i+10]))
        out = b"".join(out)
        assert out == INPUT
        assert lzd.eof
        assert lzd.unused_data == b""

    def test_decompressor_chunks_maxsize(self):
        """Test chunked decompression with max_length."""
        lzd = lzma_mt.LZMADecompressor()
        max_length = 100
        out = []

        # Feed first half the input
        len_ = len(COMPRESSED_XZ) // 2
        out.append(lzd.decompress(COMPRESSED_XZ[:len_], max_length=max_length))
        assert not lzd.needs_input
        assert len(out[-1]) == max_length

        # Retrieve more data without providing more input
        out.append(lzd.decompress(b'', max_length=max_length))
        assert not lzd.needs_input
        assert len(out[-1]) == max_length

        # Retrieve more data while providing more input
        out.append(lzd.decompress(COMPRESSED_XZ[len_:], max_length=max_length))
        assert len(out[-1]) <= max_length

        # Retrieve remaining uncompressed data
        while not lzd.eof:
            out.append(lzd.decompress(b'', max_length=max_length))
            assert len(out[-1]) <= max_length

        out = b"".join(out)
        assert out == INPUT
        assert lzd.unused_data == b""

    def test_decompressor_unused_data(self):
        """Test unused_data property with trailing bytes."""
        lzd = lzma_mt.LZMADecompressor()
        extra = b"fooblibar"
        _check_decompressor(lzd, COMPRESSED_XZ + extra, unused_data=extra)

    def test_decompressor_bad_input(self):
        """Test that bad input raises LZMAError (matching stdlib)."""
        lzd = lzma_mt.LZMADecompressor()
        with pytest.raises(lzma_mt.LZMAError):
            lzd.decompress(COMPRESSED_BOGUS)

    def test_decompressor_bug_28275(self):
        """Test that calling decompress again after error doesn't crash.

        Adapted from CPython test_decompressor_bug_28275 (Issue 28275).
        """
        lzd = lzma_mt.LZMADecompressor()
        with pytest.raises(lzma_mt.LZMAError):
            lzd.decompress(COMPRESSED_BOGUS)
        # Previously, a second call could crash due to internal inconsistency
        with pytest.raises(ValueError):
            lzd.decompress(COMPRESSED_BOGUS)

    def test_decompressor_multistream(self):
        """Test that decompressor handles multistream with unused_data.

        Adapted from CPython test_decompressor_multistream.
        LZMADecompressor does NOT handle concatenated streams automatically;
        unused_data contains the second stream.
        """
        cdata1 = lzma_mt.compress(b"first")
        cdata2 = lzma_mt.compress(b"second")
        lzd = lzma_mt.LZMADecompressor()
        result = lzd.decompress(cdata1 + cdata2)
        assert result == b"first"
        assert lzd.eof
        assert lzd.unused_data == cdata2

    def test_roundtrip_xz(self):
        """Test compress->decompress roundtrip."""
        lzc = lzma_mt.LZMACompressor()
        cdata = lzc.compress(INPUT) + lzc.flush()
        lzd = lzma_mt.LZMADecompressor()
        _check_decompressor(lzd, cdata)

    def test_roundtrip_xz_mt(self):
        """Test multithreaded compress->decompress roundtrip.

        Adapted from CPython PR #114954 test_roundtrip_xz_mt.
        """
        lzc = lzma_mt.LZMACompressor(threads=0)  # auto-detect threads
        cdata = lzc.compress(INPUT) + lzc.flush()
        lzd = lzma_mt.LZMADecompressor()
        _check_decompressor(lzd, cdata)

    def test_roundtrip_xz_mt_preset_6(self):
        """Test multithreaded compression with preset 6.

        Adapted from CPython PR #114954 test_roundtrip_xz_mt_preset_6.
        """
        lzc = lzma_mt.LZMACompressor(preset=6, threads=8)
        cdata = lzc.compress(INPUT) + lzc.flush()
        lzd = lzma_mt.LZMADecompressor()
        _check_decompressor(lzd, cdata)

    def test_roundtrip_chunks(self):
        """Test chunked compression roundtrip."""
        lzc = lzma_mt.LZMACompressor()
        cdata = []
        for i in range(0, len(INPUT), 10):
            cdata.append(lzc.compress(INPUT[i:i+10]))
        cdata.append(lzc.flush())
        cdata = b"".join(cdata)
        lzd = lzma_mt.LZMADecompressor()
        _check_decompressor(lzd, cdata)

    def test_roundtrip_empty_chunks(self):
        """Test that empty chunks in compression work."""
        lzc = lzma_mt.LZMACompressor()
        cdata = []
        for i in range(0, len(INPUT), 10):
            cdata.append(lzc.compress(INPUT[i:i+10]))
            cdata.append(lzc.compress(b''))
            cdata.append(lzc.compress(b''))
            cdata.append(lzc.compress(b''))
        cdata.append(lzc.flush())
        cdata = b"".join(cdata)
        lzd = lzma_mt.LZMADecompressor()
        _check_decompressor(lzd, cdata)

    def test_compressor_flush_twice_raises(self):
        """Test that flushing twice raises an error."""
        lzc = lzma_mt.LZMACompressor()
        lzc.flush()
        with pytest.raises(ValueError):
            lzc.flush()

    def test_compressor_compress_after_flush_raises(self):
        """Test that compressing after flush raises an error."""
        lzc = lzma_mt.LZMACompressor()
        lzc.flush()
        with pytest.raises(ValueError):
            lzc.compress(b"data")


# =============================================================================
# CompressDecompressFunctionTestCase - One-shot functions
# =============================================================================

class TestFunctionCompress:
    """Tests for lzma_mt.compress() and decompress() functions.

    Adapted from CPython CompressDecompressFunctionTestCase.
    """

    def test_decompress_memlimit(self):
        """Test memory limit in decompress function."""
        with pytest.raises(MemoryError):
            lzma_mt.decompress(COMPRESSED_XZ, memlimit=1024)

    def test_decompress_good_input(self):
        """Test decompressing valid data."""
        ddata = lzma_mt.decompress(COMPRESSED_XZ)
        assert ddata == INPUT

    def test_decompress_incomplete_input(self):
        """Test that incomplete input raises LZMAError (matching stdlib)."""
        with pytest.raises(lzma_mt.LZMAError):
            lzma_mt.decompress(COMPRESSED_XZ[:128])

    def test_decompress_bad_input(self):
        """Test that bad input raises LZMAError (matching stdlib)."""
        with pytest.raises(lzma_mt.LZMAError):
            lzma_mt.decompress(COMPRESSED_BOGUS)

    def test_roundtrip(self):
        """Test compress/decompress roundtrip."""
        cdata = lzma_mt.compress(INPUT)
        ddata = lzma_mt.decompress(cdata)
        assert ddata == INPUT

    def test_roundtrip_threads(self):
        """Test roundtrip with explicit thread count."""
        for threads in [0, 1, 2, 4]:
            cdata = lzma_mt.compress(INPUT, threads=threads)
            ddata = lzma_mt.decompress(cdata, threads=threads)
            assert ddata == INPUT

    def test_roundtrip_all_presets(self):
        """Test all preset levels."""
        for preset in range(10):
            cdata = lzma_mt.compress(INPUT, preset=preset)
            ddata = lzma_mt.decompress(cdata)
            assert ddata == INPUT

    def test_roundtrip_preset_extreme(self):
        """Test PRESET_EXTREME flag."""
        cdata = lzma_mt.compress(INPUT, preset=6 | lzma_mt.PRESET_EXTREME)
        ddata = lzma_mt.decompress(cdata)
        assert ddata == INPUT


# =============================================================================
# Parameter Validation Tests
# =============================================================================

class TestParameterValidation:
    """Test parameter validation.

    Adapted from CPython test_simple_bad_args and PR #114954.
    """

    def test_invalid_preset(self):
        """Test that invalid preset raises LZMAError (matching stdlib)."""
        with pytest.raises(lzma_mt.LZMAError):
            lzma_mt.LZMACompressor(preset=10)
        with pytest.raises(lzma_mt.LZMAError):
            lzma_mt.compress(b"data", preset=10)

    def test_negative_threads_compressor(self):
        """Test that negative threads raises ValueError.

        Adapted from CPython PR #114954 test_init_bad_threads.
        """
        with pytest.raises(ValueError):
            lzma_mt.LZMACompressor(threads=-1)

    def test_negative_threads_decompressor(self):
        """Test that negative threads raises ValueError."""
        with pytest.raises(ValueError):
            lzma_mt.LZMADecompressor(threads=-1)

    def test_negative_threads_compress(self):
        """Test that negative threads raises ValueError in compress."""
        with pytest.raises(ValueError):
            lzma_mt.compress(b"data", threads=-1)

    def test_negative_threads_decompress(self):
        """Test that negative threads raises ValueError in decompress."""
        with pytest.raises(ValueError):
            lzma_mt.decompress(COMPRESSED_XZ, threads=-1)


# =============================================================================
# Interoperability with stdlib lzma
# =============================================================================

class TestStdlibInterop:
    """Test interoperability with Python's standard library lzma module."""

    def test_decompress_stdlib_xz(self):
        """Test decompressing data from stdlib lzma."""
        # Compress with stdlib
        cdata = lzma.compress(INPUT)
        # Decompress with lzma_mt
        ddata = lzma_mt.decompress(cdata)
        assert ddata == INPUT

    def test_compress_for_stdlib(self):
        """Test that lzma_mt output can be decompressed by stdlib."""
        # Compress with lzma_mt
        cdata = lzma_mt.compress(INPUT)
        # Decompress with stdlib
        ddata = lzma.decompress(cdata)
        assert ddata == INPUT

    def test_streaming_interop(self):
        """Test streaming interop with stdlib."""
        # Compress with lzma_mt streaming
        lzc = lzma_mt.LZMACompressor()
        cdata = lzc.compress(INPUT) + lzc.flush()

        # Decompress with stdlib streaming
        lzd = lzma.LZMADecompressor()
        ddata = lzd.decompress(cdata)
        assert ddata == INPUT

    def test_stdlib_streaming_to_lzma_mt(self):
        """Test stdlib compressed stream decompressed by lzma_mt."""
        # Compress with stdlib streaming
        lzc = lzma.LZMACompressor()
        cdata = lzc.compress(INPUT) + lzc.flush()

        # Decompress with lzma_mt streaming
        lzd = lzma_mt.LZMADecompressor()
        ddata = lzd.decompress(cdata)
        assert ddata == INPUT


# =============================================================================
# Check Types Tests
# =============================================================================

class TestCheckTypes:
    """Test different integrity check types."""

    def test_check_crc32(self):
        """Test CHECK_CRC32."""
        cdata = lzma_mt.compress(INPUT, check=lzma_mt.CHECK_CRC32)
        ddata = lzma_mt.decompress(cdata)
        assert ddata == INPUT

    def test_check_crc64(self):
        """Test CHECK_CRC64 (default)."""
        cdata = lzma_mt.compress(INPUT, check=lzma_mt.CHECK_CRC64)
        ddata = lzma_mt.decompress(cdata)
        assert ddata == INPUT

    def test_check_sha256(self):
        """Test CHECK_SHA256."""
        cdata = lzma_mt.compress(INPUT, check=lzma_mt.CHECK_SHA256)
        ddata = lzma_mt.decompress(cdata)
        assert ddata == INPUT

    def test_check_none(self):
        """Test CHECK_NONE."""
        cdata = lzma_mt.compress(INPUT, check=lzma_mt.CHECK_NONE)
        ddata = lzma_mt.decompress(cdata)
        assert ddata == INPUT


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_input(self):
        """Test compressing empty data."""
        cdata = lzma_mt.compress(b"")
        ddata = lzma_mt.decompress(cdata)
        assert ddata == b""

    def test_single_byte(self):
        """Test compressing single byte."""
        cdata = lzma_mt.compress(b"x")
        ddata = lzma_mt.decompress(cdata)
        assert ddata == b"x"

    def test_all_byte_values(self):
        """Test data containing all byte values."""
        data = bytes(range(256))
        cdata = lzma_mt.compress(data)
        ddata = lzma_mt.decompress(cdata)
        assert ddata == data

    def test_repeated_byte(self):
        """Test highly compressible data."""
        data = b"x" * 100000
        cdata = lzma_mt.compress(data)
        ddata = lzma_mt.decompress(cdata)
        assert ddata == data
        # Verify compression actually happened
        assert len(cdata) < len(data)

    def test_incompressible_data(self):
        """Test incompressible (random) data."""
        import random
        random.seed(42)
        data = bytes(random.getrandbits(8) for _ in range(10000))
        cdata = lzma_mt.compress(data)
        ddata = lzma_mt.decompress(cdata)
        assert ddata == data

    def test_bytearray_input(self):
        """Test bytearray input."""
        data = bytearray(INPUT)
        cdata = lzma_mt.compress(data)
        ddata = lzma_mt.decompress(bytearray(cdata))
        assert ddata == INPUT

    def test_memoryview_input(self):
        """Test memoryview input."""
        data = memoryview(INPUT)
        cdata = lzma_mt.compress(data)
        ddata = lzma_mt.decompress(memoryview(cdata))
        assert ddata == INPUT


# =============================================================================
# Concatenated Streams
# =============================================================================

class TestConcatenatedStreams:
    """Test handling of concatenated compressed streams."""

    def test_decompress_concatenated(self):
        """Test decompressing concatenated XZ streams."""
        cdata1 = lzma_mt.compress(b"first")
        cdata2 = lzma_mt.compress(b"second")
        # The one-shot decompress should handle concatenated streams
        ddata = lzma_mt.decompress(cdata1 + cdata2)
        assert ddata == b"firstsecond"

    def test_decompress_trailing_junk(self):
        """Test that trailing junk after valid stream is handled."""
        cdata = lzma_mt.compress(INPUT)
        # Should decompress successfully, ignoring trailing junk
        ddata = lzma_mt.decompress(cdata + COMPRESSED_BOGUS)
        assert ddata == INPUT

    def test_decompress_multistream_trailing_junk(self):
        """Test that trailing junk after multiple concatenated streams is handled.

        Adapted from CPython test_decompress_multistream in CompressDecompressFunctionTestCase.
        """
        cdata1 = lzma_mt.compress(b"stream1")
        cdata2 = lzma_mt.compress(b"stream2")
        cdata3 = lzma_mt.compress(b"stream3")
        # Multiple valid streams followed by junk
        combined = cdata1 + cdata2 + cdata3 + COMPRESSED_BOGUS
        ddata = lzma_mt.decompress(combined)
        assert ddata == b"stream1stream2stream3"
