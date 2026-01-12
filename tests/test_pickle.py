"""Tests for pickle behavior - adapted from CPython's test_lzma.py

Compressors and decompressors should not be picklable since they contain
internal state that cannot be safely serialized.
"""

import pickle
import pytest

import lzma_mt


class TestPickle:
    """Test that compressor/decompressor objects cannot be pickled.

    Adapted from CPython Lib/test/test_lzma.py::CompressorDecompressorTestCase::test_pickle
    """

    def test_compressor_not_picklable(self):
        """LZMACompressor should raise TypeError when pickled."""
        compressor = lzma_mt.LZMACompressor()
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            with pytest.raises(TypeError):
                pickle.dumps(compressor, proto)

    def test_decompressor_not_picklable(self):
        """LZMADecompressor should raise TypeError when pickled."""
        decompressor = lzma_mt.LZMADecompressor()
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            with pytest.raises(TypeError):
                pickle.dumps(decompressor, proto)

    def test_compressor_in_use_not_picklable(self):
        """LZMACompressor with pending data should not be picklable."""
        compressor = lzma_mt.LZMACompressor()
        compressor.compress(b"some data")
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            with pytest.raises(TypeError):
                pickle.dumps(compressor, proto)

    def test_decompressor_in_use_not_picklable(self):
        """LZMADecompressor with pending data should not be picklable."""
        data = lzma_mt.compress(b"test data for decompression")
        decompressor = lzma_mt.LZMADecompressor()
        # Feed partial data
        decompressor.decompress(data[:10], max_length=5)
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            with pytest.raises(TypeError):
                pickle.dumps(decompressor, proto)
