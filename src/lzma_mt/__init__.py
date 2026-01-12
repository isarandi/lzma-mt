"""
lzma_mt - Multi-threaded LZMA/XZ compression

A drop-in replacement for Python's lzma module with multi-threading support.

Security note: The multi-threaded decoder in xz-utils 5.3.3alpha-5.8.0 has
CVE-2025-31115 (use-after-free). This module checks the version at runtime
and raises RuntimeError if vulnerable. Use is_mt_decoder_safe() to check,
or pass threads=1 to use single-threaded mode.
"""

import builtins
import io
import os

# _compression is a private CPython module, may not exist in all versions
try:
    from _compression import BaseStream, DecompressReader
except ImportError:
    # Fallback implementations (copied from CPython's _compression.py)
    BUFFER_SIZE = io.DEFAULT_BUFFER_SIZE

    class BaseStream(io.BufferedIOBase):
        """Mode-checking helper functions."""
        def _check_not_closed(self):
            if self.closed:
                raise ValueError("I/O operation on closed file")

        def _check_can_read(self):
            if not self.readable():
                raise io.UnsupportedOperation("File not open for reading")

        def _check_can_write(self):
            if not self.writable():
                raise io.UnsupportedOperation("File not open for writing")

        def _check_can_seek(self):
            if not self.readable():
                raise io.UnsupportedOperation(
                    "Seeking is only supported on files open for reading")
            if not self.seekable():
                raise io.UnsupportedOperation(
                    "The underlying file object does not support seeking")

    class DecompressReader(io.RawIOBase):
        """Adapts the decompressor API to a RawIOBase reader API"""
        def readable(self):
            return True

        def __init__(self, fp, decomp_factory, trailing_error=(), **decomp_args):
            self._fp = fp
            self._eof = False
            self._pos = 0
            self._size = -1
            self._decomp_factory = decomp_factory
            self._decomp_args = decomp_args
            self._decompressor = self._decomp_factory(**self._decomp_args)
            self._trailing_error = trailing_error

        def close(self):
            self._decompressor = None
            return super().close()

        def seekable(self):
            return self._fp.seekable()

        def readinto(self, b):
            with memoryview(b) as view, view.cast("B") as byte_view:
                data = self.read(len(byte_view))
                byte_view[:len(data)] = data
            return len(data)

        def read(self, size=-1):
            if size < 0:
                return self.readall()
            if not size or self._eof:
                return b""
            data = None
            while True:
                if self._decompressor.eof:
                    rawblock = (self._decompressor.unused_data or
                                self._fp.read(BUFFER_SIZE))
                    if not rawblock:
                        break
                    self._decompressor = self._decomp_factory(**self._decomp_args)
                    try:
                        data = self._decompressor.decompress(rawblock, size)
                    except self._trailing_error:
                        break
                else:
                    if self._decompressor.needs_input:
                        rawblock = self._fp.read(BUFFER_SIZE)
                        if not rawblock:
                            raise EOFError(
                                "Compressed file ended before the "
                                "end-of-stream marker was reached")
                    else:
                        rawblock = b""
                    data = self._decompressor.decompress(rawblock, size)
                if data:
                    break
            if not data:
                self._eof = True
                self._size = self._pos
                return b""
            self._pos += len(data)
            return data

# Constants - imported directly from stdlib for guaranteed compatibility
from lzma import (
    CHECK_NONE, CHECK_CRC32, CHECK_CRC64, CHECK_SHA256, CHECK_ID_MAX, CHECK_UNKNOWN,
    PRESET_DEFAULT, PRESET_EXTREME,
    FORMAT_AUTO, FORMAT_XZ, FORMAT_ALONE, FORMAT_RAW,
    FILTER_LZMA1, FILTER_LZMA2, FILTER_DELTA,
    FILTER_X86, FILTER_POWERPC, FILTER_IA64, FILTER_ARM, FILTER_ARMTHUMB, FILTER_SPARC,
    MF_HC3, MF_HC4, MF_BT2, MF_BT3, MF_BT4,
    MODE_FAST, MODE_NORMAL,
    is_check_supported,
)

from lzma_mt.lzma_mt import (
    compress,
    decompress,
    LZMACompressor,
    LZMADecompressor,
    LZMAError,
    get_xz_version,
    is_mt_decoder_safe,
)

try:
    from lzma_mt._version import __version__
except ImportError:
    __version__ = "0.0.0.dev0"


# =============================================================================
# LZMAFile with multi-threading support
# =============================================================================

_MODE_CLOSED = 0
_MODE_READ = 1
_MODE_WRITE = 3


class LZMAFile(BaseStream):
    """A file object providing transparent LZMA (de)compression with MT support.

    An LZMAFile can act as a wrapper for an existing file object, or
    refer directly to a named file on disk.

    Note that LZMAFile provides a *binary* file interface - data read
    is returned as bytes, and data to be written must be given as bytes.

    This is identical to stdlib lzma.LZMAFile but with an additional
    'threads' parameter for multi-threaded compression/decompression.
    """

    def __init__(self, filename=None, mode="r", *,
                 format=None, check=-1, preset=None, filters=None, threads=1):
        """Open an LZMA-compressed file in binary mode.

        filename can be either an actual file name (given as a str,
        bytes, or PathLike object), in which case the named file is
        opened, or it can be an existing file object to read from or
        write to.

        mode can be "r" for reading (default), "w" for (over)writing,
        "x" for creating exclusively, or "a" for appending. These can
        equivalently be given as "rb", "wb", "xb" and "ab" respectively.

        format specifies the container format to use for the file.
        If mode is "r", this defaults to FORMAT_AUTO. Otherwise, the
        default is FORMAT_XZ.

        check specifies the integrity check to use. This argument can
        only be used when opening a file for writing. For FORMAT_XZ,
        the default is CHECK_CRC64. FORMAT_ALONE and FORMAT_RAW do not
        support integrity checks - for these formats, check must be
        omitted, or be CHECK_NONE.

        When opening a file for reading, the *preset* argument is not
        meaningful, and should be omitted. The *filters* argument should
        also be omitted, except when format is FORMAT_RAW (in which case
        it is required).

        When opening a file for writing, the settings used by the
        compressor can be specified either as a preset compression
        level (with the *preset* argument), or in detail as a custom
        filter chain (with the *filters* argument). For FORMAT_XZ and
        FORMAT_ALONE, the default is to use the PRESET_DEFAULT preset
        level. For FORMAT_RAW, the caller must always specify a filter
        chain; the raw compressor does not support preset compression
        levels.

        preset (if provided) should be an integer in the range 0-9,
        optionally OR-ed with the constant PRESET_EXTREME.

        filters (if provided) should be a sequence of dicts. Each dict
        should have an entry for "id" indicating ID of the filter, plus
        additional entries for options to the filter.

        threads specifies the number of threads to use (default 1).
        Use 0 for auto-detect based on CPU count.
        """
        self._fp = None
        self._closefp = False
        self._mode = _MODE_CLOSED
        self._threads = threads

        if mode in ("r", "rb"):
            if check != -1:
                raise ValueError("Cannot specify an integrity check "
                                 "when opening a file for reading")
            if preset is not None:
                raise ValueError("Cannot specify a preset compression "
                                 "level when opening a file for reading")
            if format is None:
                format = FORMAT_AUTO
            mode_code = _MODE_READ
        elif mode in ("w", "wb", "a", "ab", "x", "xb"):
            if format is None:
                format = FORMAT_XZ
            mode_code = _MODE_WRITE
            self._compressor = LZMACompressor(format=format, check=check,
                                              preset=preset, filters=filters,
                                              threads=threads)
            self._pos = 0
        else:
            raise ValueError("Invalid mode: {!r}".format(mode))

        if isinstance(filename, (str, bytes, os.PathLike)):
            if "b" not in mode:
                mode += "b"
            self._fp = builtins.open(filename, mode)
            self._closefp = True
            self._mode = mode_code
        elif hasattr(filename, "read") or hasattr(filename, "write"):
            self._fp = filename
            self._mode = mode_code
        else:
            raise TypeError("filename must be a str, bytes, file or PathLike object")

        if self._mode == _MODE_READ:
            # Create a decompressor factory that passes threads
            def _make_decompressor():
                return LZMADecompressor(format=format, filters=filters,
                                        threads=threads)
            raw = DecompressReader(self._fp, _make_decompressor,
                trailing_error=LZMAError)
            self._buffer = io.BufferedReader(raw)

    def close(self):
        """Flush and close the file.

        May be called more than once without error. Once the file is
        closed, any other operation on it will raise a ValueError.
        """
        if self._mode == _MODE_CLOSED:
            return
        try:
            if self._mode == _MODE_READ:
                self._buffer.close()
                self._buffer = None
            elif self._mode == _MODE_WRITE:
                self._fp.write(self._compressor.flush())
                self._compressor = None
        finally:
            try:
                if self._closefp:
                    self._fp.close()
            finally:
                self._fp = None
                self._closefp = False
                self._mode = _MODE_CLOSED

    @property
    def closed(self):
        """True if this file is closed."""
        return self._mode == _MODE_CLOSED

    def fileno(self):
        """Return the file descriptor for the underlying file."""
        self._check_not_closed()
        return self._fp.fileno()

    def seekable(self):
        """Return whether the file supports seeking."""
        return self.readable() and self._buffer.seekable()

    def readable(self):
        """Return whether the file was opened for reading."""
        self._check_not_closed()
        return self._mode == _MODE_READ

    def writable(self):
        """Return whether the file was opened for writing."""
        self._check_not_closed()
        return self._mode == _MODE_WRITE

    def peek(self, size=-1):
        """Return buffered data without advancing the file position.

        Always returns at least one byte of data, unless at EOF.
        The exact number of bytes returned is unspecified.
        """
        self._check_can_read()
        return self._buffer.peek(size)

    def read(self, size=-1):
        """Read up to size uncompressed bytes from the file.

        If size is negative or omitted, read until EOF is reached.
        Returns b"" if the file is already at EOF.
        """
        self._check_can_read()
        return self._buffer.read(size)

    def read1(self, size=-1):
        """Read up to size uncompressed bytes, while trying to avoid
        making multiple reads from the underlying stream. Reads up to a
        buffer's worth of data if size is negative.

        Returns b"" if the file is at EOF.
        """
        self._check_can_read()
        if size < 0:
            size = io.DEFAULT_BUFFER_SIZE
        return self._buffer.read1(size)

    def readline(self, size=-1):
        """Read a line of uncompressed bytes from the file.

        The terminating newline (if present) is retained. If size is
        non-negative, no more than size bytes will be read (in which
        case the line may be incomplete). Returns b'' if already at EOF.
        """
        self._check_can_read()
        return self._buffer.readline(size)

    def write(self, data):
        """Write a bytes object to the file.

        Returns the number of uncompressed bytes written, which is
        always the length of data in bytes. Note that due to buffering,
        the file on disk may not reflect the data written until close()
        is called.
        """
        self._check_can_write()
        if isinstance(data, (bytes, bytearray)):
            length = len(data)
        else:
            # accept any data that supports the buffer protocol
            data = memoryview(data)
            length = data.nbytes

        compressed = self._compressor.compress(data)
        self._fp.write(compressed)
        self._pos += length
        return length

    def seek(self, offset, whence=io.SEEK_SET):
        """Change the file position.

        The new position is specified by offset, relative to the
        position indicated by whence. Possible values for whence are:

            0: start of stream (default): offset must not be negative
            1: current stream position
            2: end of stream; offset must not be positive

        Returns the new file position.

        Note that seeking is emulated, so depending on the parameters,
        this operation may be extremely slow.
        """
        self._check_can_seek()
        return self._buffer.seek(offset, whence)

    def tell(self):
        """Return the current file position."""
        self._check_not_closed()
        if self._mode == _MODE_READ:
            return self._buffer.tell()
        return self._pos


def open(filename, mode="rb", *,
         format=None, check=-1, preset=None, filters=None,
         encoding=None, errors=None, newline=None, threads=1):
    """Open an LZMA-compressed file in binary or text mode.

    filename can be either an actual file name (given as a str, bytes,
    or PathLike object), in which case the named file is opened, or it
    can be an existing file object to read from or write to.

    The mode argument can be "r", "rb" (default), "w", "wb", "x", "xb",
    "a", or "ab" for binary mode, or "rt", "wt", "xt", or "at" for text
    mode.

    The format, check, preset and filters arguments specify the
    compression settings, as for LZMACompressor, LZMADecompressor and
    LZMAFile.

    For binary mode, this function is equivalent to the LZMAFile
    constructor: LZMAFile(filename, mode, ...). In this case, the
    encoding, errors and newline arguments must not be provided.

    For text mode, an LZMAFile object is created, and wrapped in an
    io.TextIOWrapper instance with the specified encoding, error
    handling behavior, and line ending(s).

    threads specifies the number of threads to use (default 1).
    Use 0 for auto-detect based on CPU count.
    """
    if "t" in mode:
        if "b" in mode:
            raise ValueError("Invalid mode: %r" % (mode,))
    else:
        if encoding is not None:
            raise ValueError("Argument 'encoding' not supported in binary mode")
        if errors is not None:
            raise ValueError("Argument 'errors' not supported in binary mode")
        if newline is not None:
            raise ValueError("Argument 'newline' not supported in binary mode")

    lz_mode = mode.replace("t", "")
    binary_file = LZMAFile(filename, lz_mode, format=format, check=check,
                           preset=preset, filters=filters, threads=threads)

    if "t" in mode:
        encoding = io.text_encoding(encoding)
        return io.TextIOWrapper(binary_file, encoding, errors, newline)
    else:
        return binary_file


__all__ = [
    # Core functions and classes
    "compress",
    "decompress",
    "open",
    "LZMACompressor",
    "LZMADecompressor",
    "LZMAFile",
    "LZMAError",
    # Check types
    "CHECK_NONE",
    "CHECK_CRC32",
    "CHECK_CRC64",
    "CHECK_SHA256",
    "CHECK_ID_MAX",
    "CHECK_UNKNOWN",
    # Presets
    "PRESET_DEFAULT",
    "PRESET_EXTREME",
    # Format constants
    "FORMAT_AUTO",
    "FORMAT_XZ",
    "FORMAT_ALONE",
    "FORMAT_RAW",
    # Filter IDs
    "FILTER_LZMA1",
    "FILTER_LZMA2",
    "FILTER_DELTA",
    "FILTER_X86",
    "FILTER_POWERPC",
    "FILTER_IA64",
    "FILTER_ARM",
    "FILTER_ARMTHUMB",
    "FILTER_SPARC",
    # Match finder types
    "MF_HC3",
    "MF_HC4",
    "MF_BT2",
    "MF_BT3",
    "MF_BT4",
    # Compression modes
    "MODE_FAST",
    "MODE_NORMAL",
    # Utility functions
    "is_check_supported",
    "get_xz_version",
    "is_mt_decoder_safe",
    "__version__",
]

# Set the __module__ attribute of all exported functions/classes to this module.
# This is necessary for sphinx-codeautolink to correctly resolve references like
# `LZMACompressor` to `lzma_mt.LZMACompressor` in code blocks. Without this,
# sphinx-codeautolink cannot link names that are imported (e.g.,
# `from lzma_mt import LZMACompressor`) because it doesn't know that `LZMACompressor`
# refers to `lzma_mt.LZMACompressor` rather than `lzma_mt.lzma_mt.LZMACompressor`.
# The _module_original_ attribute preserves the true module for use by docs/conf.py
# when resolving source links.
# Note: Cython extension types (cdef class) are immutable and cannot have their
# attributes modified, so we skip those silently.
for _x in __all__:
    _obj = globals().get(_x)
    if _obj is not None and hasattr(_obj, "__module__"):
        try:
            _obj._module_original_ = _obj.__module__
            _obj.__module__ = __name__
        except (TypeError, AttributeError):
            # Cython extension types and other immutable types can't be modified
            pass
