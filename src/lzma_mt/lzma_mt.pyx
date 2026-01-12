# cython: language_level=3

cimport cython
from cpython.mem cimport PyMem_RawMalloc, PyMem_RawFree
from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AS_STRING
from libc.string cimport memset, memcpy
from libc.stdint cimport uint8_t, uint32_t, uint64_t, UINT64_MAX, SIZE_MAX
import sys
import lzma as _lzma  # stdlib lzma for fallback
cdef Py_ssize_t PY_SSIZE_T_MAX = sys.maxsize

cimport lzma_mt.lzma as lzma


class LZMAError(Exception):
    """Exception raised for LZMA-related errors."""
    pass


# CVE-2025-31115: lzma_stream_decoder_mt has use-after-free in xz 5.3.3alpha-5.8.0
# Version numbers: 5.3.3alpha=50030301, 5.8.0=50080000, 5.8.1=50080001
DEF LZMA_VERSION_VULNERABLE_START = 50030301  # 5.3.3alpha
DEF LZMA_VERSION_VULNERABLE_END = 50080000    # 5.8.0 (inclusive)
DEF LZMA_VERSION_MT_SAFE = 50080001           # 5.8.1

# Constants used internally (others are exported directly from __init__.py)
from lzma import (
    CHECK_CRC64, CHECK_UNKNOWN, PRESET_DEFAULT,
    FORMAT_AUTO, FORMAT_XZ, FORMAT_ALONE, FORMAT_RAW,
)

DEF INITIAL_BUFFER_SIZE = 65536  # 64 KB


# =============================================================================
# CVE-2025-31115 Version Check
# =============================================================================

cdef bint _is_mt_decoder_safe() noexcept nogil:
    """Check if MT decoder is safe (xz >= 5.8.1 or < 5.3.3alpha)."""
    cdef uint32_t version = lzma.lzma_version_number()
    if version >= LZMA_VERSION_MT_SAFE:
        return True
    if version < LZMA_VERSION_VULNERABLE_START:
        return True
    return False


cdef inline void _check_mt_decoder_version() except *:
    """Raise RuntimeError if MT decoder has CVE-2025-31115 vulnerability."""
    if not _is_mt_decoder_safe():
        version_str = lzma.lzma_version_string().decode('ascii')
        raise RuntimeError(
            f"xz-utils {version_str} has CVE-2025-31115 vulnerability in MT decoder. "
            f"Upgrade to 5.8.1+ or use threads=1 for untrusted input."
        )


def get_xz_version():
    """Return the xz-utils version string."""
    return lzma.lzma_version_string().decode('ascii')


def is_mt_decoder_safe():
    """Check if multi-threaded decoder is safe from CVE-2025-31115."""
    return _is_mt_decoder_safe()


# =============================================================================
# Custom LZMA Allocator (GIL-free with overflow check)
# =============================================================================

cdef void* _lzma_alloc(void *opaque, size_t nmemb, size_t size) noexcept nogil:
    """Allocate memory for liblzma internal use (GIL-free)."""
    if nmemb == 0 or size == 0:
        return NULL
    # Overflow check before multiplication
    if size > SIZE_MAX / nmemb:
        return NULL
    return PyMem_RawMalloc(nmemb * size)


cdef void _lzma_free(void *opaque, void *ptr) noexcept nogil:
    """Free memory allocated by _lzma_alloc."""
    PyMem_RawFree(ptr)


cdef inline void _setup_allocator(lzma.lzma_allocator *alloc) noexcept nogil:
    """Initialize custom allocator for liblzma."""
    alloc.alloc = _lzma_alloc
    alloc.free = _lzma_free
    alloc.opaque = NULL


# =============================================================================
# Helper functions
# =============================================================================

cdef inline void _init_stream(lzma.lzma_stream *strm) noexcept nogil:
    """Zero-initialize an lzma_stream."""
    memset(strm, 0, sizeof(lzma.lzma_stream))


cdef inline uint32_t _get_effective_threads(uint32_t threads) noexcept nogil:
    """Get effective thread count, handling threads=0 and unknown CPU count."""
    cdef uint32_t effective = threads
    if effective == 0:
        effective = lzma.lzma_cputhreads()
        if effective == 0:
            effective = 1  # Fallback if CPU count unknown
    return effective


cdef inline void _setup_encoder_mt(
    lzma.lzma_mt *opts,
    uint32_t preset,
    lzma.lzma_check check,
    uint32_t threads
) noexcept nogil:
    """Initialize MT encoder options."""
    memset(opts, 0, sizeof(lzma.lzma_mt))
    opts.threads = _get_effective_threads(threads)
    opts.preset = preset
    opts.check = check
    opts.block_size = 0
    opts.timeout = 0


cdef inline void _setup_decoder_mt(
    lzma.lzma_mt *opts,
    uint32_t threads,
    uint64_t memlimit_threading,
    uint64_t memlimit_stop,
    bint concatenated
) noexcept nogil:
    """Initialize MT decoder options.

    Args:
        opts: MT options struct to initialize
        threads: Thread count (0 = auto-detect)
        memlimit_threading: Soft limit - library reduces threads if exceeded
        memlimit_stop: Hard limit - operation fails if exceeded
        concatenated: Whether to handle concatenated streams
    """
    memset(opts, 0, sizeof(lzma.lzma_mt))
    opts.threads = _get_effective_threads(threads)
    opts.flags = lzma.LZMA_CONCATENATED if concatenated else 0
    opts.memlimit_threading = memlimit_threading
    opts.memlimit_stop = memlimit_stop


cdef inline void _raise_lzma_error(lzma.lzma_ret ret) except *:
    """Raise an appropriate exception for an lzma error code."""
    if ret == lzma.LZMA_MEM_ERROR:
        raise MemoryError("LZMA: Memory allocation failed")
    elif ret == lzma.LZMA_MEMLIMIT_ERROR:
        raise MemoryError("LZMA: Memory limit exceeded")
    elif ret == lzma.LZMA_FORMAT_ERROR:
        raise LZMAError("LZMA: Input format not recognized")
    elif ret == lzma.LZMA_OPTIONS_ERROR:
        raise LZMAError("LZMA: Invalid or unsupported options")
    elif ret == lzma.LZMA_DATA_ERROR:
        raise LZMAError("LZMA: Data is corrupt")
    elif ret == lzma.LZMA_BUF_ERROR:
        raise LZMAError("LZMA: Buffer error (truncated input?)")
    elif ret == lzma.LZMA_PROG_ERROR:
        raise LZMAError("LZMA: Programming error")
    else:
        raise LZMAError(f"LZMA: Unknown error code {ret}")


# Block sizes matching CPython's pycore_blocks_output_buffer.h
# Progressive growth: 32KB -> 64KB -> 256KB -> 1MB -> 4MB -> ... -> 256MB max
DEF KB = 1024
DEF MB = 1024 * 1024
DEF OUTPUT_BUFFER_MAX_BLOCK_SIZE = 256 * MB

cdef Py_ssize_t[17] BUFFER_BLOCK_SIZE
BUFFER_BLOCK_SIZE[:] = [
    32*KB, 64*KB, 256*KB, 1*MB, 4*MB, 8*MB, 16*MB, 16*MB,
    32*MB, 32*MB, 32*MB, 32*MB, 64*MB, 64*MB, 128*MB, 128*MB,
    OUTPUT_BUFFER_MAX_BLOCK_SIZE
]
DEF BUFFER_BLOCK_SIZE_LEN = 17


cdef class _BlocksOutputBuffer:
    """Block-based output buffer matching CPython's pycore_blocks_output_buffer.h

    Uses immutable bytes objects (not bytearray) for stable buffer pointers.
    Tracks allocated size internally (not using strm.total_out) for streaming support.
    """
    cdef:
        list blocks              # List of bytes objects
        Py_ssize_t allocated     # Total allocated size across all blocks
        Py_ssize_t max_length    # Max output length (-1 = unlimited)

    def __cinit__(self):
        self.blocks = None  # Set to None for error detection
        self.allocated = 0
        self.max_length = -1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline Py_ssize_t _get_block_size(self, Py_ssize_t list_len) noexcept:
        """Get block size for given list position."""
        if list_len < BUFFER_BLOCK_SIZE_LEN:
            return BUFFER_BLOCK_SIZE[list_len]
        return BUFFER_BLOCK_SIZE[BUFFER_BLOCK_SIZE_LEN - 1]

    @cython.boundscheck(False)
    cdef inline Py_ssize_t init_and_grow(
            self, Py_ssize_t max_length, uint8_t **next_out) except -1:
        """Initialize buffer and allocate first block.

        Returns allocated size on success, -1 on failure.
        """
        cdef Py_ssize_t block_size
        cdef object b

        # Ensure .blocks was set to None (not reused)
        assert self.blocks is None

        # Get block size
        block_size = BUFFER_BLOCK_SIZE[0]
        if 0 <= max_length < block_size:
            block_size = max_length if max_length > 0 else 1

        # Create first block (uninitialized bytes object)
        b = PyBytes_FromStringAndSize(NULL, block_size)
        if b is None:
            raise MemoryError("Unable to allocate output buffer")

        # Create the list
        self.blocks = [b]

        # Set variables
        self.allocated = block_size
        self.max_length = max_length

        next_out[0] = <uint8_t *>PyBytes_AS_STRING(b)
        return block_size

    @cython.boundscheck(False)
    cdef inline Py_ssize_t grow(
            self, Py_ssize_t avail_out, uint8_t **next_out) except -1:
        """Grow buffer by allocating next block.

        Must be called when avail_out == 0.
        Returns new block size on success, -1 on failure.
        """
        cdef Py_ssize_t list_len = len(self.blocks)
        cdef Py_ssize_t block_size
        cdef Py_ssize_t rest
        cdef object b

        # Ensure no gaps in the data
        if avail_out != 0:
            raise SystemError("avail_out is non-zero in _BlocksOutputBuffer.grow()")

        # Get block size based on list position
        block_size = self._get_block_size(list_len)

        # Check max_length constraint
        if self.max_length >= 0:
            rest = self.max_length - self.allocated
            if rest <= 0:
                return 0  # Should not grow
            if block_size > rest:
                block_size = rest

        # Check overflow
        if block_size > PY_SSIZE_T_MAX - self.allocated:
            raise MemoryError("Unable to allocate output buffer")

        # Create the block (uninitialized bytes object)
        b = PyBytes_FromStringAndSize(NULL, block_size)
        if b is None:
            raise MemoryError("Unable to allocate output buffer")

        self.blocks.append(b)

        # Update tracking
        self.allocated += block_size

        next_out[0] = <uint8_t *>PyBytes_AS_STRING(b)
        return block_size

    cdef inline Py_ssize_t get_data_size(self, Py_ssize_t avail_out) noexcept:
        """Return current output data size."""
        return self.allocated - avail_out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline object finish(self, Py_ssize_t avail_out):
        """Finish buffer and return bytes object.

        Returns bytes object on success, NULL on failure.
        """
        cdef Py_ssize_t list_len = len(self.blocks)
        cdef Py_ssize_t data_size = self.allocated - avail_out
        cdef object result
        cdef object block
        cdef char *posi
        cdef Py_ssize_t i
        cdef Py_ssize_t block_len

        # Fast path: single block fully used
        if list_len == 1 and avail_out == 0:
            block = self.blocks[0]
            self.blocks = None
            return block

        # Fast path: two blocks, second one completely unused
        if list_len == 2:
            block = self.blocks[1]
            if len(block) == avail_out:
                result = self.blocks[0]
                self.blocks = None
                return result

        # General case: create final bytes and copy
        if data_size == 0:
            self.blocks = None
            return b""

        result = PyBytes_FromStringAndSize(NULL, data_size)
        if result is None:
            raise MemoryError("Unable to allocate output buffer")

        # Memory copy from all blocks
        posi = PyBytes_AS_STRING(result)

        # Copy all blocks except the last one (fully used)
        for i in range(list_len - 1):
            block = self.blocks[i]
            block_len = len(block)
            memcpy(posi, PyBytes_AS_STRING(block), block_len)
            posi += block_len

        # Copy last block (partially used)
        if list_len > 0:
            block = self.blocks[list_len - 1]
            block_len = len(block) - avail_out
            if block_len > 0:
                memcpy(posi, PyBytes_AS_STRING(block), block_len)

        self.blocks = None
        return result

    cdef inline void on_error(self) noexcept:
        """Clean up on error."""
        self.blocks = None


# =============================================================================
# One-shot functions
# =============================================================================

def compress(data, format=FORMAT_XZ, check=-1, preset=None, filters=None, *, threads=1):
    """
    Compress data using LZMA/XZ compression.

    Matches the stdlib lzma.compress() API exactly, with an additional
    'threads' parameter for multi-threaded compression.

    Args:
        data: Bytes-like object to compress.
        format: Container format (FORMAT_XZ, FORMAT_ALONE, FORMAT_RAW).
               Default is FORMAT_XZ.
        check: Integrity check type. -1 means default (CHECK_CRC64 for XZ).
        preset: Compression level 0-9, optionally OR'd with PRESET_EXTREME.
               Default is PRESET_DEFAULT (6).
        filters: Custom filter chain (list of dicts). If specified, preset
                is ignored.
        threads: Number of threads (default 1). Use 0 for auto-detect.
                Only used for FORMAT_XZ without custom filters.

    Returns:
        Compressed data as bytes.
    """
    # Fall back to stdlib for non-XZ formats or custom filters
    if format != FORMAT_XZ or filters is not None:
        return _lzma.compress(data, format=format, check=check,
                              preset=preset, filters=filters)

    # Handle defaults
    if preset is None:
        preset = PRESET_DEFAULT
    if check == -1:
        check = CHECK_CRC64

    # Parameter validation
    if not isinstance(preset, int):
        raise TypeError("an integer is required")
    if threads < 0:
        raise ValueError(f"threads must be non-negative, got {threads}")

    cdef uint32_t c_preset = <uint32_t>preset
    cdef:
        lzma.lzma_stream strm
        lzma.lzma_allocator alloc
        lzma.lzma_mt mt_options
        lzma.lzma_ret ret
        _BlocksOutputBuffer buf = _BlocksOutputBuffer()
        const unsigned char[::1] input_view = data  # zero-copy buffer view
        size_t input_len = <size_t>input_view.shape[0]
        # Avoid UB: don't take address of empty buffer
        const uint8_t *input_data = &input_view[0] if input_len > 0 else NULL
        uint8_t *next_out
        Py_ssize_t avail_out

    _init_stream(&strm)
    _setup_allocator(&alloc)
    strm.allocator = &alloc
    _setup_encoder_mt(&mt_options, c_preset, <lzma.lzma_check>check, <uint32_t>threads)

    ret = lzma.lzma_stream_encoder_mt(&strm, &mt_options)
    if ret != lzma.LZMA_OK:
        _raise_lzma_error(ret)

    try:
        avail_out = buf.init_and_grow(-1, &next_out)
        strm.next_out = next_out
        strm.avail_out = <size_t>avail_out
        strm.next_in = input_data
        strm.avail_in = input_len

        while True:
            with nogil:
                ret = lzma.lzma_code(&strm, lzma.LZMA_FINISH)
            if ret == lzma.LZMA_STREAM_END:
                break
            elif ret == lzma.LZMA_OK and strm.avail_out == 0:
                avail_out = buf.grow(0, &next_out)
                strm.next_out = next_out
                strm.avail_out = <size_t>avail_out
            elif ret != lzma.LZMA_OK:
                _raise_lzma_error(ret)

        return buf.finish(<Py_ssize_t>strm.avail_out)
    finally:
        lzma.lzma_end(&strm)
        buf.on_error()


def decompress(data, format=FORMAT_AUTO, memlimit=None, filters=None, *, threads=1):
    """
    Decompress LZMA/XZ data.

    Matches the stdlib lzma.decompress() API exactly, with an additional
    'threads' parameter for multi-threaded decompression.

    Args:
        data: Compressed bytes-like object.
        format: Container format (FORMAT_AUTO, FORMAT_XZ, FORMAT_ALONE, FORMAT_RAW).
               Default is FORMAT_AUTO which auto-detects.
        memlimit: Memory limit in bytes. None means no limit (default).
        filters: Custom filter chain for FORMAT_RAW (list of dicts).
        threads: Number of threads (default 1). Use 0 for auto-detect.
                Only used for XZ format.

    Returns:
        Decompressed data as bytes.
    """
    # Fall back to stdlib for formats that don't support MT, or custom filters
    # MT decoder only supports XZ format (FORMAT_XZ and FORMAT_AUTO which auto-detects)
    if format == FORMAT_RAW or format == FORMAT_ALONE or filters is not None:
        return _lzma.decompress(data, format=format, memlimit=memlimit, filters=filters)

    # For FORMAT_AUTO or FORMAT_XZ, we can use MT decoder
    # Parameter validation
    if threads < 0:
        raise ValueError(f"threads must be non-negative, got {threads}")

    # CVE-2025-31115: silently fall back to single-threaded on vulnerable versions
    if threads != 1 and not _is_mt_decoder_safe():
        threads = 1

    # Handle concatenated streams like CPython: decompress each stream separately
    # and catch errors on trailing junk (matching lzma.decompress behavior)
    results = []
    remaining = bytes(data)

    while remaining:
        try:
            decompressor = LZMADecompressor(
                format=format,
                memlimit=memlimit,
                threads=threads
            )
            result = decompressor.decompress(remaining)
        except (LZMAError, MemoryError):
            if results:
                # We already have results; treat this as trailing junk
                break
            else:
                # Error on first stream; re-raise
                raise

        results.append(result)

        if not decompressor.eof:
            raise LZMAError(
                "Compressed data ended before the end-of-stream marker was reached")

        remaining = decompressor.unused_data

    return b"".join(results)


# =============================================================================
# Streaming classes
# =============================================================================

cdef class LZMACompressor:
    """Streaming LZMA compressor with multi-threading support.

    Matches the stdlib lzma.LZMACompressor API exactly, with an additional
    'threads' parameter for multi-threaded compression.

    Note on thread safety: Methods are protected by internal locks and will not
    crash when called from multiple Python threads. However, interleaved calls
    produce output in undefined order. For predictable output, use one thread
    per compressor instance or serialize access externally.
    """
    cdef:
        lzma.lzma_stream strm
        lzma.lzma_allocator alloc
        lzma.PyThread_type_lock lock
        bint initialized
        bint flushed
        object _fallback  # stdlib LZMACompressor for non-MT cases

    def __cinit__(self, format=FORMAT_XZ, check=-1, preset=None,
                  filters=None, *, threads=1):
        cdef lzma.lzma_mt mt_options
        cdef lzma.lzma_ret ret

        self._fallback = None
        self.initialized = False
        self.flushed = False
        self.lock = NULL

        # Fall back to stdlib for non-XZ formats or custom filters
        if format != FORMAT_XZ or filters is not None:
            self._fallback = _lzma.LZMACompressor(
                format=format, check=check, preset=preset, filters=filters)
            return

        # Handle defaults
        if preset is None:
            preset = PRESET_DEFAULT
        if check == -1:
            check = CHECK_CRC64

        # Parameter validation
        if not isinstance(preset, int):
            raise TypeError("an integer is required")
        cdef uint32_t c_preset = <uint32_t>preset
        cdef lzma.lzma_check c_check = <lzma.lzma_check>check
        if threads < 0:
            raise ValueError(f"threads must be non-negative, got {threads}")

        _init_stream(&self.strm)

        # Allocate cross-platform lock (works on Windows, Linux, macOS)
        self.lock = lzma.PyThread_allocate_lock()
        if self.lock == NULL:
            raise MemoryError("Failed to allocate lock")

        # Set up custom allocator
        _setup_allocator(&self.alloc)
        self.strm.allocator = &self.alloc

        _setup_encoder_mt(&mt_options, c_preset, c_check, <uint32_t>threads)
        ret = lzma.lzma_stream_encoder_mt(&self.strm, &mt_options)
        if ret != lzma.LZMA_OK:
            lzma.lzma_end(&self.strm)  # Clean up any partial allocations
            lzma.PyThread_free_lock(self.lock)
            self.lock = NULL
            _raise_lzma_error(ret)

        self.initialized = True

    def __dealloc__(self):
        # Acquire lock to prevent race with concurrent compress()/flush() calls
        if self.lock != NULL:
            lzma.PyThread_acquire_lock(self.lock, lzma.WAIT_LOCK)
        if self.initialized:
            lzma.lzma_end(&self.strm)
        if self.lock != NULL:
            lzma.PyThread_release_lock(self.lock)
            lzma.PyThread_free_lock(self.lock)

    def compress(self, data):
        """Compress data and return any available output.

        Thread-safe: protected by internal lock.
        """
        # Handle fallback to stdlib
        if self._fallback is not None:
            return self._fallback.compress(data)

        cdef:
            _BlocksOutputBuffer buf
            const unsigned char[::1] input_view
            const uint8_t *input_data
            size_t input_len
            lzma.lzma_ret ret
            uint8_t *next_out
            Py_ssize_t avail_out

        # Release GIL before acquiring lock to prevent deadlock:
        # Otherwise Thread A (holding lock, waiting for GIL) and
        # Thread B (holding GIL, waiting for lock) would deadlock.
        with nogil:
            lzma.PyThread_acquire_lock(self.lock, lzma.WAIT_LOCK)
        try:
            if self.flushed:
                raise ValueError("Compressor has been flushed")

            # Fast path for empty input
            if len(data) == 0:
                return b""

            input_view = data  # zero-copy buffer view
            input_data = &input_view[0]
            input_len = <size_t>input_view.shape[0]

            buf = _BlocksOutputBuffer()
            avail_out = buf.init_and_grow(-1, &next_out)
            self.strm.next_out = next_out
            self.strm.avail_out = <size_t>avail_out
            self.strm.next_in = input_data
            self.strm.avail_in = input_len

            while self.strm.avail_in > 0:
                with nogil:
                    ret = lzma.lzma_code(&self.strm, lzma.LZMA_RUN)
                if ret != lzma.LZMA_OK:
                    buf.on_error()
                    _raise_lzma_error(ret)
                if self.strm.avail_out == 0:
                    avail_out = buf.grow(0, &next_out)
                    self.strm.next_out = next_out
                    self.strm.avail_out = <size_t>avail_out

            return buf.finish(<Py_ssize_t>self.strm.avail_out)
        finally:
            lzma.PyThread_release_lock(self.lock)

    def flush(self):
        """Finish compression and return remaining data.

        Thread-safe: protected by internal lock.
        """
        # Handle fallback to stdlib
        if self._fallback is not None:
            return self._fallback.flush()

        cdef:
            _BlocksOutputBuffer buf
            lzma.lzma_ret ret
            uint8_t *next_out
            Py_ssize_t avail_out

        # Release GIL before acquiring lock to prevent deadlock
        with nogil:
            lzma.PyThread_acquire_lock(self.lock, lzma.WAIT_LOCK)
        try:
            if self.flushed:
                raise ValueError("Compressor has been flushed")

            buf = _BlocksOutputBuffer()
            avail_out = buf.init_and_grow(-1, &next_out)
            self.strm.next_out = next_out
            self.strm.avail_out = <size_t>avail_out
            self.strm.next_in = NULL
            self.strm.avail_in = 0

            while True:
                with nogil:
                    ret = lzma.lzma_code(&self.strm, lzma.LZMA_FINISH)
                if ret == lzma.LZMA_STREAM_END:
                    break
                elif ret == lzma.LZMA_OK and self.strm.avail_out == 0:
                    avail_out = buf.grow(0, &next_out)
                    self.strm.next_out = next_out
                    self.strm.avail_out = <size_t>avail_out
                elif ret != lzma.LZMA_OK:
                    buf.on_error()
                    _raise_lzma_error(ret)

            self.flushed = True
            return buf.finish(<Py_ssize_t>self.strm.avail_out)
        finally:
            lzma.PyThread_release_lock(self.lock)


cdef class LZMADecompressor:
    """Streaming LZMA decompressor with multi-threading support.

    Matches the stdlib lzma.LZMADecompressor API exactly, with an additional
    'threads' parameter for multi-threaded decompression.

    WARNING: When processing untrusted input, always set memlimit to prevent
    decompression bombs from exhausting memory.

    Note on thread safety: Methods are protected by internal locks and will not
    crash when called from multiple Python threads. However, decompression is
    inherently sequential, so concurrent calls are rarely useful. For most use
    cases, use one decompressor instance per thread.
    """
    cdef:
        lzma.lzma_stream strm
        lzma.lzma_allocator alloc
        lzma.PyThread_type_lock lock
        bint initialized
        bint _eof
        bint _needs_input
        bint _errored  # Track if decompression has encountered an error
        bytes _unused_data
        bytes _input_buffer  # Buffered input not yet consumed by liblzma
        object _fallback  # stdlib LZMADecompressor for non-MT cases

    def __cinit__(self, format=FORMAT_AUTO, memlimit=None, filters=None, *, threads=1):
        """
        Initialize the decompressor.

        Args:
            format: Container format (FORMAT_AUTO, FORMAT_XZ, FORMAT_ALONE, FORMAT_RAW).
            memlimit: Memory limit in bytes. None means no limit.
            filters: Custom filter chain for FORMAT_RAW.
            threads: Number of threads (default 1). Use 0 for auto-detect.
                    Only used for XZ format. Silently falls back to 1 on
                    xz-utils versions with CVE-2025-31115.
        """
        cdef lzma.lzma_mt mt_options
        cdef lzma.lzma_ret ret
        cdef uint64_t mem_stop

        self._fallback = None
        self.initialized = False
        self._eof = False
        self._needs_input = True
        self._errored = False
        self._unused_data = b""
        self._input_buffer = b""
        self.lock = NULL

        # Fall back to stdlib for formats that don't support MT, or custom filters
        # MT decoder only supports XZ format (FORMAT_XZ and FORMAT_AUTO)
        if format == FORMAT_RAW or format == FORMAT_ALONE or filters is not None:
            self._fallback = _lzma.LZMADecompressor(
                format=format, memlimit=memlimit, filters=filters)
            return

        # Parameter validation
        if threads < 0:
            raise ValueError(f"threads must be non-negative, got {threads}")

        if memlimit is not None:
            mem_stop = <uint64_t>memlimit
        else:
            mem_stop = UINT64_MAX

        # CVE-2025-31115: silently fall back to single-threaded on vulnerable versions
        if threads != 1 and not _is_mt_decoder_safe():
            threads = 1

        _init_stream(&self.strm)

        # Allocate cross-platform lock (works on Windows, Linux, macOS)
        self.lock = lzma.PyThread_allocate_lock()
        if self.lock == NULL:
            raise MemoryError("Failed to allocate lock")

        # Set up custom allocator
        _setup_allocator(&self.alloc)
        self.strm.allocator = &self.alloc

        _setup_decoder_mt(&mt_options, <uint32_t>threads, mem_stop, mem_stop, False)
        ret = lzma.lzma_stream_decoder_mt(&self.strm, &mt_options)
        if ret != lzma.LZMA_OK:
            lzma.lzma_end(&self.strm)  # Clean up any partial allocations
            lzma.PyThread_free_lock(self.lock)
            self.lock = NULL
            _raise_lzma_error(ret)

        self.initialized = True

    def __dealloc__(self):
        # Acquire lock to prevent race with concurrent decompress() calls
        if self.lock != NULL:
            lzma.PyThread_acquire_lock(self.lock, lzma.WAIT_LOCK)
        if self.initialized:
            lzma.lzma_end(&self.strm)
        if self.lock != NULL:
            lzma.PyThread_release_lock(self.lock)
            lzma.PyThread_free_lock(self.lock)

    @property
    def eof(self):
        """True if end of stream has been reached."""
        if self._fallback is not None:
            return self._fallback.eof
        return self._eof

    @property
    def needs_input(self):
        """True if more input data is needed to continue decompression.

        This is False after EOF is reached or when the output buffer was filled
        before all input was consumed.
        """
        if self._fallback is not None:
            return self._fallback.needs_input
        return self._needs_input

    @property
    def unused_data(self):
        """Data found after the end of the compressed stream."""
        if self._fallback is not None:
            return self._fallback.unused_data
        return self._unused_data

    @property
    def check(self):
        """Return the integrity check type used by the compressed stream.

        Returns CHECK_UNKNOWN before decompression begins or if the
        check type cannot be determined.
        """
        if self._fallback is not None:
            return self._fallback.check
        if not self.initialized:
            return CHECK_UNKNOWN
        return lzma.lzma_get_check(&self.strm)

    def decompress(self, data, max_length=-1):
        """
        Decompress data and return decompressed output.

        Thread-safe: protected by internal lock.

        Args:
            data: Compressed data to decompress.
            max_length: Maximum bytes to return (-1 for unlimited).

        Returns:
            Decompressed bytes.
        """
        # Handle fallback to stdlib
        if self._fallback is not None:
            return self._fallback.decompress(data, max_length)

        cdef:
            _BlocksOutputBuffer buf
            const unsigned char[::1] input_view
            const uint8_t *input_data
            size_t input_len
            size_t remaining_start
            lzma.lzma_ret ret
            uint8_t *next_out
            Py_ssize_t avail_out
            bytes combined_input

        # Release GIL before acquiring lock to prevent deadlock
        with nogil:
            lzma.PyThread_acquire_lock(self.lock, lzma.WAIT_LOCK)
        try:
            if self._eof:
                raise ValueError("Decompressor has reached end of stream")
            if self._errored:
                raise ValueError("Decompressor encountered an error")

            # Combine buffered input with new data
            if self._input_buffer:
                combined_input = self._input_buffer + bytes(data)
                self._input_buffer = b""
            else:
                combined_input = bytes(data)

            input_len = <size_t>len(combined_input)

            # Handle max_length=0: buffer input without producing output
            if max_length == 0:
                if input_len > 0:
                    self._input_buffer = combined_input
                    self._needs_input = False  # We have buffered data
                return b""

            # Fast path: no input and no buffered data
            if input_len == 0:
                self._needs_input = True
                return b""

            # Get pointer to combined input
            input_view = combined_input
            input_data = &input_view[0]

            buf = _BlocksOutputBuffer()
            avail_out = buf.init_and_grow(<Py_ssize_t>max_length, &next_out)
            self.strm.next_out = next_out
            self.strm.avail_out = <size_t>avail_out
            self.strm.next_in = input_data
            self.strm.avail_in = input_len

            while True:
                with nogil:
                    ret = lzma.lzma_code(&self.strm, lzma.LZMA_RUN)

                if ret == lzma.LZMA_STREAM_END:
                    self._eof = True
                    self._needs_input = False
                    if self.strm.avail_in > 0:
                        remaining_start = input_len - self.strm.avail_in
                        self._unused_data = combined_input[remaining_start:]
                    break
                elif ret == lzma.LZMA_OK:
                    if self.strm.avail_in == 0:
                        self._needs_input = True  # Need more input
                        break
                    if self.strm.avail_out == 0:
                        if max_length >= 0:
                            # Output limit hit, buffer remaining input
                            remaining_start = input_len - self.strm.avail_in
                            self._input_buffer = combined_input[remaining_start:]
                            self._needs_input = False
                            break
                        avail_out = buf.grow(0, &next_out)
                        self.strm.next_out = next_out
                        self.strm.avail_out = <size_t>avail_out
                else:
                    buf.on_error()
                    self._errored = True
                    _raise_lzma_error(ret)

            return buf.finish(<Py_ssize_t>self.strm.avail_out)
        finally:
            lzma.PyThread_release_lock(self.lock)
