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


# Use the stdlib exception class so that `except lzma.LZMAError` and
# `except lzma_mt.LZMAError` are interchangeable, including for errors
# raised by the stdlib fallback paths.
LZMAError = _lzma.LZMAError


# CVE-2025-31115: lzma_stream_decoder_mt has use-after-free in xz 5.3.3alpha-5.8.0
# liblzma encodes versions as major*10000000 + minor*10000 + patch*10 + stability,
# where stability is 0=alpha, 1=beta, 2=stable (see lzma/version.h).
DEF LZMA_VERSION_VULNERABLE_START = 50030030  # 5.3.3alpha
DEF LZMA_VERSION_MT_SAFE = 50080012           # 5.8.1 (first fixed release)

# Constants used internally (others are exported directly from __init__.py)
from lzma import (
    CHECK_CRC64, CHECK_UNKNOWN, PRESET_DEFAULT,
    FORMAT_AUTO, FORMAT_XZ, FORMAT_ALONE, FORMAT_RAW,
)

DEF INITIAL_BUFFER_SIZE = 65536  # 64 KB

# Magic bytes at the start of every .xz stream
_XZ_MAGIC = b"\xfd7zXZ\x00"


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


cdef inline uint64_t _default_memlimit_threading() noexcept nogil:
    """Default soft memory limit for MT decoding: a quarter of physical RAM.

    When decoding would need more than this, liblzma reduces the thread
    count or falls back to direct mode (output stays correct). This keeps
    adversarial or very large-block streams from causing huge allocations,
    mirroring the xz tool's default.
    """
    cdef uint64_t phys = lzma.lzma_physmem()
    if phys == 0:
        return 1024 * 1024 * 1024  # RAM size unknown; assume 1 GiB is fine
    return phys // 4


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
    # TELL flags make lzma_code() report the integrity check type once the
    # stream header has been parsed (used for the .check property)
    opts.flags = lzma.LZMA_TELL_NO_CHECK | lzma.LZMA_TELL_ANY_CHECK
    if concatenated:
        opts.flags |= lzma.LZMA_CONCATENATED
    opts.memlimit_threading = memlimit_threading
    opts.memlimit_stop = memlimit_stop


cdef inline void _raise_lzma_error(lzma.lzma_ret ret) except *:
    """Raise an appropriate exception for an lzma error code.

    Exception types and messages match CPython's _lzmamodule.c, so code
    written against the stdlib sees the same errors.
    """
    if ret == lzma.LZMA_MEM_ERROR:
        raise MemoryError
    elif ret == lzma.LZMA_MEMLIMIT_ERROR:
        raise LZMAError("Memory usage limit exceeded")
    elif ret == lzma.LZMA_FORMAT_ERROR:
        raise LZMAError("Input format not supported by decoder")
    elif ret == lzma.LZMA_OPTIONS_ERROR:
        raise LZMAError("Invalid or unsupported options")
    elif ret == lzma.LZMA_DATA_ERROR:
        raise LZMAError("Corrupt input data")
    elif ret == lzma.LZMA_BUF_ERROR:
        raise LZMAError("Insufficient buffer space")
    elif ret == lzma.LZMA_PROG_ERROR:
        raise LZMAError("Internal error")
    elif ret == lzma.LZMA_UNSUPPORTED_CHECK:
        raise LZMAError("Unsupported integrity check")
    else:
        raise LZMAError(f"Unrecognized error from liblzma: {ret}")


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
                Multi-threading applies to XZ data only, and falls back to
                the single-threaded decoder on xz-utils versions affected
                by CVE-2025-31115.

    Returns:
        Decompressed data as bytes.
    """
    # The stdlib handles formats that liblzma's XZ/auto decoders don't
    # cover, as well as custom filter chains
    if format == FORMAT_RAW or format == FORMAT_ALONE or filters is not None:
        return _lzma.decompress(data, format=format, memlimit=memlimit, filters=filters)

    if format != FORMAT_AUTO and format != FORMAT_XZ:
        raise ValueError(f"Invalid container format: {format}")

    if threads < 0:
        raise ValueError(f"threads must be non-negative, got {threads}")

    # Match stdlib: accept any bytes-like object, reject the rest
    if not isinstance(data, bytes):
        data = bytes(memoryview(data))

    # Handle concatenated streams like the stdlib: decompress each stream
    # separately and ignore trailing junk after a complete stream
    results = []
    while True:
        decompressor = LZMADecompressor(format=format, memlimit=memlimit, threads=threads)
        try:
            result = decompressor.decompress(data)
        except LZMAError:
            if results:
                break  # Leftover data is not a valid LZMA/XZ stream; ignore it
            else:
                raise  # Error on the first stream

        results.append(result)

        if not decompressor.eof:
            raise LZMAError(
                "Compressed data ended before the end-of-stream marker was reached")

        data = decompressor.unused_data
        if not data:
            break

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

    With threads=1 (the default), the same single-threaded liblzma decoder
    as the stdlib is used, for all formats. With threads != 1, liblzma's
    multi-threaded decoder is used; it only supports the XZ format, so for
    FORMAT_AUTO the container format is detected from the first input bytes
    and non-XZ data is handed to the stdlib decompressor instead.

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
        int _check     # Integrity check type, CHECK_UNKNOWN until known
        bytes _unused_data
        bytes _input_buffer  # Buffered input not yet consumed by liblzma
        object _fallback  # stdlib LZMADecompressor for non-MT cases
        object _memlimit  # memlimit as passed by the caller
        # Deferred MT decoder setup for FORMAT_AUTO: the first input bytes
        # decide between the MT XZ decoder and the stdlib decompressor
        bint _pending_init
        uint32_t _pending_threads
        uint64_t _pending_memstop

    def __cinit__(self, format=FORMAT_AUTO, memlimit=None, filters=None, *, threads=1):
        """
        Initialize the decompressor.

        Args:
            format: Container format (FORMAT_AUTO, FORMAT_XZ, FORMAT_ALONE, FORMAT_RAW).
            memlimit: Memory limit in bytes. None means no limit.
            filters: Custom filter chain for FORMAT_RAW.
            threads: Number of threads (default 1). Use 0 for auto-detect.
                    Only used for XZ format. Falls back to the single-threaded
                    decoder on xz-utils versions with CVE-2025-31115.
        """
        cdef uint64_t mem_stop

        self._fallback = None
        self.initialized = False
        self._eof = False
        self._needs_input = True
        self._errored = False
        self._check = CHECK_UNKNOWN
        self._unused_data = b""
        self._input_buffer = b""
        self._memlimit = memlimit
        self._pending_init = False
        self.lock = NULL

        # The stdlib handles formats that liblzma's XZ/auto decoders don't
        # cover, as well as custom filter chains
        if format == FORMAT_RAW or format == FORMAT_ALONE or filters is not None:
            self._fallback = _lzma.LZMADecompressor(
                format=format, memlimit=memlimit, filters=filters)
            return

        if format != FORMAT_AUTO and format != FORMAT_XZ:
            raise ValueError(f"Invalid container format: {format}")

        # Parameter validation
        if threads < 0:
            raise ValueError(f"threads must be non-negative, got {threads}")

        if memlimit is not None:
            mem_stop = <uint64_t>memlimit
        else:
            mem_stop = UINT64_MAX

        # CVE-2025-31115: the MT decoder in xz 5.3.3alpha-5.8.0 has a
        # use-after-free bug, so use the single-threaded decoder there
        if threads != 1 and not _is_mt_decoder_safe():
            threads = 1

        # Allocate cross-platform lock (works on Windows, Linux, macOS)
        self.lock = lzma.PyThread_allocate_lock()
        if self.lock == NULL:
            raise MemoryError("Failed to allocate lock")

        try:
            if threads == 1:
                # Same decoder as the stdlib; under FORMAT_AUTO it also
                # handles FORMAT_ALONE data
                self._init_st_decoder(format, mem_stop)
            elif format == FORMAT_XZ:
                self._init_mt_decoder(<uint32_t>threads, mem_stop)
            else:
                # FORMAT_AUTO with multi-threading: the MT decoder only
                # supports XZ, so wait for the first input bytes to
                # identify the container format (see decompress())
                self._pending_init = True
                self._pending_threads = <uint32_t>threads
                self._pending_memstop = mem_stop
        except:
            lzma.PyThread_free_lock(self.lock)
            self.lock = NULL
            raise

    cdef void _init_st_decoder(self, object format, uint64_t mem_stop) except *:
        """Initialize the single-threaded decoder (same as the stdlib uses)."""
        cdef lzma.lzma_ret ret
        # TELL flags make lzma_code() report the integrity check type once
        # the stream header has been parsed (used for the .check property)
        cdef uint32_t flags = lzma.LZMA_TELL_NO_CHECK | lzma.LZMA_TELL_ANY_CHECK

        _init_stream(&self.strm)
        _setup_allocator(&self.alloc)
        self.strm.allocator = &self.alloc

        if format == FORMAT_XZ:
            ret = lzma.lzma_stream_decoder(&self.strm, mem_stop, flags)
        else:
            ret = lzma.lzma_auto_decoder(&self.strm, mem_stop, flags)
        if ret != lzma.LZMA_OK:
            lzma.lzma_end(&self.strm)  # Clean up any partial allocations
            _raise_lzma_error(ret)
        self.initialized = True

    cdef void _init_mt_decoder(self, uint32_t threads, uint64_t mem_stop) except *:
        """Initialize the multi-threaded decoder (XZ format only)."""
        cdef lzma.lzma_mt mt_options
        cdef lzma.lzma_ret ret
        cdef uint64_t mem_threading

        # Soft limit above which liblzma decodes with fewer threads or in
        # direct mode instead of failing
        if mem_stop != UINT64_MAX:
            mem_threading = mem_stop
        else:
            mem_threading = _default_memlimit_threading()

        _init_stream(&self.strm)
        _setup_allocator(&self.alloc)
        self.strm.allocator = &self.alloc

        _setup_decoder_mt(&mt_options, threads, mem_threading, mem_stop, False)
        ret = lzma.lzma_stream_decoder_mt(&self.strm, &mt_options)
        if ret != lzma.LZMA_OK:
            lzma.lzma_end(&self.strm)  # Clean up any partial allocations
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

        Returns CHECK_UNKNOWN until decompression has progressed far
        enough to determine it.
        """
        if self._fallback is not None:
            return self._fallback.check
        return self._check

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
            bint is_xz

        # Match stdlib: accept any bytes-like object, reject the rest
        if not isinstance(data, bytes):
            data = bytes(memoryview(data))

        # Release GIL before acquiring lock to prevent deadlock
        with nogil:
            lzma.PyThread_acquire_lock(self.lock, lzma.WAIT_LOCK)
        try:
            if self._eof:
                raise ValueError("Decompressor has reached end of stream")
            if self._errored:
                raise LZMAError("Decompressor is in an error state")

            # Combine buffered input with new data
            if self._input_buffer:
                combined_input = self._input_buffer + data
                self._input_buffer = b""
            else:
                combined_input = data

            # Deferred decoder choice for multi-threaded FORMAT_AUTO
            if self._pending_init:
                if len(combined_input) < len(_XZ_MAGIC):
                    if _XZ_MAGIC.startswith(combined_input):
                        # Not enough bytes yet to identify the format
                        self._input_buffer = combined_input
                        self._needs_input = True
                        return b""
                    is_xz = False
                else:
                    is_xz = combined_input.startswith(_XZ_MAGIC)
                self._pending_init = False
                if is_xz:
                    self._init_mt_decoder(self._pending_threads, self._pending_memstop)
                else:
                    # Not XZ (e.g. legacy .lzma): multi-threading does not
                    # apply, so delegate to the stdlib decompressor
                    self._fallback = _lzma.LZMADecompressor(
                        format=FORMAT_AUTO, memlimit=self._memlimit)
                    return self._fallback.decompress(combined_input, max_length)

            input_len = <size_t>len(combined_input)

            # Handle max_length=0: buffer input without producing output
            if max_length == 0:
                if input_len > 0:
                    self._input_buffer = combined_input
                    self._needs_input = False  # We have buffered data
                return b""

            input_view = combined_input
            # Avoid UB: don't take the address of an empty buffer. Empty
            # input still pumps lzma_code() once, so that output buffered
            # inside liblzma from earlier calls can be retrieved.
            input_data = &input_view[0] if input_len > 0 else NULL

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
