# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

from cpython.bytes cimport PyBytes_AS_STRING, PyBytes_FromStringAndSize
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libc.string cimport memset, memcpy
from libc.stdint cimport uint8_t, uint32_t, uint64_t, UINT64_MAX

cimport lzma_mt.lzma as lzma

# Constants matching stdlib lzma module
CHECK_NONE = lzma.LZMA_CHECK_NONE
CHECK_CRC32 = lzma.LZMA_CHECK_CRC32
CHECK_CRC64 = lzma.LZMA_CHECK_CRC64
CHECK_SHA256 = lzma.LZMA_CHECK_SHA256

PRESET_DEFAULT = lzma.LZMA_PRESET_DEFAULT
PRESET_EXTREME = lzma.LZMA_PRESET_EXTREME

# Initial buffer size for compression output
DEF INITIAL_BUFFER_SIZE = 65536  # 64 KB


cdef inline void _init_stream(lzma.lzma_stream *strm) noexcept nogil:
    """Zero-initialize an lzma_stream (equivalent to LZMA_STREAM_INIT)."""
    memset(strm, 0, sizeof(lzma.lzma_stream))


cdef bytes _raise_lzma_error(lzma.lzma_ret ret):
    """Raise an appropriate exception for an lzma error code."""
    if ret == lzma.LZMA_MEM_ERROR:
        raise MemoryError("LZMA: Memory allocation failed")
    elif ret == lzma.LZMA_MEMLIMIT_ERROR:
        raise MemoryError("LZMA: Memory limit exceeded")
    elif ret == lzma.LZMA_FORMAT_ERROR:
        raise ValueError("LZMA: Input format not recognized")
    elif ret == lzma.LZMA_OPTIONS_ERROR:
        raise ValueError("LZMA: Invalid or unsupported options")
    elif ret == lzma.LZMA_DATA_ERROR:
        raise ValueError("LZMA: Data is corrupt")
    elif ret == lzma.LZMA_BUF_ERROR:
        raise ValueError("LZMA: Buffer error (truncated input?)")
    elif ret == lzma.LZMA_PROG_ERROR:
        raise RuntimeError("LZMA: Programming error")
    else:
        raise RuntimeError(f"LZMA: Unknown error code {ret}")


def compress(data, int preset=6, *, unsigned int threads=0, int check=CHECK_CRC64):
    """
    Compress data using multi-threaded LZMA/XZ compression.

    Args:
        data: Bytes-like object to compress.
        preset: Compression level 0-9 (default 6). Higher = better compression but slower.
        threads: Number of threads. 0 = auto-detect CPU count (default).
        check: Integrity check type (CHECK_CRC64 by default).

    Returns:
        Compressed data as bytes.
    """
    cdef:
        lzma.lzma_stream strm
        lzma.lzma_mt mt_options
        lzma.lzma_ret ret
        const uint8_t *input_data
        Py_ssize_t input_len
        uint8_t *output_buf = NULL
        size_t output_size = INITIAL_BUFFER_SIZE
        size_t output_pos = 0
        bytes input_bytes
        bytes result

    # Convert input to bytes
    if isinstance(data, memoryview):
        input_bytes = bytes(data)
    else:
        input_bytes = bytes(data)

    input_data = <const uint8_t *>PyBytes_AS_STRING(input_bytes)
    input_len = len(input_bytes)

    # Initialize stream
    _init_stream(&strm)

    # Set up MT options
    memset(&mt_options, 0, sizeof(lzma.lzma_mt))
    mt_options.threads = threads if threads > 0 else lzma.lzma_cputhreads()
    mt_options.preset = <uint32_t>preset
    mt_options.check = <lzma.lzma_check>check
    mt_options.block_size = 0  # Let liblzma choose
    mt_options.timeout = 0  # No timeout

    # Initialize MT encoder
    ret = lzma.lzma_stream_encoder_mt(&strm, &mt_options)
    if ret != lzma.LZMA_OK:
        _raise_lzma_error(ret)

    try:
        # Allocate output buffer
        output_buf = <uint8_t *>PyMem_Malloc(output_size)
        if output_buf == NULL:
            raise MemoryError("Failed to allocate output buffer")

        strm.next_in = input_data
        strm.avail_in = <size_t>input_len
        strm.next_out = output_buf
        strm.avail_out = output_size

        while True:
            ret = lzma.lzma_code(&strm, lzma.LZMA_FINISH)

            if ret == lzma.LZMA_STREAM_END:
                break
            elif ret == lzma.LZMA_OK:
                # Need more output space
                if strm.avail_out == 0:
                    output_pos = output_size
                    output_size *= 2
                    output_buf = <uint8_t *>PyMem_Realloc(output_buf, output_size)
                    if output_buf == NULL:
                        raise MemoryError("Failed to reallocate output buffer")
                    strm.next_out = output_buf + output_pos
                    strm.avail_out = output_size - output_pos
            else:
                _raise_lzma_error(ret)

        # Create result bytes
        result = PyBytes_FromStringAndSize(<char *>output_buf, strm.total_out)
        return result

    finally:
        lzma.lzma_end(&strm)
        if output_buf != NULL:
            PyMem_Free(output_buf)


def decompress(data, *, unsigned int threads=0, memlimit=None):
    """
    Decompress LZMA/XZ data using multi-threaded decompression.

    Args:
        data: Compressed bytes-like object.
        threads: Number of threads. 0 = auto-detect CPU count (default).
        memlimit: Memory limit in bytes, or None for no limit.

    Returns:
        Decompressed data as bytes.
    """
    cdef:
        lzma.lzma_stream strm
        lzma.lzma_mt mt_options
        lzma.lzma_ret ret
        const uint8_t *input_data
        Py_ssize_t input_len
        uint8_t *output_buf = NULL
        size_t output_size = INITIAL_BUFFER_SIZE
        size_t output_pos = 0
        bytes input_bytes
        bytes result
        uint64_t mem_limit

    # Convert input to bytes
    if isinstance(data, memoryview):
        input_bytes = bytes(data)
    else:
        input_bytes = bytes(data)

    input_data = <const uint8_t *>PyBytes_AS_STRING(input_bytes)
    input_len = len(input_bytes)

    # Parse memlimit
    if memlimit is None:
        mem_limit = UINT64_MAX
    else:
        mem_limit = <uint64_t>memlimit

    # Initialize stream
    _init_stream(&strm)

    # Set up MT options for decoder
    memset(&mt_options, 0, sizeof(lzma.lzma_mt))
    mt_options.threads = threads if threads > 0 else lzma.lzma_cputhreads()
    mt_options.flags = lzma.LZMA_CONCATENATED  # Handle concatenated streams
    mt_options.memlimit_threading = mem_limit
    mt_options.memlimit_stop = mem_limit

    # Initialize MT decoder
    ret = lzma.lzma_stream_decoder_mt(&strm, &mt_options)
    if ret != lzma.LZMA_OK:
        _raise_lzma_error(ret)

    try:
        # Allocate output buffer
        output_buf = <uint8_t *>PyMem_Malloc(output_size)
        if output_buf == NULL:
            raise MemoryError("Failed to allocate output buffer")

        strm.next_in = input_data
        strm.avail_in = <size_t>input_len
        strm.next_out = output_buf
        strm.avail_out = output_size

        while True:
            ret = lzma.lzma_code(&strm, lzma.LZMA_FINISH)

            if ret == lzma.LZMA_STREAM_END:
                break
            elif ret == lzma.LZMA_OK:
                # Need more output space
                if strm.avail_out == 0:
                    output_pos = output_size
                    output_size *= 2
                    output_buf = <uint8_t *>PyMem_Realloc(output_buf, output_size)
                    if output_buf == NULL:
                        raise MemoryError("Failed to reallocate output buffer")
                    strm.next_out = output_buf + output_pos
                    strm.avail_out = output_size - output_pos
            else:
                _raise_lzma_error(ret)

        # Create result bytes
        result = PyBytes_FromStringAndSize(<char *>output_buf, strm.total_out)
        return result

    finally:
        lzma.lzma_end(&strm)
        if output_buf != NULL:
            PyMem_Free(output_buf)


cdef class LZMACompressor:
    """
    Streaming LZMA compressor with multi-threading support.
    """
    cdef:
        lzma.lzma_stream strm
        bint initialized
        bint flushed

    def __cinit__(self, int preset=6, int check=CHECK_CRC64, *, unsigned int threads=0):
        cdef lzma.lzma_mt mt_options
        cdef lzma.lzma_ret ret

        _init_stream(&self.strm)
        self.initialized = False
        self.flushed = False

        memset(&mt_options, 0, sizeof(lzma.lzma_mt))
        mt_options.threads = threads if threads > 0 else lzma.lzma_cputhreads()
        mt_options.preset = <uint32_t>preset
        mt_options.check = <lzma.lzma_check>check
        mt_options.block_size = 0
        mt_options.timeout = 0

        ret = lzma.lzma_stream_encoder_mt(&self.strm, &mt_options)
        if ret != lzma.LZMA_OK:
            _raise_lzma_error(ret)

        self.initialized = True

    def __dealloc__(self):
        if self.initialized:
            lzma.lzma_end(&self.strm)

    def compress(self, data):
        """
        Compress data and return any available compressed output.
        """
        if self.flushed:
            raise ValueError("Compressor has been flushed")

        cdef:
            const uint8_t *input_data
            Py_ssize_t input_len
            uint8_t *output_buf = NULL
            size_t output_size = INITIAL_BUFFER_SIZE
            size_t output_pos = 0
            lzma.lzma_ret ret
            bytes input_bytes
            bytes result

        if isinstance(data, memoryview):
            input_bytes = bytes(data)
        else:
            input_bytes = bytes(data)

        input_data = <const uint8_t *>PyBytes_AS_STRING(input_bytes)
        input_len = len(input_bytes)

        output_buf = <uint8_t *>PyMem_Malloc(output_size)
        if output_buf == NULL:
            raise MemoryError("Failed to allocate output buffer")

        try:
            self.strm.next_in = input_data
            self.strm.avail_in = <size_t>input_len
            self.strm.next_out = output_buf
            self.strm.avail_out = output_size

            while self.strm.avail_in > 0:
                ret = lzma.lzma_code(&self.strm, lzma.LZMA_RUN)

                if ret != lzma.LZMA_OK:
                    _raise_lzma_error(ret)

                if self.strm.avail_out == 0:
                    output_pos = output_size
                    output_size *= 2
                    output_buf = <uint8_t *>PyMem_Realloc(output_buf, output_size)
                    if output_buf == NULL:
                        raise MemoryError("Failed to reallocate output buffer")
                    self.strm.next_out = output_buf + output_pos
                    self.strm.avail_out = output_size - output_pos

            result = PyBytes_FromStringAndSize(
                <char *>output_buf, output_size - self.strm.avail_out)
            return result

        finally:
            if output_buf != NULL:
                PyMem_Free(output_buf)

    def flush(self):
        """
        Finish compression and return remaining compressed data.
        """
        if self.flushed:
            raise ValueError("Compressor has already been flushed")

        cdef:
            uint8_t *output_buf = NULL
            size_t output_size = INITIAL_BUFFER_SIZE
            size_t output_pos = 0
            lzma.lzma_ret ret
            bytes result

        output_buf = <uint8_t *>PyMem_Malloc(output_size)
        if output_buf == NULL:
            raise MemoryError("Failed to allocate output buffer")

        try:
            self.strm.next_in = NULL
            self.strm.avail_in = 0
            self.strm.next_out = output_buf
            self.strm.avail_out = output_size

            while True:
                ret = lzma.lzma_code(&self.strm, lzma.LZMA_FINISH)

                if ret == lzma.LZMA_STREAM_END:
                    break
                elif ret == lzma.LZMA_OK:
                    if self.strm.avail_out == 0:
                        output_pos = output_size
                        output_size *= 2
                        output_buf = <uint8_t *>PyMem_Realloc(output_buf, output_size)
                        if output_buf == NULL:
                            raise MemoryError("Failed to reallocate output buffer")
                        self.strm.next_out = output_buf + output_pos
                        self.strm.avail_out = output_size - output_pos
                else:
                    _raise_lzma_error(ret)

            self.flushed = True
            result = PyBytes_FromStringAndSize(
                <char *>output_buf, output_size - self.strm.avail_out)
            return result

        finally:
            if output_buf != NULL:
                PyMem_Free(output_buf)


cdef class LZMADecompressor:
    """
    Streaming LZMA decompressor with multi-threading support.
    """
    cdef:
        lzma.lzma_stream strm
        bint initialized
        bint _eof
        bytes _unused_data

    def __cinit__(self, *, unsigned int threads=0, memlimit=None):
        cdef lzma.lzma_mt mt_options
        cdef lzma.lzma_ret ret
        cdef uint64_t mem_limit

        _init_stream(&self.strm)
        self.initialized = False
        self._eof = False
        self._unused_data = b""

        if memlimit is None:
            mem_limit = UINT64_MAX
        else:
            mem_limit = <uint64_t>memlimit

        memset(&mt_options, 0, sizeof(lzma.lzma_mt))
        mt_options.threads = threads if threads > 0 else lzma.lzma_cputhreads()
        mt_options.flags = lzma.LZMA_CONCATENATED
        mt_options.memlimit_threading = mem_limit
        mt_options.memlimit_stop = mem_limit

        ret = lzma.lzma_stream_decoder_mt(&self.strm, &mt_options)
        if ret != lzma.LZMA_OK:
            _raise_lzma_error(ret)

        self.initialized = True

    def __dealloc__(self):
        if self.initialized:
            lzma.lzma_end(&self.strm)

    @property
    def eof(self):
        """True if end of stream has been reached."""
        return self._eof

    @property
    def unused_data(self):
        """Data found after the end of the compressed stream."""
        return self._unused_data

    def decompress(self, data, max_length=-1):
        """
        Decompress data and return decompressed output.

        Args:
            data: Compressed data to decompress.
            max_length: Maximum bytes to return (-1 for unlimited).

        Returns:
            Decompressed bytes.
        """
        if self._eof:
            raise ValueError("Decompressor has reached end of stream")

        cdef:
            const uint8_t *input_data
            Py_ssize_t input_len
            uint8_t *output_buf = NULL
            size_t output_size
            size_t output_pos = 0
            lzma.lzma_ret ret
            bytes input_bytes
            bytes result

        if isinstance(data, memoryview):
            input_bytes = bytes(data)
        else:
            input_bytes = bytes(data)

        input_data = <const uint8_t *>PyBytes_AS_STRING(input_bytes)
        input_len = len(input_bytes)

        if max_length < 0:
            output_size = INITIAL_BUFFER_SIZE
        else:
            output_size = <size_t>max_length

        output_buf = <uint8_t *>PyMem_Malloc(output_size)
        if output_buf == NULL:
            raise MemoryError("Failed to allocate output buffer")

        try:
            self.strm.next_in = input_data
            self.strm.avail_in = <size_t>input_len
            self.strm.next_out = output_buf
            self.strm.avail_out = output_size

            while True:
                ret = lzma.lzma_code(&self.strm, lzma.LZMA_RUN)

                if ret == lzma.LZMA_STREAM_END:
                    self._eof = True
                    # Save unused data
                    if self.strm.avail_in > 0:
                        self._unused_data = input_bytes[input_len - self.strm.avail_in:]
                    break
                elif ret == lzma.LZMA_OK:
                    if self.strm.avail_in == 0:
                        break  # Need more input
                    if self.strm.avail_out == 0:
                        if max_length >= 0:
                            break  # Hit max_length limit
                        # Grow buffer
                        output_pos = output_size
                        output_size *= 2
                        output_buf = <uint8_t *>PyMem_Realloc(output_buf, output_size)
                        if output_buf == NULL:
                            raise MemoryError("Failed to reallocate output buffer")
                        self.strm.next_out = output_buf + output_pos
                        self.strm.avail_out = output_size - output_pos
                else:
                    _raise_lzma_error(ret)

            result = PyBytes_FromStringAndSize(
                <char *>output_buf, output_size - self.strm.avail_out)
            return result

        finally:
            if output_buf != NULL:
                PyMem_Free(output_buf)
