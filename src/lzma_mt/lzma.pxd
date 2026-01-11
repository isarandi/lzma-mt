# cython: language_level=3
# Cython declarations for liblzma

from libc.stdint cimport uint8_t, uint32_t, uint64_t, UINT64_MAX
from libc.stddef cimport size_t

cdef extern from "lzma.h":
    # Return codes
    ctypedef enum lzma_ret:
        LZMA_OK
        LZMA_STREAM_END
        LZMA_NO_CHECK
        LZMA_UNSUPPORTED_CHECK
        LZMA_GET_CHECK
        LZMA_MEM_ERROR
        LZMA_MEMLIMIT_ERROR
        LZMA_FORMAT_ERROR
        LZMA_OPTIONS_ERROR
        LZMA_DATA_ERROR
        LZMA_BUF_ERROR
        LZMA_PROG_ERROR

    # Actions for lzma_code()
    ctypedef enum lzma_action:
        LZMA_RUN
        LZMA_SYNC_FLUSH
        LZMA_FULL_FLUSH
        LZMA_FULL_BARRIER
        LZMA_FINISH

    # Integrity check types
    ctypedef enum lzma_check:
        LZMA_CHECK_NONE
        LZMA_CHECK_CRC32
        LZMA_CHECK_CRC64
        LZMA_CHECK_SHA256

    # Reserved enum placeholder
    ctypedef enum lzma_reserved_enum:
        LZMA_RESERVED_ENUM

    # Filter struct (opaque for now)
    ctypedef struct lzma_filter:
        uint64_t id
        void *options

    # Internal state (opaque)
    ctypedef struct lzma_internal:
        pass

    # Allocator (opaque)
    ctypedef struct lzma_allocator:
        pass

    # Main stream struct
    ctypedef struct lzma_stream:
        const uint8_t *next_in
        size_t avail_in
        uint64_t total_in
        uint8_t *next_out
        size_t avail_out
        uint64_t total_out
        const lzma_allocator *allocator
        lzma_internal *internal
        void *reserved_ptr1
        void *reserved_ptr2
        void *reserved_ptr3
        void *reserved_ptr4
        uint64_t seek_pos
        uint64_t reserved_int2
        size_t reserved_int3
        size_t reserved_int4
        lzma_reserved_enum reserved_enum1
        lzma_reserved_enum reserved_enum2

    # Multi-threading options
    ctypedef struct lzma_mt:
        uint32_t flags
        uint32_t threads
        uint64_t block_size
        uint32_t timeout
        uint32_t preset
        const lzma_filter *filters
        lzma_check check
        lzma_reserved_enum reserved_enum1
        lzma_reserved_enum reserved_enum2
        lzma_reserved_enum reserved_enum3
        uint32_t reserved_int1
        uint32_t reserved_int2
        uint32_t reserved_int3
        uint32_t reserved_int4
        uint64_t memlimit_threading
        uint64_t memlimit_stop
        uint64_t reserved_int7
        uint64_t reserved_int8
        void *reserved_ptr1
        void *reserved_ptr2
        void *reserved_ptr3
        void *reserved_ptr4

    # Presets
    uint32_t LZMA_PRESET_DEFAULT
    uint32_t LZMA_PRESET_EXTREME

    # Decoder flags
    uint32_t LZMA_TELL_NO_CHECK
    uint32_t LZMA_TELL_UNSUPPORTED_CHECK
    uint32_t LZMA_TELL_ANY_CHECK
    uint32_t LZMA_IGNORE_CHECK
    uint32_t LZMA_CONCATENATED

    # Core functions
    lzma_ret lzma_code(lzma_stream *strm, lzma_action action) nogil
    void lzma_end(lzma_stream *strm) nogil

    # MT encoder
    lzma_ret lzma_stream_encoder_mt(lzma_stream *strm, const lzma_mt *options) nogil
    uint64_t lzma_stream_encoder_mt_memusage(const lzma_mt *options) nogil

    # MT decoder
    lzma_ret lzma_stream_decoder_mt(lzma_stream *strm, const lzma_mt *options) nogil

    # Single-threaded alternatives (for fallback/comparison)
    lzma_ret lzma_easy_encoder(lzma_stream *strm, uint32_t preset, lzma_check check) nogil
    lzma_ret lzma_stream_decoder(lzma_stream *strm, uint64_t memlimit, uint32_t flags) nogil
    lzma_ret lzma_auto_decoder(lzma_stream *strm, uint64_t memlimit, uint32_t flags) nogil

    # Utility
    uint64_t lzma_easy_encoder_memusage(uint32_t preset) nogil
    uint64_t lzma_easy_decoder_memusage(uint32_t preset) nogil
    uint64_t lzma_physmem() nogil
    uint32_t lzma_cputhreads() nogil
