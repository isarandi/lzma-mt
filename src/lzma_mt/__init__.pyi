"""Type stubs for lzma_mt."""

from typing import IO, Mapping, Sequence, Union
from os import PathLike

# Constants (re-exported from lzma)
CHECK_NONE: int
CHECK_CRC32: int
CHECK_CRC64: int
CHECK_SHA256: int
CHECK_ID_MAX: int
CHECK_UNKNOWN: int

PRESET_DEFAULT: int
PRESET_EXTREME: int

FORMAT_AUTO: int
FORMAT_XZ: int
FORMAT_ALONE: int
FORMAT_RAW: int

FILTER_LZMA1: int
FILTER_LZMA2: int
FILTER_DELTA: int
FILTER_X86: int
FILTER_POWERPC: int
FILTER_IA64: int
FILTER_ARM: int
FILTER_ARMTHUMB: int
FILTER_SPARC: int

MF_HC3: int
MF_HC4: int
MF_BT2: int
MF_BT3: int
MF_BT4: int

MODE_FAST: int
MODE_NORMAL: int

__version__: str

_FilterChain = Sequence[Mapping[str, object]]

class LZMAError(Exception):
    """Exception raised for LZMA-related errors."""
    ...

class LZMACompressor:
    """Create a compressor object for compressing data incrementally.

    format specifies the container format to use for the output.
    check specifies the integrity check to use (ignored for FORMAT_ALONE/RAW).
    preset sets compression level (0-9, optionally OR-ed with PRESET_EXTREME).
    filters specifies a custom filter chain (overrides preset if given).
    threads specifies number of threads (0=auto, 1=single-threaded).
    """
    def __init__(
        self,
        format: int = ...,
        check: int = ...,
        preset: int | None = ...,
        filters: _FilterChain | None = ...,
        *,
        threads: int = ...
    ) -> None: ...
    def compress(self, data: bytes) -> bytes:
        """Compress data and return a bytes object with compressed data."""
        ...
    def flush(self) -> bytes:
        """Finish the compression process and return remaining compressed data."""
        ...

class LZMADecompressor:
    """Create a decompressor object for decompressing data incrementally.

    format specifies the container format of the input.
    memlimit limits memory usage; LZMAError is raised if exceeded.
    filters specifies a custom filter chain (required for FORMAT_RAW).
    threads specifies number of threads (0=auto, 1=single-threaded).
    """
    def __init__(
        self,
        format: int = ...,
        memlimit: int | None = ...,
        filters: _FilterChain | None = ...,
        *,
        threads: int = ...
    ) -> None: ...
    def decompress(self, data: bytes, max_length: int = ...) -> bytes:
        """Decompress data and return a bytes object with decompressed data."""
        ...
    @property
    def check(self) -> int:
        """ID of the integrity check used by the input stream."""
        ...
    @property
    def eof(self) -> bool:
        """True if the end-of-stream marker has been reached."""
        ...
    @property
    def unused_data(self) -> bytes:
        """Data found after the end of the compressed stream."""
        ...
    @property
    def needs_input(self) -> bool:
        """True if more input is needed before more output can be produced."""
        ...

class LZMAFile(IO[bytes]):
    """A file object providing transparent LZMA (de)compression with MT support."""
    def __init__(
        self,
        filename: str | bytes | PathLike[str] | PathLike[bytes] | IO[bytes] | None = ...,
        mode: str = ...,
        *,
        format: int | None = ...,
        check: int = ...,
        preset: int | None = ...,
        filters: _FilterChain | None = ...,
        threads: int = ...
    ) -> None: ...
    def close(self) -> None: ...
    @property
    def closed(self) -> bool: ...
    def fileno(self) -> int: ...
    def seekable(self) -> bool: ...
    def readable(self) -> bool: ...
    def writable(self) -> bool: ...
    def peek(self, size: int = ...) -> bytes: ...
    def read(self, size: int = ...) -> bytes: ...
    def read1(self, size: int = ...) -> bytes: ...
    def readline(self, size: int = ...) -> bytes: ...
    def write(self, data: bytes) -> int: ...
    def seek(self, offset: int, whence: int = ...) -> int: ...
    def tell(self) -> int: ...
    def __enter__(self) -> LZMAFile: ...
    def __exit__(self, *args: object) -> None: ...
    def __iter__(self) -> LZMAFile: ...
    def __next__(self) -> bytes: ...

def compress(
    data: bytes,
    format: int = ...,
    check: int = ...,
    preset: int | None = ...,
    filters: _FilterChain | None = ...,
    *,
    threads: int = ...
) -> bytes:
    """Compress data and return it as a bytes object.

    format specifies the container format (default FORMAT_XZ).
    check specifies integrity check (default CHECK_CRC64 for XZ).
    preset sets compression level (0-9, optionally OR-ed with PRESET_EXTREME).
    filters specifies a custom filter chain (overrides preset).
    threads specifies number of threads (0=auto, 1=single-threaded default).
    """
    ...

def decompress(
    data: bytes,
    format: int = ...,
    memlimit: int | None = ...,
    filters: _FilterChain | None = ...,
    *,
    threads: int = ...
) -> bytes:
    """Decompress data and return it as a bytes object.

    format specifies the container format (default FORMAT_AUTO).
    memlimit limits memory usage; LZMAError raised if exceeded.
    filters specifies a custom filter chain (required for FORMAT_RAW).
    threads specifies number of threads (0=auto, 1=single-threaded default).
    """
    ...

def open(
    filename: str | bytes | PathLike[str] | PathLike[bytes] | IO[bytes],
    mode: str = ...,
    *,
    format: int | None = ...,
    check: int = ...,
    preset: int | None = ...,
    filters: _FilterChain | None = ...,
    encoding: str | None = ...,
    errors: str | None = ...,
    newline: str | None = ...,
    threads: int = ...
) -> LZMAFile | IO[str]:
    """Open an LZMA-compressed file in binary or text mode.

    filename can be a file path or an existing file object.
    mode can be "rb", "wb", "xb", "ab" for binary or "rt", "wt", "xt", "at" for text.
    threads specifies number of threads (0=auto, 1=single-threaded default).
    """
    ...

def is_check_supported(check: int) -> bool:
    """Return True if the given integrity check is supported."""
    ...

def get_xz_version() -> str:
    """Return the version string of the underlying xz-utils library."""
    ...

def is_mt_decoder_safe() -> bool:
    """Return True if multi-threaded decoding is safe (not affected by CVE-2025-31115)."""
    ...
