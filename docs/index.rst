lzma_mt
=======

Multi-threaded LZMA/XZ compression via Cython wrapper around liblzma.

A drop-in replacement for Python's :mod:`lzma` module with multi-threading support.

Features
--------

- Full API compatibility with Python's :mod:`lzma` module
- Multi-threaded compression and decompression
- Zero-copy buffer handling for performance
- GIL-free operation during compression/decompression
- Thread-safe streaming classes

Installation
------------

.. code-block:: bash

    pip install lzma_mt

Quick Start
-----------

One-shot compression/decompression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import lzma_mt

    # Compress with multiple threads
    data = b"Hello World!" * 1000
    compressed = lzma_mt.compress(data, threads=4)

    # Decompress
    decompressed = lzma_mt.decompress(compressed, threads=4)
    assert decompressed == data

Streaming compression
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import lzma_mt

    # Streaming compression
    compressor = lzma_mt.LZMACompressor(threads=4)
    compressed = compressor.compress(b"Hello ")
    compressed += compressor.compress(b"World!")
    compressed += compressor.flush()

    # Streaming decompression
    decompressor = lzma_mt.LZMADecompressor(threads=4)
    result = decompressor.decompress(compressed)

File operations
~~~~~~~~~~~~~~~

.. code-block:: python

    import lzma_mt

    # Write compressed file
    with lzma_mt.open("data.xz", "wb", threads=4) as f:
        f.write(b"Hello World!")

    # Read compressed file
    with lzma_mt.open("data.xz", "rb", threads=4) as f:
        data = f.read()

Security Note
-------------

The multi-threaded decoder in xz-utils 5.3.3alpha-5.8.0 has CVE-2025-31115
(use-after-free). This module checks the version at runtime and raises
``RuntimeError`` if vulnerable. Use :func:`~lzma_mt.is_mt_decoder_safe` to check,
or pass ``threads=1`` to use single-threaded mode.

API Reference
-------------

See the full API documentation at :mod:`lzma_mt`.

.. toctree::
   :maxdepth: 3
   :caption: Contents
   :hidden:

* :ref:`genindex`
