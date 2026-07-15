"""Tests for LZMAFile and open().

Covers file round-trips (paths, path-like objects, file objects), text mode,
append mode, seeking, stdlib interoperability, and error conditions.
"""

import io
import lzma
import pathlib

import pytest

import lzma_mt


DATA = b"The rain in Spain stays mainly in the plain. " * 500


class TestRoundtrip:
    def test_str_path(self, tmp_path):
        path = str(tmp_path / "file.xz")
        with lzma_mt.LZMAFile(path, "w") as f:
            f.write(DATA)
        with lzma_mt.LZMAFile(path) as f:
            assert f.read() == DATA

    def test_pathlike(self, tmp_path):
        path = tmp_path / "file.xz"
        assert isinstance(path, pathlib.Path)
        with lzma_mt.LZMAFile(path, "w") as f:
            f.write(DATA)
        with lzma_mt.LZMAFile(path) as f:
            assert f.read() == DATA

    def test_bytes_path(self, tmp_path):
        path = str(tmp_path / "file.xz").encode()
        with lzma_mt.LZMAFile(path, "w") as f:
            f.write(DATA)
        with lzma_mt.LZMAFile(path) as f:
            assert f.read() == DATA

    def test_file_object(self):
        bio = io.BytesIO()
        with lzma_mt.LZMAFile(bio, "w") as f:
            f.write(DATA)
        bio.seek(0)
        with lzma_mt.LZMAFile(bio) as f:
            assert f.read() == DATA

    @pytest.mark.parametrize("threads", [0, 1, 4])
    def test_threads(self, tmp_path, threads):
        path = tmp_path / "file.xz"
        with lzma_mt.open(path, "wb", threads=threads) as f:
            f.write(DATA)
        with lzma_mt.open(path, "rb", threads=threads) as f:
            assert f.read() == DATA

    def test_write_memoryview(self):
        bio = io.BytesIO()
        with lzma_mt.LZMAFile(bio, "w") as f:
            assert f.write(memoryview(DATA)) == len(DATA)
        assert lzma_mt.decompress(bio.getvalue()) == DATA

    def test_large_multiblock_mt(self, tmp_path):
        """MT-compressed multi-block file reads back correctly with MT."""
        data = b"multiblock " * 2_000_000  # ~22 MB, splits into blocks
        path = tmp_path / "big.xz"
        with lzma_mt.open(path, "wb", threads=4, preset=1) as f:
            f.write(data)
        with lzma_mt.open(path, "rb", threads=4) as f:
            assert f.read() == data


class TestModes:
    def test_append_creates_multistream(self, tmp_path):
        path = tmp_path / "file.xz"
        with lzma_mt.LZMAFile(path, "w") as f:
            f.write(b"first")
        with lzma_mt.LZMAFile(path, "a") as f:
            f.write(b"second")
        with lzma_mt.LZMAFile(path) as f:
            assert f.read() == b"firstsecond"

    def test_exclusive_mode(self, tmp_path):
        path = tmp_path / "file.xz"
        with lzma_mt.LZMAFile(path, "x") as f:
            f.write(DATA)
        with pytest.raises(FileExistsError):
            lzma_mt.LZMAFile(path, "x")

    def test_invalid_mode(self, tmp_path):
        with pytest.raises(ValueError):
            lzma_mt.LZMAFile(tmp_path / "f.xz", "q")

    def test_text_mode(self, tmp_path):
        path = tmp_path / "file.xz"
        text = "Hello, 世界!\nSecond line.\n"
        with lzma_mt.open(path, "wt", encoding="utf-8") as f:
            f.write(text)
        with lzma_mt.open(path, "rt", encoding="utf-8") as f:
            assert f.read() == text

    def test_text_binary_mode_conflicts(self, tmp_path):
        with pytest.raises(ValueError):
            lzma_mt.open(tmp_path / "f.xz", "rbt")
        with pytest.raises(ValueError):
            lzma_mt.open(tmp_path / "f.xz", "rb", encoding="utf-8")

    def test_read_mode_rejects_writer_args(self):
        with pytest.raises(ValueError):
            lzma_mt.LZMAFile(io.BytesIO(), "r", check=lzma_mt.CHECK_CRC64)
        with pytest.raises(ValueError):
            lzma_mt.LZMAFile(io.BytesIO(), "r", preset=6)

    def test_invalid_filename_type(self):
        with pytest.raises(TypeError):
            lzma_mt.LZMAFile(123, "r")


class TestStdlibInterop:
    def test_read_stdlib_written_file(self, tmp_path):
        path = tmp_path / "file.xz"
        with lzma.open(path, "wb") as f:
            f.write(DATA)
        for threads in (1, 4):
            with lzma_mt.open(path, "rb", threads=threads) as f:
                assert f.read() == DATA

    def test_stdlib_reads_our_file(self, tmp_path):
        path = tmp_path / "file.xz"
        with lzma_mt.open(path, "wb", threads=4) as f:
            f.write(DATA)
        with lzma.open(path, "rb") as f:
            assert f.read() == DATA

    @pytest.mark.parametrize("threads", [1, 4])
    def test_read_alone_format_file(self, tmp_path, threads):
        """Default FORMAT_AUTO must read legacy .lzma files."""
        path = tmp_path / "file.lzma"
        with lzma.open(path, "wb", format=lzma.FORMAT_ALONE) as f:
            f.write(DATA)
        with lzma_mt.open(path, "rb", threads=threads) as f:
            assert f.read() == DATA


class TestReading:
    @pytest.fixture
    def xz_file(self, tmp_path):
        path = tmp_path / "file.xz"
        with lzma_mt.LZMAFile(path, "w") as f:
            f.write(DATA)
        return path

    def test_read_chunks(self, xz_file):
        with lzma_mt.LZMAFile(xz_file) as f:
            result = b""
            while chunk := f.read(1000):
                result += chunk
            assert result == DATA

    def test_read1(self, xz_file):
        with lzma_mt.LZMAFile(xz_file) as f:
            result = b""
            while chunk := f.read1():
                result += chunk
            assert result == DATA

    def test_peek(self, xz_file):
        with lzma_mt.LZMAFile(xz_file) as f:
            peeked = f.peek()
            assert len(peeked) > 0
            assert f.read(len(peeked)) == peeked

    def test_readline_and_iteration(self, tmp_path):
        path = tmp_path / "lines.xz"
        lines = [b"line %d\n" % i for i in range(100)]
        with lzma_mt.LZMAFile(path, "w") as f:
            f.write(b"".join(lines))
        with lzma_mt.LZMAFile(path) as f:
            assert f.readline() == lines[0]
            assert list(f) == lines[1:]

    def test_seek_and_tell(self, xz_file):
        with lzma_mt.LZMAFile(xz_file) as f:
            assert f.seekable()
            f.seek(100)
            assert f.tell() == 100
            assert f.read(10) == DATA[100:110]
            f.seek(-10, io.SEEK_END)
            assert f.read() == DATA[-10:]
            f.seek(0)
            assert f.read() == DATA

    def test_truncated_file_raises_eoferror(self, xz_file):
        truncated = xz_file.read_bytes()[:-20]
        with lzma_mt.LZMAFile(io.BytesIO(truncated)) as f:
            with pytest.raises(EOFError):
                f.read()


class TestClosedFile:
    def test_operations_on_closed_file(self, tmp_path):
        path = tmp_path / "file.xz"
        f = lzma_mt.LZMAFile(path, "w")
        f.write(DATA)
        f.close()
        assert f.closed
        f.close()  # Second close is a no-op
        with pytest.raises(ValueError):
            f.write(b"more")
        with pytest.raises(ValueError):
            f.fileno()

        f = lzma_mt.LZMAFile(path)
        f.close()
        with pytest.raises(ValueError):
            f.read()

    def test_close_does_not_close_external_fp(self):
        bio = io.BytesIO()
        f = lzma_mt.LZMAFile(bio, "w")
        f.write(DATA)
        f.close()
        assert not bio.closed  # Caller-owned file object stays open
