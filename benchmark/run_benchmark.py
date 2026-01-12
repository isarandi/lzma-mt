"""Benchmark lzma_mt against stdlib lzma and xz binary.

Run with: python benchmark/run_benchmark.py
         python benchmark/run_benchmark.py --large
         python benchmark/run_benchmark.py --wiki /tmp/enwik8
"""

import argparse
import lzma
import os
import sqlite3
import subprocess
import tempfile
import time
from pathlib import Path

import lzma_mt

# Size presets
SIZES_SMALL = [1024, 10 * 1024, 100 * 1024, 1024 * 1024, 10 * 1024 * 1024, 50 * 1024 * 1024]
SIZES_LARGE = [
    100 * 1024 * 1024,      # 100 MB
    500 * 1024 * 1024,      # 500 MB
    1024 * 1024 * 1024,     # 1 GB
    2 * 1024 * 1024 * 1024, # 2 GB
    5 * 1024 * 1024 * 1024, # 5 GB
    10 * 1024 * 1024 * 1024, # 10 GB
]


def main():
    parser = argparse.ArgumentParser(description='Benchmark lzma_mt compression')
    parser.add_argument('--db', default='results.db', help='SQLite database file')
    parser.add_argument(
        '--implementations',
        nargs='+',
        choices=['lzma_mt', 'stdlib', 'xz'],
        default=['lzma_mt', 'stdlib'],
    )
    parser.add_argument(
        '--operations',
        nargs='+',
        choices=['compress', 'decompress'],
        default=['compress', 'decompress'],
    )
    parser.add_argument('--large', action='store_true', help='Run large-scale benchmarks (100MB-10GB)')
    parser.add_argument('--wiki', type=str, help='Path to Wikipedia dump file (e.g., enwik8) for natural text')
    parser.add_argument('--sizes', nargs='+', type=str, help='Custom sizes (e.g., 100M 1G 5G)')
    parser.add_argument('--preset', type=int, default=6, help='Compression preset 0-9 (default: 6, use 1-3 for faster benchmarks)')
    args = parser.parse_args()

    out_dir = Path(__file__).parent
    conn = init_db(out_dir / args.db)

    # Determine data sizes
    if args.sizes:
        data_sizes = [parse_size(s) for s in args.sizes]
    elif args.large:
        data_sizes = SIZES_LARGE
    else:
        data_sizes = SIZES_SMALL

    thread_counts = [1, 2, 4, 8, 0]  # 0 = auto

    # Load Wikipedia data if provided
    wiki_data = None
    if args.wiki:
        wiki_path = Path(args.wiki)
        if wiki_path.exists():
            print(f'Loading Wikipedia data from {wiki_path}...')
            wiki_data = wiki_path.read_bytes()
            print(f'Loaded {len(wiki_data) / 1024 / 1024:.1f} MB of Wikipedia text')
        else:
            print(f'Warning: Wiki file {wiki_path} not found, using synthetic data')

    for operation in args.operations:
        print(f'\n{"=" * 60}')
        print(f'Benchmarking: {operation}')
        print(f'{"=" * 60}')

        results = run_benchmarks(
            operation=operation,
            data_sizes=data_sizes,
            thread_counts=thread_counts,
            implementations=args.implementations,
            wiki_data=wiki_data,
            preset=args.preset,
        )

        # Print table
        print_results_table(results, thread_counts, args.implementations)

        # Insert into DB
        for r in results:
            for impl in args.implementations:
                for threads in thread_counts:
                    key = f'{impl}_t{threads}'
                    if key in r:
                        conn.execute(
                            '''INSERT OR REPLACE INTO results
                               (implementation, operation, threads, data_size_kb, time_ms, throughput_mbs)
                               VALUES (?, ?, ?, ?, ?, ?)''',
                            (
                                impl,
                                operation,
                                threads,
                                r['data_size'] // 1024,
                                r[key],
                                r['data_size'] / 1024 / 1024 / (r[key] / 1000) if r[key] > 0 else 0,
                            ),
                        )
        conn.commit()

    conn.close()
    print(f'\nSaved results to {out_dir / args.db}')


def parse_size(s):
    """Parse size string like '100M', '1G', '500K' to bytes."""
    s = s.upper().strip()
    multipliers = {'K': 1024, 'M': 1024**2, 'G': 1024**3}
    if s[-1] in multipliers:
        return int(float(s[:-1]) * multipliers[s[-1]])
    return int(s)


def run_benchmarks(operation, data_sizes, thread_counts, implementations, wiki_data=None, preset=6):
    """Run benchmarks for all configurations."""
    results = []

    for data_size in data_sizes:
        # Generate test data
        data = generate_test_data(data_size, wiki_data)

        # For large data, compress with MT to create multiple blocks (better for MT decompression)
        compressed_data = lzma_mt.compress(data, threads=0, preset=preset)
        ratio = len(data) / len(compressed_data)

        size_str = format_size(data_size)
        comp_str = format_size(len(compressed_data))
        print(f'\nData size: {size_str} (compressed: {comp_str}, ratio: {ratio:.1f}x)')

        timings = {'data_size': data_size}

        # Adjust iterations based on data size
        if data_size >= 1024 * 1024 * 1024:  # >= 1 GB
            n_iter = 1
        elif data_size >= 100 * 1024 * 1024:  # >= 100 MB
            n_iter = 2
        else:
            n_iter = max(3, min(20, 100 * 1024 * 1024 // data_size))

        for impl in implementations:
            for threads in thread_counts:
                key = f'{impl}_t{threads}'

                if impl == 'stdlib' and threads != 1:
                    # stdlib doesn't support threading, skip non-1 thread counts
                    continue

                if operation == 'compress':
                    timing = benchmark_compress(impl, data, threads, n_iter, preset)
                else:
                    timing = benchmark_decompress(impl, compressed_data, threads, n_iter)

                timings[key] = timing

                # Print progress for large data
                if data_size >= 100 * 1024 * 1024:
                    speed = data_size / 1024 / 1024 / (timing / 1000)
                    t_label = 'auto' if threads == 0 else threads
                    print(f'  {impl} t={t_label}: {timing:.0f} ms ({speed:.1f} MB/s)')

        results.append(timings)

        # Free memory for large data
        del data, compressed_data

    return results


def format_size(size_bytes):
    """Format size in human-readable form."""
    if size_bytes >= 1024**3:
        return f'{size_bytes / 1024**3:.1f} GB'
    elif size_bytes >= 1024**2:
        return f'{size_bytes / 1024**2:.1f} MB'
    elif size_bytes >= 1024:
        return f'{size_bytes / 1024:.1f} KB'
    return f'{size_bytes} B'


def generate_test_data(size, wiki_data=None):
    """Generate test data, optionally based on Wikipedia text."""
    if wiki_data is not None:
        # Repeat Wikipedia data to reach target size
        if len(wiki_data) >= size:
            return wiki_data[:size]
        repeats = size // len(wiki_data) + 1
        return (wiki_data * repeats)[:size]

    # Fallback: mix of patterns and random data for realistic compression ratios
    pattern = b'Hello world! This is a test of LZMA compression. ' * 100
    random_data = os.urandom(min(500, size // 10))
    chunk = pattern + random_data
    repeats = size // len(chunk) + 1
    return (chunk * repeats)[:size]


def benchmark_compress(impl, data, threads, n_iter, preset=6):
    """Benchmark compression."""
    if impl == 'lzma_mt':
        # Warmup
        lzma_mt.compress(data, threads=threads, preset=preset)

        start = time.perf_counter()
        for _ in range(n_iter):
            lzma_mt.compress(data, threads=threads, preset=preset)
        elapsed = time.perf_counter() - start

    elif impl == 'stdlib':
        # Warmup
        lzma.compress(data, preset=preset)

        start = time.perf_counter()
        for _ in range(n_iter):
            lzma.compress(data, preset=preset)
        elapsed = time.perf_counter() - start

    elif impl == 'xz':
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(data)
            input_file = f.name

        try:
            # Warmup
            subprocess.run(
                ['xz', '-c', f'-{preset}', '-T', str(threads), input_file],
                capture_output=True,
                check=True,
            )

            start = time.perf_counter()
            for _ in range(n_iter):
                subprocess.run(
                    ['xz', '-c', f'-{preset}', '-T', str(threads), input_file],
                    capture_output=True,
                    check=True,
                )
            elapsed = time.perf_counter() - start
        finally:
            os.unlink(input_file)

    return elapsed / n_iter * 1000  # Convert to ms


def benchmark_decompress(impl, compressed_data, threads, n_iter):
    """Benchmark decompression."""
    if impl == 'lzma_mt':
        # Warmup
        lzma_mt.decompress(compressed_data, threads=threads)

        start = time.perf_counter()
        for _ in range(n_iter):
            lzma_mt.decompress(compressed_data, threads=threads)
        elapsed = time.perf_counter() - start

    elif impl == 'stdlib':
        # Warmup
        lzma.decompress(compressed_data)

        start = time.perf_counter()
        for _ in range(n_iter):
            lzma.decompress(compressed_data)
        elapsed = time.perf_counter() - start

    elif impl == 'xz':
        with tempfile.NamedTemporaryFile(suffix='.xz', delete=False) as f:
            f.write(compressed_data)
            input_file = f.name

        try:
            # Warmup
            subprocess.run(
                ['xz', '-dc', '-T', str(threads), input_file],
                capture_output=True,
                check=True,
            )

            start = time.perf_counter()
            for _ in range(n_iter):
                subprocess.run(
                    ['xz', '-dc', '-T', str(threads), input_file],
                    capture_output=True,
                    check=True,
                )
            elapsed = time.perf_counter() - start
        finally:
            os.unlink(input_file)

    return elapsed / n_iter * 1000  # Convert to ms


def print_results_table(results, thread_counts, implementations):
    """Print results as a formatted table."""
    # Build header
    headers = ['Size (KB)']
    for impl in implementations:
        for threads in thread_counts:
            if impl == 'stdlib' and threads != 1:
                continue
            t_label = 'auto' if threads == 0 else threads
            headers.append(f'{impl[:6]} t={t_label}')

    print('\n' + ' | '.join(f'{h:>12}' for h in headers))
    print('-' * (14 * len(headers)))

    for r in results:
        row = [f"{r['data_size'] // 1024:>12}"]
        for impl in implementations:
            for threads in thread_counts:
                if impl == 'stdlib' and threads != 1:
                    continue
                key = f'{impl}_t{threads}'
                if key in r:
                    row.append(f'{r[key]:>12.2f}')
                else:
                    row.append(f'{"N/A":>12}')
        print(' | '.join(row))


def init_db(db_path):
    """Initialize SQLite database."""
    conn = sqlite3.connect(db_path)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS results (
            implementation TEXT,
            operation TEXT,
            threads INTEGER,
            data_size_kb INTEGER,
            time_ms REAL,
            throughput_mbs REAL,
            PRIMARY KEY (implementation, operation, threads, data_size_kb)
        )
    ''')
    conn.commit()
    return conn


if __name__ == '__main__':
    main()
