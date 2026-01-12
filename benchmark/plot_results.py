"""Generate benchmark plots from results.db."""

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt


# Style configuration: color by implementation, linestyle by thread count
IMPL_COLORS = {
    'lzma_mt': 'C0',
    'stdlib': 'C1',
    'xz': 'C2',
}

THREAD_STYLES = {
    1: {'linestyle': '-', 'marker': 'o'},
    2: {'linestyle': '--', 'marker': 's'},
    4: {'linestyle': '-.', 'marker': '^'},
    8: {'linestyle': ':', 'marker': 'D'},
    0: {'linestyle': '-', 'marker': 'v', 'linewidth': 2.5},  # auto, make it stand out
}


def query_results(conn, implementation, operation, threads):
    """Query throughput results for a specific configuration."""
    cursor = conn.execute(
        '''SELECT data_size_kb, throughput_mbs
           FROM results
           WHERE implementation = ? AND operation = ? AND threads = ?
           ORDER BY data_size_kb''',
        (implementation, operation, threads),
    )
    rows = cursor.fetchall()
    if not rows:
        return [], []
    sizes, throughputs = zip(*rows)
    return list(sizes), list(throughputs)


def plot_throughput():
    """Plot throughput (MB/s) vs data size for all configurations."""
    out_dir = Path(__file__).parent
    conn = sqlite3.connect(out_dir / 'results.db')

    # Get available data from DB
    implementations = [r[0] for r in conn.execute('SELECT DISTINCT implementation FROM results')]
    operations = [r[0] for r in conn.execute('SELECT DISTINCT operation FROM results')]
    threads_available = [r[0] for r in conn.execute('SELECT DISTINCT threads FROM results')]

    fig, axes = plt.subplots(1, len(operations), figsize=(6 * len(operations), 5))
    if len(operations) == 1:
        axes = [axes]

    for ax, operation in zip(axes, operations):
        for impl in implementations:
            color = IMPL_COLORS.get(impl, 'C3')

            for threads in threads_available:
                # Skip non-1 threads for stdlib (doesn't support MT)
                if impl == 'stdlib' and threads != 1:
                    continue

                sizes, throughputs = query_results(conn, impl, operation, threads)
                if not sizes:
                    continue

                style = THREAD_STYLES.get(threads, {'linestyle': '-', 'marker': 'o'})
                t_label = 'auto' if threads == 0 else threads
                label = f'{impl} (t={t_label})'

                ax.plot(
                    sizes,
                    throughputs,
                    color=color,
                    label=label,
                    markersize=6,
                    **style,
                )

        ax.set_xlabel('Data Size (KB)')
        ax.set_ylabel('Throughput (MB/s)')
        ax.set_title(f'{operation.capitalize()}')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / 'benchmark_throughput.png', dpi=150)
    print(f'Saved {out_dir / "benchmark_throughput.png"}')
    conn.close()


def plot_comparison():
    """Plot lzma_mt vs stdlib and xz binary for key thread counts."""
    out_dir = Path(__file__).parent
    conn = sqlite3.connect(out_dir / 'results.db')

    operations = [r[0] for r in conn.execute('SELECT DISTINCT operation FROM results')]

    fig, axes = plt.subplots(1, len(operations), figsize=(6 * len(operations), 5))
    if len(operations) == 1:
        axes = [axes]

    # Focus on: lzma_mt auto vs stdlib vs xz auto
    comparisons = [
        ('lzma_mt', 0, 'lzma_mt (auto)'),
        ('lzma_mt', 1, 'lzma_mt (t=1)'),
        ('stdlib', 1, 'stdlib'),
        ('xz', 0, 'xz (auto)'),
        ('xz', 1, 'xz (t=1)'),
    ]

    colors = ['C0', 'C0', 'C1', 'C2', 'C2']
    linestyles = ['-', '--', '-', '-', '--']
    markers = ['o', 's', '^', 'D', 'v']

    for ax, operation in zip(axes, operations):
        for (impl, threads, label), color, ls, marker in zip(
            comparisons, colors, linestyles, markers
        ):
            sizes, throughputs = query_results(conn, impl, operation, threads)
            if not sizes:
                continue

            ax.plot(
                sizes,
                throughputs,
                color=color,
                linestyle=ls,
                marker=marker,
                label=label,
                linewidth=2,
                markersize=7,
            )

        ax.set_xlabel('Data Size (KB)')
        ax.set_ylabel('Throughput (MB/s)')
        ax.set_title(f'{operation.capitalize()}: lzma_mt vs stdlib vs xz')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / 'benchmark_comparison.png', dpi=150)
    print(f'Saved {out_dir / "benchmark_comparison.png"}')
    conn.close()


def plot_scaling():
    """Plot thread scaling for lzma_mt at different data sizes."""
    out_dir = Path(__file__).parent
    conn = sqlite3.connect(out_dir / 'results.db')

    operations = [r[0] for r in conn.execute('SELECT DISTINCT operation FROM results')]
    threads_list = [r[0] for r in conn.execute(
        "SELECT DISTINCT threads FROM results WHERE implementation = 'lzma_mt' ORDER BY threads"
    )]

    # Get a few representative data sizes
    all_sizes = [r[0] for r in conn.execute('SELECT DISTINCT data_size_kb FROM results ORDER BY data_size_kb')]
    # Pick small, medium, large
    if len(all_sizes) >= 3:
        selected_sizes = [all_sizes[0], all_sizes[len(all_sizes) // 2], all_sizes[-1]]
    else:
        selected_sizes = all_sizes

    fig, axes = plt.subplots(1, len(operations), figsize=(6 * len(operations), 5))
    if len(operations) == 1:
        axes = [axes]

    colors = ['C0', 'C1', 'C2', 'C3', 'C4']

    for ax, operation in zip(axes, operations):
        for i, data_size_kb in enumerate(selected_sizes):
            throughputs = []
            threads_used = []

            for threads in threads_list:
                cursor = conn.execute(
                    '''SELECT throughput_mbs FROM results
                       WHERE implementation = 'lzma_mt' AND operation = ? AND threads = ? AND data_size_kb = ?''',
                    (operation, threads, data_size_kb),
                )
                row = cursor.fetchone()
                if row:
                    throughputs.append(row[0])
                    threads_used.append(threads if threads != 0 else 'auto')

            if throughputs:
                # Convert 'auto' to a position at the end
                x_labels = [str(t) for t in threads_used]
                x_pos = list(range(len(x_labels)))

                ax.plot(
                    x_pos,
                    throughputs,
                    color=colors[i % len(colors)],
                    marker='o',
                    linewidth=2,
                    markersize=8,
                    label=f'{data_size_kb} KB',
                )

        ax.set_xlabel('Threads')
        ax.set_ylabel('Throughput (MB/s)')
        ax.set_title(f'{operation.capitalize()}: Thread Scaling')
        ax.set_xticks(range(len(threads_list)))
        ax.set_xticklabels([str(t) if t != 0 else 'auto' for t in threads_list])
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / 'benchmark_scaling.png', dpi=150)
    print(f'Saved {out_dir / "benchmark_scaling.png"}')
    conn.close()


def print_summary():
    """Print a summary table of results."""
    out_dir = Path(__file__).parent
    conn = sqlite3.connect(out_dir / 'results.db')

    print('\n' + '=' * 80)
    print('Benchmark Summary')
    print('=' * 80)

    for operation in ['compress', 'decompress']:
        print(f'\n{operation.upper()}:')
        print('-' * 60)

        cursor = conn.execute(
            '''SELECT implementation, threads, AVG(throughput_mbs) as avg_throughput
               FROM results
               WHERE operation = ?
               GROUP BY implementation, threads
               ORDER BY avg_throughput DESC''',
            (operation,),
        )

        print(f'{"Implementation":<15} {"Threads":<10} {"Avg Throughput (MB/s)":<20}')
        for impl, threads, avg_tp in cursor.fetchall():
            t_label = 'auto' if threads == 0 else threads
            print(f'{impl:<15} {t_label:<10} {avg_tp:>15.1f}')

    conn.close()


def plot_preset_comparison():
    """Plot preset 1 vs preset 5 comparison using Wikipedia benchmark data."""
    out_dir = Path(__file__).parent

    db_files = [
        ('results_large.db', 'Preset 1'),
        ('results_preset5.db', 'Preset 5'),
    ]

    # Check which databases exist
    available_dbs = [(f, label) for f, label in db_files if (out_dir / f).exists()]
    if not available_dbs:
        print('No benchmark databases found (results_large.db or results_preset5.db)')
        return

    fig, axes = plt.subplots(2, len(available_dbs), figsize=(7 * len(available_dbs), 10))
    if len(available_dbs) == 1:
        axes = axes.reshape(-1, 1)

    styles = {
        ('lzma_mt', 1): ('C0', '--', 'o', 'lzma_mt t=1'),
        ('lzma_mt', 0): ('C0', '-', 's', 'lzma_mt t=auto'),
        ('stdlib', 1): ('C1', '--', '^', 'stdlib'),
        ('xz', 0): ('C2', '-', 'v', 'xz t=auto'),
    }

    for col, (db_file, preset_label) in enumerate(available_dbs):
        conn = sqlite3.connect(out_dir / db_file)

        for row, op in enumerate(['compress', 'decompress']):
            ax = axes[row, col]

            for (impl, threads), (color, ls, marker, label) in styles.items():
                sizes, throughputs = query_results(conn, impl, op, threads)
                if sizes:
                    ax.plot(
                        [s / 1024 for s in sizes],
                        throughputs,
                        linestyle=ls,
                        marker=marker,
                        color=color,
                        label=label,
                        linewidth=2,
                        markersize=8,
                    )

            ax.set_xlabel('Data Size (MB)')
            ax.set_ylabel('Throughput (MB/s)')
            ax.set_title(f'{op.capitalize()} - {preset_label}')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
            ax.set_yscale('log')

        conn.close()

    plt.tight_layout()
    plt.savefig(out_dir / 'benchmark_preset_comparison.png', dpi=150)
    print(f'Saved {out_dir / "benchmark_preset_comparison.png"}')


if __name__ == '__main__':
    plot_preset_comparison()
    print_summary()
