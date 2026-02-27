"""
Compare benchmark results from JSON report files.

The first file (or the latest when auto-discovered) is used as the
**reference** for speedup computation against the others.

Usage:
    # Compare two specific runs (first = reference)
    python compare_benchmarks.py old_baseline.json new_run.json

    # Compare latest run against all compatible previous runs
    python compare_benchmarks.py

    # Compare a specific reference against all compatible runs
    python compare_benchmarks.py my_baseline.json

    # List all available benchmark results
    python compare_benchmarks.py --list

    # Compare latest N runs (most recent = reference)
    python compare_benchmarks.py --latest 3

    # Filter comparison to a specific similarity
    python compare_benchmarks.py --similarity cosine

Requirements:
    pip install -e ".[bench]"
"""

import argparse
import json
import sys
from pathlib import Path


# ── I/O helpers ──────────────────────────────────────────────────────────────

def load_benchmark(filepath):
    """Load a benchmark JSON report file.

    Parameters
    ----------
    filepath : Path
        Path to the JSON file.

    Returns
    -------
    dict
        Parsed benchmark report.
    """
    with open(filepath) as f:
        return json.load(f)


def find_json_reports(bench_dir, pattern=None):
    """Find all JSON benchmark reports in a directory.

    Parameters
    ----------
    bench_dir : Path
        Directory to scan.
    pattern : str, optional
        Filter filenames containing this substring.

    Returns
    -------
    list of Path
        Sorted list of matching JSON files (oldest first).
    """
    files = sorted(bench_dir.glob("result_*.json"))
    if pattern:
        files = [f for f in files if pattern in f.stem]
    return files


def resolve_path(raw, bench_dir):
    """Resolve a user-supplied path to an existing file.

    Tries absolute/relative first, then relative to *bench_dir*.

    Parameters
    ----------
    raw : str
        Path string from CLI.
    bench_dir : Path
        Fallback directory.

    Returns
    -------
    Path
    """
    path = Path(raw)
    if path.exists():
        return path
    alt = bench_dir / path
    if alt.exists():
        return alt
    print(f"Error: file not found: {raw}")
    sys.exit(1)


def get_dataset_keys(report):
    """Return the set of dataset keys (e.g. ``{'movielens:32m'}``) in a report."""
    return set(report.get("results", {}).keys())


def find_compatible_reports(bench_dir, reference_report, exclude_paths=None, pattern=None):
    """Find reports sharing at least one dataset with the reference.

    Parameters
    ----------
    bench_dir : Path
        Directory to scan.
    reference_report : dict
        The parsed reference report.
    exclude_paths : set of Path, optional
        Paths to skip (e.g. the reference itself).
    pattern : str, optional
        Filter filenames containing this substring.

    Returns
    -------
    list of (Path, dict)
        Compatible ``(filepath, report)`` pairs, oldest first.
    """
    ref_datasets = get_dataset_keys(reference_report)
    exclude = set(exclude_paths or [])
    compatible = []
    for filepath in find_json_reports(bench_dir, pattern):
        if filepath in exclude:
            continue
        try:
            report = load_benchmark(filepath)
            if get_dataset_keys(report) & ref_datasets:
                compatible.append((filepath, report))
        except (json.JSONDecodeError, KeyError):
            continue
    return compatible


# ── Display helpers ──────────────────────────────────────────────────────────


def format_time(computation_time, std_time):
    """Format time with optional ± std."""
    if std_time and std_time > 0:
        return f"{computation_time:.2f} ± {std_time:.2f}"
    return f"{computation_time:.2f}"


def format_report_summary(filepath, report):
    """Format a one-line summary of a benchmark report.

    Parameters
    ----------
    filepath : Path
        Path to the report file.
    report : dict
        Parsed report data.

    Returns
    -------
    str
        Formatted summary line.
    """
    meta = report["metadata"]
    config = report["config"]
    datasets = ", ".join(f"{d}:{v}" for d, v in config["datasets"])
    sims = ", ".join(config["similarities"])
    note = meta.get("note", "")
    note_str = f"  [{note}]" if note else ""

    return (
        f"{filepath.name:<60} "
        f"v{meta['similaripy_version']:<8} "
        f"{meta['timestamp']:<20} "
        f"{datasets:<20} "
        f"{sims:<30} "
        f"k={config['k']:<5} "
        f"rounds={config['rounds']}{note_str}"
    )


def list_reports(bench_dir, pattern=None):
    """Print a table of all available benchmark reports.

    Parameters
    ----------
    bench_dir : Path
        Directory to scan.
    pattern : str, optional
        Filter filenames containing this substring.
    """
    files = find_json_reports(bench_dir, pattern)
    if not files:
        print(f"No JSON benchmark reports found in {bench_dir}")
        if pattern:
            print(f"  (filter: '{pattern}')")
        return

    print(f"\nFound {len(files)} benchmark report(s) in {bench_dir}:\n")
    header = (
        f"{'#':<4} "
        f"{'Filename':<60} "
        f"{'Version':<8} "
        f"{'Timestamp':<20} "
        f"{'Datasets':<20} "
        f"{'Similarities':<30} "
        f"{'Params'}"
    )
    print(header)
    print("-" * len(header))

    for i, filepath in enumerate(files, 1):
        try:
            report = load_benchmark(filepath)
            print(f"{i:<4} {format_report_summary(filepath, report)}")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"{i:<4} {filepath.name:<60} [ERROR: {e}]")

    print()


# ── Comparison logic ─────────────────────────────────────────────────────────

def compare_reports(reports, similarity_filter=None):
    """Print a comparison table with speedups relative to the reference.

    The first entry in *reports* is the **reference**.  Every subsequent
    entry is compared against it.  Output is grouped by similarity type,
    with one row per benchmark run.

    Parameters
    ----------
    reports : list of (Path, dict)
        ``(filepath, parsed_report)`` tuples.  First = reference.
    similarity_filter : str, optional
        Only show results for this similarity type.
    """
    if len(reports) < 2:
        print("Need at least 2 reports to compare.")
        return

    ref_path, ref_report = reports[0]
    others = reports[1:]
    labels = [chr(66 + i) for i in range(len(others))]  # B, C, D, ...

    # ── Legend ────────────────────────────────────────────────────────────
    w = 130
    print(f"\n{'=' * w}")
    print("BENCHMARK COMPARISON")
    print(f"{'=' * w}")

    ref_meta = ref_report["metadata"]
    ref_note = ref_meta.get("note", "")
    ref_note_str = f"  [{ref_note}]" if ref_note else ""
    print(
        f"  [REF] {ref_path.name} — "
        f"v{ref_meta['similaripy_version']}, "
        f"{ref_meta['timestamp']}, "
        f"git:{ref_meta.get('git_hash', '?')}"
        f"{ref_note_str}"
    )
    for label, (filepath, report) in zip(labels, others):
        meta = report["metadata"]
        note = meta.get("note", "")
        note_str = f"  [{note}]" if note else ""
        print(
            f"  [{label}]   {filepath.name} — "
            f"v{meta['similaripy_version']}, "
            f"{meta['timestamp']}, "
            f"git:{meta.get('git_hash', '?')}"
            f"{note_str}"
        )
    print(f"{'=' * w}")

    # ── Iterate over dataset keys from the reference ─────────────────────
    ref_datasets = sorted(ref_report.get("results", {}).keys())

    for ds_key in ref_datasets:
        ref_ds_results = ref_report["results"].get(ds_key, {})
        if not ref_ds_results:
            continue

        similarities = sorted(ref_ds_results.keys())
        if similarity_filter:
            similarities = [s for s in similarities if s == similarity_filter]
        if not similarities:
            continue

        print(f"\nDataset: {ds_key}")

        for sim_type in similarities:
            ref_result = ref_ds_results[sim_type]
            ref_time = ref_result["computation_time"]

            # ── Similarity sub-header + column header ────────────────────
            print(f"\n{sim_type.upper()}")
            header = (
                f"{'Label':<8} "
                f"{'Version':<10} "
                f"{'Time (s)':<18} "
                f"{'Throughput':<12} "
                f"{'Avg Neighbors':<15} "
                f"{'Rounds':<8} "
                f"{'Speedup':<10} "
                f"{'Note'}"
            )
            print(header)
            print("-" * len(header))

            # ── REF row ──────────────────────────────────────────────────
            time_str = format_time(ref_time, ref_result.get("std_time", 0))
            row = (
                f"{'[REF]':<8} "
                f"{'v' + ref_meta['similaripy_version']:<10} "
                f"{time_str:<18} "
                f"{ref_result['throughput']:<12.1f} "
                f"{ref_result['avg_neighbors']:<15.1f} "
                f"{ref_result['rounds']:<8} "
                f"{'-':<10} "
                f"{ref_note}"
            )
            print(row)

            # ── Other rows ───────────────────────────────────────────────
            for label, (filepath, report) in zip(labels, others):
                meta = report["metadata"]
                note_str = meta.get("note", "")
                result = report.get("results", {}).get(ds_key, {}).get(sim_type)

                if result:
                    t = result["computation_time"]
                    time_str = format_time(t, result.get("std_time", 0))
                    speedup = ref_time / t if t > 0 else float("inf")
                    speedup_str = f"{speedup:.2f}x"
                    row = (
                        f"{'[' + label + ']':<8} "
                        f"{'v' + meta['similaripy_version']:<10} "
                        f"{time_str:<18} "
                        f"{result['throughput']:<12.1f} "
                        f"{result['avg_neighbors']:<15.1f} "
                        f"{result['rounds']:<8} "
                        f"{speedup_str:<10} "
                        f"{note_str}"
                    )
                else:
                    row = (
                        f"{'[' + label + ']':<8} "
                        f"{'v' + meta['similaripy_version']:<10} "
                        f"{'N/A':<18} "
                        f"{'N/A':<12} "
                        f"{'N/A':<15} "
                        f"{'N/A':<8} "
                        f"{'N/A':<10} "
                        f"{note_str}"
                    )
                print(row)

    print(f"\n{'=' * w}")
    print("Speedup = Time REF / Time.  >1.00x = faster than reference.")
    print()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare similaripy benchmark results from JSON reports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all benchmark results
  python compare_benchmarks.py --list

  # Compare latest run against all compatible previous runs
  python compare_benchmarks.py

  # Compare a specific reference against all compatible runs
  python compare_benchmarks.py my_baseline.json

  # Compare two specific files (first = reference for speedup)
  python compare_benchmarks.py old_baseline.json new_run.json

  # Compare latest 3 runs (most recent = reference)
  python compare_benchmarks.py --latest 3

  # Filter comparison to a specific similarity
  python compare_benchmarks.py --latest 2 --similarity cosine
        """,
    )

    parser.add_argument(
        "files", nargs="*",
        help="JSON report files to compare. First = reference for speedup. "
             "If omitted, latest run is used as reference.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available benchmark reports",
    )
    parser.add_argument(
        "--latest", type=int, default=None, metavar="N",
        help="Compare the latest N benchmark runs (most recent = reference)",
    )
    parser.add_argument(
        "--filter", type=str, default=None, metavar="PATTERN",
        help="Filter report filenames containing PATTERN (e.g., version number)",
    )
    parser.add_argument(
        "--similarity", type=str, default=None,
        help="Only compare results for this similarity type",
    )
    parser.add_argument(
        "--bench-dir", type=str, default="bench_results",
        help="Directory containing benchmark results (default: bench_results)",
    )

    args = parser.parse_args()
    bench_dir = Path(args.bench_dir)

    if not bench_dir.exists():
        print(f"Error: benchmark directory '{bench_dir}' not found.")
        sys.exit(1)

    # ── List mode ────────────────────────────────────────────────────────
    if args.list:
        list_reports(bench_dir, args.filter)
        return

    # ── Determine reference + comparison files ───────────────────────────
    reports = []

    if args.files:
        # Explicit files: first = reference
        for f in args.files:
            path = resolve_path(f, bench_dir)
            reports.append((path, load_benchmark(path)))

        if len(reports) == 1:
            # Only reference given → find compatible others
            ref_path, ref_report = reports[0]
            compatible = find_compatible_reports(
                bench_dir, ref_report,
                exclude_paths={ref_path},
                pattern=args.filter,
            )
            if not compatible:
                print("No compatible reports found for comparison.")
                sys.exit(1)
            reports.extend(compatible)

    elif args.latest:
        # Latest N: most recent = reference
        all_files = find_json_reports(bench_dir, args.filter)
        if len(all_files) < 2:
            print("Not enough JSON reports found. Run benchmarks first or use --list.")
            sys.exit(1)
        selected = all_files[-args.latest:] if args.latest <= len(all_files) else all_files
        # Most recent last → reverse so most recent = first (reference)
        selected = list(reversed(selected))
        for filepath in selected:
            reports.append((filepath, load_benchmark(filepath)))

    else:
        # No arguments: latest = reference, find compatible others
        all_files = find_json_reports(bench_dir, args.filter)
        if not all_files:
            print("No JSON reports found. Run benchmarks first or use --list.")
            sys.exit(1)
        ref_path = all_files[-1]
        ref_report = load_benchmark(ref_path)
        compatible = find_compatible_reports(
            bench_dir, ref_report,
            exclude_paths={ref_path},
            pattern=args.filter,
        )
        if not compatible:
            print("No compatible reports found for comparison.")
            sys.exit(1)
        reports = [(ref_path, ref_report)] + compatible

    if len(reports) < 2:
        print("Need at least 2 reports to compare. Use --list to see available reports.")
        sys.exit(1)

    compare_reports(reports, similarity_filter=args.similarity)


if __name__ == "__main__":
    main()
