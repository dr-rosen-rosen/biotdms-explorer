#!/usr/bin/env python3
"""
cli.py — Command-line entrypoint for BioTDMS data processing pipeline.

Called from analysis notebooks via os.system() or subprocess to process
new physiological sensor data without launching the Streamlit UI.

Usage:
  # Process raw merged CSVs into session parquets
  python cli.py --raw-dir /path/to/DCE_data --output-dir data/processed_sessions

  # Also install entropy/AMI CSV(s) and subtask lookup table(s)
  python cli.py --raw-dir /path/to/DCE_data --output-dir data/processed_sessions \\
      --data-dir data/ \\
      --entropy /path/to/team_entropy_ami_DCE2.csv \\
      --entropy /path/to/team_entropy_ami_DCE3.csv \\
      --subtask /path/to/SubTask_LookupTable_DCE3.xlsx

  # Force reprocessing of all sessions (even if parquets exist)
  python cli.py --raw-dir /path/to/DCE_data --output-dir data/processed_sessions --force

  # Quiet mode (summary only)
  python cli.py --raw-dir /path/to/DCE_data --output-dir data/processed_sessions --quiet

Exit codes:
  0  Success (all sessions processed without errors)
  1  Completed with errors (some sessions failed)
  2  Fatal error (bad arguments, missing directories)
"""

import argparse
import sys
import logging
from pathlib import Path

# Ensure the project root is importable so 'core.data_ingest' resolves.
# When called as `python cli.py` from the project root, this is automatic.
# When called from a notebook in a different directory, the notebook should
# either cd to the project root or add it to sys.path before invoking.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.data_ingest import (
    ingest_sessions,
    install_entropy_csv,
    install_subtask_excel,
    rebuild_index,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="biotdms-cli",
        description="Process BioTDMS physiological sensor data for the Explorer app.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required paths
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Path to raw data directory (DCE{N}/Team{N}/... structure with merged CSVs)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Path to processed_sessions output directory",
    )

    # Optional data directory for entropy/subtask file installation
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="App data directory where entropy CSVs and subtask tables are installed (default: data/)",
    )

    # Repeatable file arguments
    parser.add_argument(
        "--entropy",
        type=Path,
        action="append",
        default=[],
        metavar="FILE",
        help=(
            "Path to an entropy/AMI CSV file to install. "
            "Can be specified multiple times for multiple DCEs. "
            "Files are copied preserving their original filenames."
        ),
    )
    parser.add_argument(
        "--subtask",
        type=Path,
        action="append",
        default=[],
        metavar="FILE",
        help=(
            "Path to a subtask lookup table (Excel) to install. "
            "Can be specified multiple times for multiple DCEs. "
            "Files are copied preserving their original filenames."
        ),
    )

    # Flags
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess all sessions even if parquets already exist",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-session output; print summary only",
    )
    parser.add_argument(
        "--rebuild-index-only",
        action="store_true",
        help="Skip processing; just rebuild the sessions_index.parquet from existing parquets",
    )

    return parser


def validate_args(args) -> list:
    """Validate arguments and return a list of error messages (empty if OK)."""
    errors = []

    if not args.rebuild_index_only:
        if not args.raw_dir.exists():
            errors.append(f"Raw data directory not found: {args.raw_dir}")
        elif not args.raw_dir.is_dir():
            errors.append(f"Raw data path is not a directory: {args.raw_dir}")

    for f in args.entropy:
        if not f.exists():
            errors.append(f"Entropy file not found: {f}")
        elif f.suffix.lower() != '.csv':
            errors.append(f"Entropy file is not a CSV: {f}")

    for f in args.subtask:
        if not f.exists():
            errors.append(f"Subtask file not found: {f}")
        elif f.suffix.lower() not in ('.xlsx', '.xls'):
            errors.append(f"Subtask file is not an Excel file: {f}")

    return errors


def run(args) -> int:
    """Execute the pipeline. Returns exit code."""
    had_errors = False

    # ── Rebuild index only mode ─────────────────────────────────────
    if args.rebuild_index_only:
        if not args.output_dir.exists():
            print(f"ERROR: Output directory not found: {args.output_dir}", file=sys.stderr)
            return 2
        print(f"Rebuilding session index from {args.output_dir}...")
        index_df = rebuild_index(args.output_dir)
        print(f"Index rebuilt: {len(index_df)} sessions")
        return 0

    # ── Process sessions ────────────────────────────────────────────
    if not args.quiet:
        print(f"Raw data:   {args.raw_dir.resolve()}")
        print(f"Output:     {args.output_dir.resolve()}")
        print(f"Data dir:   {args.data_dir.resolve()}")
        if args.entropy:
            print(f"Entropy:    {', '.join(str(f) for f in args.entropy)}")
        if args.subtask:
            print(f"Subtask:    {', '.join(str(f) for f in args.subtask)}")
        if args.force:
            print(f"Mode:       FORCE (reprocessing all)")
        print()

    # Progress callback for non-quiet mode
    def progress_cb(current, total, message):
        if not args.quiet:
            idx = min(current + 1, total)
            print(f"  [{idx}/{total}] {message}")

    report = ingest_sessions(
        raw_root=args.raw_dir,
        output_root=args.output_dir,
        skip_existing=not args.force,
        progress_callback=progress_cb if not args.quiet else None,
    )

    # ── Install entropy files ───────────────────────────────────────
    for entropy_file in args.entropy:
        if install_entropy_csv(entropy_file, args.data_dir):
            report.details.append(f"Entropy CSV installed: {entropy_file.name}")
            if not args.quiet:
                print(f"  Installed entropy: {entropy_file.name} -> {args.data_dir}/")
        else:
            report.errors.append(f"Failed to install entropy CSV: {entropy_file}")
            had_errors = True

    # ── Install subtask files ───────────────────────────────────────
    for subtask_file in args.subtask:
        if install_subtask_excel(subtask_file, args.data_dir):
            report.details.append(f"Subtask table installed: {subtask_file.name}")
            if not args.quiet:
                print(f"  Installed subtask: {subtask_file.name} -> {args.data_dir}/")
        else:
            report.errors.append(f"Failed to install subtask table: {subtask_file}")
            had_errors = True

    # ── Summary ─────────────────────────────────────────────────────
    print()
    print(report.summary())

    if report.errors:
        had_errors = True

    return 1 if had_errors else 0


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    parser = build_parser()
    args = parser.parse_args()

    # Validate
    validation_errors = validate_args(args)
    if validation_errors:
        for err in validation_errors:
            print(f"ERROR: {err}", file=sys.stderr)
        sys.exit(2)

    exit_code = run(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
