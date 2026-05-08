"""
core/data_ingest.py — Data ingestion for BioTDMS Explorer

Converts raw merged per-person CSVs into session parquets that the existing
DataLoader / UC2 pipeline expects. Also handles installing entropy/AMI CSVs
and subtask lookup tables into the app data directory.

Directory convention (input):
  {raw_root}/DCE{N}/Team{N}/Session{N}/{Role}_Subj{NNN}/merged/merged_{Day}_{Session}_{Role}_Subj{NNN}_1hz.csv

Output convention (matching existing app):
  data/processed_sessions/{DCE}/{Team}_{Day}_{Session}.parquet
  data/processed_sessions/sessions_index.parquet

Entropy/AMI and subtask files are installed by copying them into the data
directory with their original filenames preserved, so multiple per-DCE files
can coexist (e.g., team_entropy_ami_DCE2.csv and team_entropy_ami_DCE3.csv).
"""

import re
import shutil
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable

import pandas as pd

logger = logging.getLogger(__name__)

# ── Filename / path parsing ─────────────────────────────────────────

MERGED_FILENAME_RE = re.compile(
    r'^merged_(?P<day>Day\d+)_(?P<session>Session\d+)_(?P<role>[A-Za-z]+)_(?P<subject>Subj\d+)_(?P<rate>\w+)\.csv$'
)

# Columns that are metadata (don't get role-prefixed)
META_COLUMNS = {'time_stamp', 'trial_running'}


# ── Data classes ────────────────────────────────────────────────────

@dataclass
class MemberFile:
    """Parsed info for one merged CSV file."""
    dce: str
    team: str
    day: str
    session: str
    role: str
    subject: str
    path: Path


@dataclass
class SessionGroup:
    """A group of member files forming one team session."""
    dce: str
    team: str
    day: str
    session: str
    members: List[MemberFile] = field(default_factory=list)

    @property
    def key(self) -> Tuple[str, str, str, str]:
        return (self.dce, self.team, self.day, self.session)

    @property
    def parquet_name(self) -> str:
        return f"{self.team}_{self.day}_{self.session}.parquet"

    @property
    def roles(self) -> List[str]:
        return sorted(set(m.role for m in self.members))


@dataclass
class IngestReport:
    """Report from an ingestion run."""
    discovered: int = 0
    new_sessions: int = 0
    skipped_existing: int = 0
    processed: int = 0
    errors: List[str] = field(default_factory=list)
    details: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"Discovered: {self.discovered} sessions",
            f"New:        {self.new_sessions}",
            f"Processed:  {self.processed}",
            f"Skipped:    {self.skipped_existing}",
            f"Errors:     {len(self.errors)}",
        ]
        if self.errors:
            lines.append("")
            lines.append("Errors:")
            for err in self.errors:
                lines.append(f"  - {err}")
        if self.details:
            lines.append("")
            lines.append("Details:")
            for d in self.details:
                lines.append(f"  - {d}")
        return "\n".join(lines)


# ── Parsing ─────────────────────────────────────────────────────────

def parse_merged_csv(filepath: Path) -> Optional[MemberFile]:
    """Parse metadata from a merged CSV filepath."""
    fname_match = MERGED_FILENAME_RE.match(filepath.name)
    if not fname_match:
        return None

    # Walk up path to find DCE/Team folders
    parts = filepath.parts
    dce = next((p for p in parts if re.match(r'^DCE\d+$', p)), None)
    team = next((p for p in parts if re.match(r'^Team\d+$', p)), None)

    if not dce or not team:
        return None

    return MemberFile(
        dce=dce,
        team=team,
        day=fname_match.group('day'),
        session=fname_match.group('session'),
        role=fname_match.group('role'),
        subject=fname_match.group('subject'),
        path=filepath,
    )


# ── Discovery ───────────────────────────────────────────────────────

def discover_sessions(raw_root: Path) -> List[SessionGroup]:
    """Discover all merged CSV files and group by team session."""
    groups: Dict[tuple, SessionGroup] = {}

    for csv_path in sorted(raw_root.rglob('merged_*_1hz.csv')):
        # Only files inside a 'merged' subfolder
        if csv_path.parent.name != 'merged':
            continue

        member = parse_merged_csv(csv_path)
        if not member:
            continue

        key = (member.dce, member.team, member.day, member.session)
        if key not in groups:
            groups[key] = SessionGroup(
                dce=member.dce, team=member.team,
                day=member.day, session=member.session
            )
        groups[key].members.append(member)

    return sorted(groups.values(), key=lambda g: g.key)


def find_new_sessions(
    raw_root: Path,
    output_root: Path
) -> Tuple[List[SessionGroup], List[SessionGroup]]:
    """Discover sessions and split into new vs. already-processed.

    Returns: (new_sessions, existing_sessions)
    """
    all_sessions = discover_sessions(raw_root)
    new = []
    existing = []

    for sg in all_sessions:
        out_path = output_root / sg.dce / sg.parquet_name
        if out_path.exists():
            existing.append(sg)
        else:
            new.append(sg)

    return new, existing


# ── Processing ──────────────────────────────────────────────────────

def process_session(sg: SessionGroup, output_root: Path) -> Optional[Path]:
    """Process one team session: load member CSVs, merge, write parquet.

    Returns path to the written parquet, or None on error.
    """
    member_dfs = []

    for member in sorted(sg.members, key=lambda m: m.role):
        try:
            df = pd.read_csv(member.path)
        except Exception as e:
            raise RuntimeError(f"Failed to read {member.path.name}: {e}")

        # Separate meta and signal columns
        signal_cols = [c for c in df.columns if c not in META_COLUMNS]

        # Rename signal columns with role prefix
        rename_map = {col: f"{member.role}_{col}" for col in signal_cols}
        df = df.rename(columns=rename_map)

        # Keep time_stamp + prefixed signals + trial_running
        keep = ['time_stamp'] + list(rename_map.values())
        if 'trial_running' in df.columns:
            keep.append('trial_running')
        df = df[[c for c in keep if c in df.columns]]

        member_dfs.append((member.role, df))

    if not member_dfs:
        return None

    # Merge all members on time_stamp
    merged = member_dfs[0][1]
    for role, df in member_dfs[1:]:
        # Avoid duplicate trial_running
        if 'trial_running' in df.columns and 'trial_running' in merged.columns:
            df = df.drop(columns=['trial_running'])
        merged = pd.merge(merged, df, on='time_stamp', how='outer',
                          suffixes=('', f'_dup_{role}'))

    merged = merged.sort_values('time_stamp').reset_index(drop=True)

    # Add metadata columns (matching existing parquet convention)
    merged.insert(0, 'dce', sg.dce)
    merged.insert(1, 'team', sg.team)
    merged.insert(2, 'day', sg.day)
    merged.insert(3, 'session', sg.session)

    # Write parquet
    out_dir = output_root / sg.dce
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / sg.parquet_name
    merged.to_parquet(out_path, index=False)

    return out_path


def rebuild_index(output_root: Path) -> pd.DataFrame:
    """Rebuild the sessions_index.parquet from all session parquets."""
    records = []

    for pq in sorted(output_root.rglob('*.parquet')):
        if pq.name == 'sessions_index.parquet':
            continue

        try:
            df = pd.read_parquet(pq)
            meta = {'dce', 'team', 'day', 'session', 'time_stamp', 'trial_running',
                    'scenario_name', 'session_time'}
            signal_cols = [c for c in df.columns if c not in meta]

            # Extract roles from column prefixes
            roles = sorted(set(
                c.split('_')[0] for c in signal_cols
                if '_' in c and c.split('_')[0].isalpha()
                and c.split('_')[0][0].isupper()
            ))

            records.append({
                'dce': df['dce'].iloc[0] if 'dce' in df.columns else pq.parent.name,
                'team': df['team'].iloc[0] if 'team' in df.columns else pq.stem.split('_')[0],
                'day': df['day'].iloc[0] if 'day' in df.columns else pq.stem.split('_')[1],
                'session': df['session'].iloc[0] if 'session' in df.columns else '_'.join(pq.stem.split('_')[2:]),
                'n_rows': len(df),
                'n_signals': len(signal_cols),
                'roles': ','.join(roles),
                'n_roles': len(roles),
                'path': str(pq.relative_to(output_root)),
            })
        except Exception as e:
            logger.warning(f"Could not index {pq}: {e}")

    index_df = pd.DataFrame(records)
    if not index_df.empty:
        index_path = output_root / 'sessions_index.parquet'
        index_df.to_parquet(index_path, index=False)

    return index_df


# ── Main ingestion orchestrator ─────────────────────────────────────

def ingest_sessions(
    raw_root: Path,
    output_root: Path,
    skip_existing: bool = True,
    progress_callback: Optional[Callable] = None
) -> IngestReport:
    """Run the full ingestion pipeline.

    Args:
        raw_root: Path to raw data (DCE{N}/Team{N}/... structure)
        output_root: Path to processed_sessions output
        skip_existing: If True, skip sessions that already have parquets
        progress_callback: Optional callable(current, total, message) for UI updates

    Returns:
        IngestReport with counts and any errors
    """
    report = IngestReport()

    # Discover
    all_sessions = discover_sessions(raw_root)
    report.discovered = len(all_sessions)

    if skip_existing:
        new, existing = find_new_sessions(raw_root, output_root)
        report.skipped_existing = len(existing)
        to_process = new
    else:
        to_process = all_sessions

    report.new_sessions = len(to_process)

    if not to_process:
        report.details.append("No new sessions to process.")
        return report

    # Process each session
    for i, sg in enumerate(to_process):
        label = f"{sg.dce}/{sg.team}/{sg.day}/{sg.session}"

        if progress_callback:
            progress_callback(i, len(to_process), f"Processing {label}...")

        try:
            out_path = process_session(sg, output_root)
            if out_path:
                report.processed += 1
                roles = ', '.join(sg.roles)
                report.details.append(f"{label} ({roles}) -> {out_path.name}")
            else:
                report.errors.append(f"{label}: no data produced")
        except Exception as e:
            report.errors.append(f"{label}: {e}")

    if progress_callback:
        progress_callback(len(to_process), len(to_process), "Rebuilding index...")

    # Rebuild index
    rebuild_index(output_root)

    return report


# ── File installation helpers ───────────────────────────────────────
#
# These copy supporting data files into the app's data/ directory,
# preserving original filenames so multiple per-DCE files can coexist.
# For example:
#   team_entropy_ami_DCE2.csv  and  team_entropy_ami_DCE3.csv
#   SubTask_LookupTable_DCE2.xlsx  and  SubTask_LookupTable_DCE3.xlsx

def install_data_file(source_path: Path, data_dir: Path) -> bool:
    """Copy a data file into the app data directory, preserving its filename.

    Returns True if successful.
    """
    dest = data_dir / source_path.name
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest)
        logger.info(f"Installed {source_path.name} -> {dest}")
        return True
    except Exception as e:
        logger.error(f"Error installing {source_path.name}: {e}")
        return False


def install_entropy_csv(source_path: Path, data_dir: Path) -> bool:
    """Copy an entropy/AMI CSV into the app's data directory.

    Preserves the original filename so multiple per-DCE files can coexist.
    The DataLoader should glob for team_entropy_ami*.csv.

    Returns True if successful.
    """
    return install_data_file(source_path, data_dir)


def install_subtask_excel(source_path: Path, data_dir: Path) -> bool:
    """Copy a subtask lookup table Excel file into the app's data directory.

    Preserves the original filename so multiple per-DCE files can coexist.
    The DataLoader should glob for SubTask_LookupTable*.xlsx.

    Returns True if successful.
    """
    return install_data_file(source_path, data_dir)


def install_speaking_dir(source_dir: Path, data_dir: Path) -> Tuple[int, List[str]]:
    """Copy zoom_timeseries_*.csv files from `source_dir` into the app's
    `com_timeseries` subdirectory.

    The SpeakingLoader globs for `data/com_timeseries/zoom_timeseries_*.csv`,
    so any non-zoom_timeseries CSVs in `source_dir` are skipped (and logged).

    Returns:
        (n_copied, errors) — count of files copied and a list of error messages.
    """
    errors: List[str] = []
    if not source_dir.exists() or not source_dir.is_dir():
        errors.append(f"Speaking source dir not found or not a directory: {source_dir}")
        return 0, errors

    target = data_dir / 'com_timeseries'
    target.mkdir(parents=True, exist_ok=True)

    n_copied = 0
    for path in sorted(source_dir.glob('zoom_timeseries_*.csv')):
        try:
            shutil.copy2(path, target / path.name)
            n_copied += 1
            logger.info(f"Installed speaking file: {path.name} -> {target}/")
        except Exception as e:
            err = f"Failed to copy {path.name}: {e}"
            errors.append(err)
            logger.error(err)

    # Tally non-matching CSVs for visibility
    skipped = [p.name for p in source_dir.glob('*.csv')
               if not p.name.startswith('zoom_timeseries_')]
    if skipped:
        logger.info(
            f"Skipped {len(skipped)} non-zoom_timeseries CSVs in {source_dir}: "
            f"{', '.join(skipped[:5])}" + ('...' if len(skipped) > 5 else '')
        )

    return n_copied, errors
