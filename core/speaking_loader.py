"""
core/speaking_loader.py — Speaking activity loader for BioTDMS Explorer.

Reads zoom_timeseries_*.csv files from data/com_timeseries/, each containing a dense
binary speaking trace (`time`, `Talking`) for one (team, day, session, role).

Canonical processed form: per-role speaking-proportion grid at the entropy sampling
rate (1 Hz by default), aligned to the entropy time axis (with the upstream 159s
lead-in drop applied, mirroring the entropy/AMI convention).

Time alignment knobs live in config (speaking.role_offsets_sec, speaking.lead_in_skip_sec,
speaking.target_hz). All defaults assume zoom_timeseries `time` is seconds-from-session-start
and that the audio recording origin matches the entropy session origin. Both are
unconfirmed upstream assumptions — adjust the offsets once the data team confirms.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


ZOOM_FILENAME_RE = re.compile(
    r'^zoom_timeseries_Team(?P<team>\d+)Day(?P<day>\d+)Session(?P<session>\d+)(?P<role>[A-Za-z]+)\.csv$'
)

DEFAULT_TARGET_HZ = 1.0
DEFAULT_LEAD_IN_SKIP_SEC = 159.0
DEFAULT_OVERLAY_WINDOW_SEC = 60.0


# ── Data classes ────────────────────────────────────────────────────


@dataclass
class SpeakingFile:
    team: str       # "Team1"
    day: str        # "Day2"
    session: str    # "Session2"
    role: str       # "FOA"
    path: Path

    @property
    def session_key(self) -> Tuple[str, str, str]:
        return (self.team, self.day, self.session)


@dataclass
class SpeakingDiscoveryReport:
    matched: List[SpeakingFile] = field(default_factory=list)
    unmatched: List[Path] = field(default_factory=list)
    ignored_other_formats: List[Path] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Matched: {len(self.matched)} zoom_timeseries files",
            f"Unmatched zoom_timeseries: {len(self.unmatched)}",
            f"Ignored (non-zoom format): {len(self.ignored_other_formats)}",
        ]
        if self.unmatched:
            lines.append("Unmatched filenames:")
            for p in self.unmatched:
                lines.append(f"  - {p.name}")
        return "\n".join(lines)


# ── Discovery ───────────────────────────────────────────────────────


def parse_zoom_filename(path: Path) -> Optional[SpeakingFile]:
    """Parse `zoom_timeseries_Team{N}Day{N}Session{N}{Role}.csv` into a SpeakingFile."""
    m = ZOOM_FILENAME_RE.match(path.name)
    if not m:
        return None
    return SpeakingFile(
        team=f"Team{m.group('team')}",
        day=f"Day{m.group('day')}",
        session=f"Session{m.group('session')}",
        role=m.group('role'),
        path=path,
    )


def discover_speaking_files(com_dir: Path) -> SpeakingDiscoveryReport:
    """Walk com_timeseries dir, classify CSV files.

    Diarization-format CSVs (Team*.csv without zoom_timeseries_ prefix) are
    classified as `ignored_other_formats` — they're not the canonical source.
    """
    report = SpeakingDiscoveryReport()
    if not com_dir.exists():
        return report

    for path in sorted(com_dir.glob('*.csv')):
        if path.name.startswith('zoom_timeseries_'):
            parsed = parse_zoom_filename(path)
            if parsed is None:
                report.unmatched.append(path)
            else:
                report.matched.append(parsed)
        else:
            report.ignored_other_formats.append(path)

    return report


# ── Resampling ──────────────────────────────────────────────────────


def to_speaking_grid(
    df: pd.DataFrame,
    target_hz: float = DEFAULT_TARGET_HZ,
    role_offset_sec: float = 0.0,
    lead_in_skip_sec: float = DEFAULT_LEAD_IN_SKIP_SEC,
    session_duration_sec: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample (time, Talking) to a uniform grid aligned to the entropy axis.

    Args:
        df: must have columns 'time' (seconds, float) and 'Talking' (0/1).
        target_hz: output grid frequency (1.0 Hz to match entropy by default).
        role_offset_sec: added to df['time'] to shift to a common session origin.
            Use a positive value if the audio recording started AFTER session t=0.
        lead_in_skip_sec: seconds dropped from the start to mirror the upstream
            entropy lead-in convention. Output time axis starts at 0, corresponding
            to session second `lead_in_skip_sec`.
        session_duration_sec: if given, the grid extends to this many session-relative
            seconds; otherwise it ends at the last sample's adjusted time.

    Returns:
        (timestamps, proportions) — same axis as load_entropy_timeseries output.
        Proportions are mean(Talking) per bin in [0, 1]; bins with no samples are NaN.
    """
    if df.empty or 'time' not in df.columns or 'Talking' not in df.columns:
        return np.array([]), np.array([])

    t = df['time'].to_numpy(dtype=float) + float(role_offset_sec)
    v = df['Talking'].to_numpy(dtype=float)

    grid_start_sec = float(lead_in_skip_sec)
    grid_end_sec = float(session_duration_sec) if session_duration_sec is not None else float(t.max())
    if grid_end_sec <= grid_start_sec:
        return np.array([]), np.array([])

    bin_width = 1.0 / target_hz
    edges = np.arange(grid_start_sec, grid_end_sec + bin_width, bin_width)
    if len(edges) < 2:
        return np.array([]), np.array([])

    n_bins = len(edges) - 1
    idx = np.searchsorted(edges, t, side='right') - 1
    valid = (idx >= 0) & (idx < n_bins)

    sums = np.zeros(n_bins, dtype=float)
    counts = np.zeros(n_bins, dtype=int)
    np.add.at(sums, idx[valid], v[valid])
    np.add.at(counts, idx[valid], 1)

    with np.errstate(invalid='ignore', divide='ignore'):
        props = np.where(counts > 0, sums / counts, np.nan)

    timestamps = edges[:-1] - grid_start_sec
    return timestamps, props


# ── Loader ──────────────────────────────────────────────────────────


class SpeakingLoader:
    """Loads speaking-proportion timeseries from data/com_timeseries/."""

    def __init__(
        self,
        com_dir: Path,
        role_offsets_sec: Optional[Dict[str, float]] = None,
        lead_in_skip_sec: float = DEFAULT_LEAD_IN_SKIP_SEC,
        target_hz: float = DEFAULT_TARGET_HZ,
        overlay_window_sec: float = DEFAULT_OVERLAY_WINDOW_SEC,
    ):
        self.com_dir = Path(com_dir)
        self.role_offsets_sec: Dict[str, float] = dict(role_offsets_sec or {})
        self.lead_in_skip_sec = float(lead_in_skip_sec)
        self.target_hz = float(target_hz)
        self.overlay_window_sec = float(overlay_window_sec)
        self._discovery: Optional[SpeakingDiscoveryReport] = None
        self._index: Optional[pd.DataFrame] = None

    @property
    def discovery(self) -> SpeakingDiscoveryReport:
        if self._discovery is None:
            self._discovery = discover_speaking_files(self.com_dir)
        return self._discovery

    @property
    def index(self) -> pd.DataFrame:
        if self._index is None:
            rows = [
                {
                    'team': f.team, 'day': f.day, 'session': f.session,
                    'role': f.role, 'path': str(f.path),
                }
                for f in self.discovery.matched
            ]
            self._index = pd.DataFrame(rows)
        return self._index

    def has_data(self) -> bool:
        return not self.index.empty

    def find_file(self, team: str, day: str, session: str, role: str) -> Optional[Path]:
        if self.index.empty:
            return None
        m = self.index[
            (self.index['team'] == team) &
            (self.index['day'] == day) &
            (self.index['session'] == session) &
            (self.index['role'] == role)
        ]
        if m.empty:
            return None
        return Path(m.iloc[0]['path'])

    def load_speaking_grid(
        self,
        team: str,
        day: str,
        session: str,
        role: str,
        target_hz: Optional[float] = None,
        session_duration_sec: Optional[float] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return (timestamps, speaking_proportion) for one role/session, or None."""
        path = self.find_file(team, day, session, role)
        if path is None:
            return None
        try:
            df = pd.read_csv(path)
        except Exception as e:
            logger.warning(f"Could not load {path.name}: {e}")
            return None
        return to_speaking_grid(
            df,
            target_hz=target_hz if target_hz is not None else self.target_hz,
            role_offset_sec=self.role_offsets_sec.get(role, 0.0),
            lead_in_skip_sec=self.lead_in_skip_sec,
            session_duration_sec=session_duration_sec,
        )

    def available_sessions(self) -> List[Tuple[str, str, str]]:
        if self.index.empty:
            return []
        return sorted({(r['team'], r['day'], r['session']) for _, r in self.index.iterrows()})

    def available_roles(self, team: str, day: str, session: str) -> List[str]:
        if self.index.empty:
            return []
        m = self.index[
            (self.index['team'] == team) &
            (self.index['day'] == day) &
            (self.index['session'] == session)
        ]
        return sorted(m['role'].unique().tolist())

    def load_session_overlay(
        self,
        team: str,
        day: str,
        session: str,
        roles: List[str],
        window_sec: Optional[float] = None,
    ) -> Optional[Dict[str, object]]:
        """Build an overlay-ready data bundle for one session.

        Returns a dict with:
          - 'window_sec': float, the bin width used
          - 'timestamps': np.ndarray of bin-start times (seconds, post-lead-in axis)
            All roles share this axis (extended to the longest role's max time).
          - 'role_props': dict[role -> np.ndarray of shape (n_bins,)] with mean
            speaking proportion per bin (NaN where the role has no data in that bin).
          - 'roles_present': list[str] roles that had a file (subset of `roles`).
          - 'roles_missing': list[str] roles with no zoom file for this session.

        Returns None if no roles in `roles` had any data for this session.
        """
        ws = float(window_sec if window_sec is not None else self.overlay_window_sec)
        if ws <= 0:
            return None

        target_hz = 1.0 / ws

        # Load per-role grids at the overlay rate
        per_role: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        roles_missing: List[str] = []
        for role in roles:
            grid = self.load_speaking_grid(
                team=team, day=day, session=session, role=role,
                target_hz=target_hz,
            )
            if grid is None:
                roles_missing.append(role)
                continue
            ts, props = grid
            if len(ts) == 0:
                roles_missing.append(role)
                continue
            per_role[role] = (ts, props)

        if not per_role:
            return None

        # Build a shared time axis covering the longest role
        max_n = max(len(ts) for ts, _ in per_role.values())
        # All grids start at 0 with the same bin width, so we can take the
        # longest as the canonical axis.
        canonical_ts = next(
            ts for ts, _ in per_role.values() if len(ts) == max_n
        )

        role_props: Dict[str, np.ndarray] = {}
        for role, (ts, props) in per_role.items():
            if len(props) == max_n:
                role_props[role] = props
            else:
                # Pad shorter roles with NaN so all arrays share the axis
                padded = np.full(max_n, np.nan, dtype=float)
                padded[:len(props)] = props
                role_props[role] = padded

        return {
            'window_sec': ws,
            'timestamps': canonical_ts,
            'role_props': role_props,
            'roles_present': list(role_props.keys()),
            'roles_missing': roles_missing,
        }
