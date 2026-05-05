"""
core/speaking_loader.py — Speaking activity loader for BioTDMS Explorer.

Reads zoom_timeseries_*.csv files from data/com_timeseries/, each containing a dense
binary speaking trace (`time`, `Talking`) for one (team, day, session, role).

Canonical processed form: per-role speaking-proportion grid at the entropy sampling
rate (1 Hz by default), end-aligned to the entropy time axis.

Alignment policy (per upstream confirmation):
  - The comm file and the entropy data end at the same wall-clock second
    (session stop). All files in a session share their END point.
  - Per-role recording start times vary; some recordings dropped out before
    the session ended. Each file's `time` column is "seconds since that
    role's recording start", so we anchor each file's MAX time to the
    common end and work backwards by `target_length_sec` (= number of
    entropy rows for that scenario).
  - Anything before the entropy data's first second is dropped. If a comm
    file is shorter than the entropy axis (recording dropped early), the
    start of its grid is padded with NaN.
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
    target_length_sec: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample (time, Talking) to a uniform end-anchored grid.

    Anchors the LAST sample of the comm file to the entropy end and works
    backwards. Output timestamps run 0..target_length_sec, where 0 is the
    entropy axis start (= session_end - target_length_sec).

    Args:
        df: must have columns 'time' (seconds, float) and 'Talking' (0/1).
        target_hz: output grid frequency (1.0 Hz to match entropy by default).
        role_offset_sec: added to df['time'] before alignment. Use only if the
            recording end-time itself is known to differ from the entropy end
            by a fixed per-role amount.
        target_length_sec: if given, output is exactly target_length_sec * target_hz
            bins, end-aligned. Anything before (file_end - target_length_sec) is
            dropped. If the comm file is shorter than target_length_sec, the start
            is NaN-padded. If None, output covers the full file from the first
            sample.

    Returns:
        (timestamps, proportions) — timestamps in seconds on the entropy axis;
        proportions are mean(Talking) per bin in [0, 1]; bins with no samples
        are NaN.
    """
    if df.empty or 'time' not in df.columns or 'Talking' not in df.columns:
        return np.array([]), np.array([])

    t = df['time'].to_numpy(dtype=float) + float(role_offset_sec)
    v = df['Talking'].to_numpy(dtype=float)

    if target_hz <= 0:
        return np.array([]), np.array([])
    bin_width = 1.0 / target_hz

    file_end = float(t.max())
    if target_length_sec is None:
        grid_start_sec = float(t.min())
        grid_end_sec = file_end
    else:
        grid_end_sec = file_end
        grid_start_sec = file_end - float(target_length_sec)

    if grid_end_sec <= grid_start_sec:
        return np.array([]), np.array([])

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

    # Output timestamps: 0 = grid_start_sec (= entropy start on the entropy axis)
    timestamps = edges[:-1] - grid_start_sec
    return timestamps, props


# ── Loader ──────────────────────────────────────────────────────────


class SpeakingLoader:
    """Loads speaking-proportion timeseries from data/com_timeseries/."""

    def __init__(
        self,
        com_dir: Path,
        role_offsets_sec: Optional[Dict[str, float]] = None,
        target_hz: float = DEFAULT_TARGET_HZ,
        overlay_window_sec: float = DEFAULT_OVERLAY_WINDOW_SEC,
    ):
        self.com_dir = Path(com_dir)
        self.role_offsets_sec: Dict[str, float] = dict(role_offsets_sec or {})
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
        target_length_sec: Optional[float] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return (timestamps, speaking_proportion) for one role/session, or None.

        If target_length_sec is given, the grid is end-aligned: the file's last
        second is anchored to the entropy end, and output covers exactly
        target_length_sec seconds working backwards. Files shorter than the
        target are NaN-padded at the start.
        """
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
            target_length_sec=target_length_sec,
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
        union_fine_hz: float = 10.0,
        target_length_sec: Optional[float] = None,
    ) -> Optional[Dict[str, object]]:
        """Build an overlay-ready data bundle for one session.

        Computes both per-role and team-union (any-role-speaking) proportions
        at the overlay window. The union is taken at the fine `union_fine_hz`
        grid (default 10 Hz / 100 ms bins) by OR-ing role activity per fine
        sample, then averaging into the wider window — this avoids double-
        counting overlapping speech that summing per-role proportions would
        produce, and is fine-grained enough that brief utterances aren't
        rounded up to a full active second.

        Returns a dict with:
          - 'window_sec': float, the bin width used
          - 'timestamps': np.ndarray of bin-start times (seconds, post-lead-in axis)
          - 'role_props': dict[role -> np.ndarray] mean per-role speaking
            proportion per bin (in [0, 1]; can sum to more than 1 across roles).
          - 'union_prop': np.ndarray of "any role speaking" proportion per bin
            (in [0, 1]); 1 - union_prop is the silence proportion.
          - 'roles_present': list[str] roles that had a file (subset of `roles`).
          - 'roles_missing': list[str] roles with no zoom file for this session.

        Returns None if no roles in `roles` had any data for this session.
        """
        ws = float(window_sec if window_sec is not None else self.overlay_window_sec)
        if ws <= 0:
            return None

        if union_fine_hz <= 0:
            return None
        fine_hz = float(union_fine_hz)

        roles_missing: List[str] = []
        per_role_fine: Dict[str, np.ndarray] = {}
        for role in roles:
            grid = self.load_speaking_grid(
                team=team, day=day, session=session, role=role,
                target_hz=fine_hz,
                target_length_sec=target_length_sec,
            )
            if grid is None:
                roles_missing.append(role)
                continue
            _ts, props = grid
            if len(props) == 0:
                roles_missing.append(role)
                continue
            per_role_fine[role] = props

        if not per_role_fine:
            return None

        # Pad all role arrays to common length with NaN. When end-aligned
        # (target_length_sec given), pad at the START to preserve end-anchor;
        # otherwise pad at the END (start-aligned, legacy behavior).
        fine_n = max(len(arr) for arr in per_role_fine.values())
        pad_at_start = target_length_sec is not None
        for role in list(per_role_fine.keys()):
            arr = per_role_fine[role]
            if len(arr) < fine_n:
                padded = np.full(fine_n, np.nan, dtype=float)
                if pad_at_start:
                    padded[fine_n - len(arr):] = arr
                else:
                    padded[:len(arr)] = arr
                per_role_fine[role] = padded

        # Fine-grain union: 1 if ANY role spoke in this fine bin (any binary
        # sample within the bin was 1). At 10 Hz / 100 ms bins this is fine
        # enough that we can use a strict > 0 test without inflating brief
        # speech. NaN (role missing data) is treated as not speaking.
        speaking_mask = np.zeros(fine_n, dtype=float)
        for arr in per_role_fine.values():
            active = np.where(np.isnan(arr), 0.0, (arr > 0.0).astype(float))
            speaking_mask = np.maximum(speaking_mask, active)

        # Bin to overlay window
        samples_per_window = int(round(ws * fine_hz))
        if samples_per_window <= 0:
            return None
        n_bins = fine_n // samples_per_window
        if n_bins == 0:
            return None

        def _bin_mean(arr: np.ndarray) -> np.ndarray:
            truncated = arr[:n_bins * samples_per_window]
            reshaped = truncated.reshape(n_bins, samples_per_window)
            # All-NaN slices (trailing bins for shorter roles) → NaN; suppress
            # the resulting RuntimeWarning since we expect this and treat NaN
            # downstream as zero contribution.
            with np.errstate(invalid='ignore'):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=RuntimeWarning)
                    return np.nanmean(reshaped, axis=1)

        role_props_binned: Dict[str, np.ndarray] = {
            role: _bin_mean(arr) for role, arr in per_role_fine.items()
        }
        union_prop = _bin_mean(speaking_mask)
        timestamps = np.arange(n_bins, dtype=float) * ws

        return {
            'window_sec': ws,
            'timestamps': timestamps,
            'role_props': role_props_binned,
            'union_prop': union_prop,
            'roles_present': list(role_props_binned.keys()),
            'roles_missing': roles_missing,
        }
