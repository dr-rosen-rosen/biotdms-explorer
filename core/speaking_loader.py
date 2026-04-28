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
        union_fine_hz: float = 10.0,
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

        # Pad all role arrays to common length (longest role) with NaN
        fine_n = max(len(arr) for arr in per_role_fine.values())
        for role in list(per_role_fine.keys()):
            arr = per_role_fine[role]
            if len(arr) < fine_n:
                padded = np.full(fine_n, np.nan, dtype=float)
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
