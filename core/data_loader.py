"""
BioTDMS Unified Data Loading Module

Supports:
1. Entropy/AMI aggregated data (CSV) - team & role level
2. Session-level physiological data (Parquet) - role level
3. Subtask event data (Excel) - overlay regions

Key design decisions:
- Entropy Time_Window is an abstract index; we convert to seconds-from-session-start
  using a configurable window_duration (default 1s since data appears ~1Hz)
- Session physio timestamps are Unix epoch; we convert to seconds-from-start for plotting
- Subtask times are clock times; we compute seconds-from-first-subtask-start
- All three share a common x-axis: "seconds from session start"
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import yaml
import pandas as pd
import numpy as np


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SignatureDefinition:
    """Definition of a data signature from YAML config"""
    id: str
    name: str
    description: str
    category: str
    level: str              # "team", "role", or "summary"
    measure_type: str       # "entropy", "ami", "cardiac", "eeg", etc.

    # Data source
    data_source: str        # "entropy_ami" or "session_physio"
    column_template: Optional[str] = None   # e.g. "{role}_{signal}_Entropy"
    column_name: Optional[str] = None       # e.g. "Mean_Entropy_Layers" (fixed)

    # Visualization
    y_label: str = ""
    unit: str = ""
    base_color: str = "#3b82f6"

    # EEG specific
    sensor: Optional[str] = None
    band: Optional[str] = None
    channels: Optional[List[str]] = None

    # Ontology mapping (for future linking)
    ontology_uri: Optional[str] = None
    modality: Optional[str] = None
    technique: Optional[str] = None
    construct: Optional[str] = None


@dataclass
class RoleConfig:
    """Configuration for a team role"""
    id: str
    name: str
    color: str


@dataclass
class TeamScenario:
    """Represents a unified team/scenario combination across data sources.
    
    scenario_id is the canonical key: "Day1_Session2" format.
    entropy_run holds the Run number for entropy CSV lookups (may differ from scenario_id).
    data_source can be "entropy_ami", "session_physio", or "both".
    """
    team_id: str
    scenario_id: str        # Canonical: "Day1_Session2"
    data_source: str        # "entropy_ami", "session_physio", or "both"
    description: str = ""
    # Session-specific
    dce: Optional[str] = None
    day: Optional[str] = None
    session: Optional[str] = None
    session_label: Optional[str] = None
    # Entropy-specific
    entropy_run: Optional[str] = None  # Run number for entropy CSV


@dataclass
class TimeseriesData:
    """Container for a single timeseries trace"""
    label: str              # Display name (e.g. "FOA Entropy (All)")
    timestamps: List[float] # Seconds from session start
    values: List[float]
    color: str = "#3b82f6"
    unit: str = ""
    role: Optional[str] = None
    source_column: str = ""

    @property
    def n_points(self) -> int:
        return len(self.values)


@dataclass
class SubtaskEvent:
    """A subtask event for overlay rendering"""
    subtask_id: int
    subtask_mr: Optional[int]
    start_sec: float        # Seconds from session start
    end_sec: float
    members: str            # e.g. "All", "JTAC,Lead"
    category: Optional[int] = None
    label: str = ""


# =============================================================================
# SIGNATURE REGISTRY
# =============================================================================

class SignatureRegistry:
    """
    Loads and manages signature definitions from YAML config.
    Handles template expansion for role × signal × channel combinations.
    """

    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self._raw: Optional[Dict] = None
        self._signatures: Optional[Dict[str, SignatureDefinition]] = None
        self._roles: Optional[List[RoleConfig]] = None

    @property
    def raw_config(self) -> Dict:
        if self._raw is None:
            with open(self.config_path) as f:
                self._raw = yaml.safe_load(f)
        return self._raw

    @property
    def global_config(self) -> Dict:
        return self.raw_config.get('_config', {})

    @property
    def roles(self) -> List[RoleConfig]:
        if self._roles is None:
            self._roles = []
            for r in self.global_config.get('roles', []):
                self._roles.append(RoleConfig(
                    id=r['id'], name=r['name'], color=r.get('color', '#888')
                ))
        return self._roles

    @property
    def role_ids(self) -> List[str]:
        return [r.id for r in self.roles]

    @property
    def role_colors(self) -> Dict[str, str]:
        return {r.id: r.color for r in self.roles}

    @property
    def signal_types(self) -> List[Dict]:
        return self.global_config.get('signal_types', [])

    @property
    def signal_type_ids(self) -> List[str]:
        return [s['id'] for s in self.signal_types]

    # ---- Parsing ----

    def _parse_signature(self, sig_id: str, d: Dict) -> SignatureDefinition:
        viz = d.get('visualization', {})
        return SignatureDefinition(
            id=sig_id,
            name=d.get('name', sig_id),
            description=d.get('description', ''),
            category=d.get('category', 'Other'),
            level=d.get('level', 'unknown'),
            measure_type=d.get('measure_type', 'unknown'),
            data_source=d.get('data_source', 'entropy_ami'),
            column_template=d.get('column_template'),
            column_name=d.get('column_name'),
            y_label=viz.get('y_label', d.get('name', sig_id)),
            unit=viz.get('unit', ''),
            base_color=viz.get('base_color', viz.get('color', '#3b82f6')),
            sensor=d.get('sensor'),
            band=d.get('band'),
            channels=d.get('channels'),
        )

    def get_all_signatures(self) -> Dict[str, SignatureDefinition]:
        if self._signatures is None:
            self._signatures = {}
            for sig_id, sig_data in self.raw_config.items():
                if sig_id.startswith('_') or not isinstance(sig_data, dict):
                    continue
                self._signatures[sig_id] = self._parse_signature(sig_id, sig_data)
        return self._signatures

    def get_by_id(self, sig_id: str) -> Optional[SignatureDefinition]:
        return self.get_all_signatures().get(sig_id)

    # ---- Filtering ----

    def get_categories(self) -> List[str]:
        return sorted(set(s.category for s in self.get_all_signatures().values()))

    def get_by_category(self, category: str) -> List[SignatureDefinition]:
        return [s for s in self.get_all_signatures().values() if s.category == category]

    def get_by_data_source(self, ds: str) -> List[SignatureDefinition]:
        return [s for s in self.get_all_signatures().values() if s.data_source == ds]

    def get_by_level(self, level: str) -> List[SignatureDefinition]:
        return [s for s in self.get_all_signatures().values() if s.level == level]

    def get_by_measure_type(self, mt: str) -> List[SignatureDefinition]:
        return [s for s in self.get_all_signatures().values() if s.measure_type == mt]

    # ---- Template expansion ----

    def expand_columns(
        self,
        sig: SignatureDefinition,
        roles: Optional[List[str]] = None,
        signal_type: str = "All",
        channel: Optional[str] = None
    ) -> List[Tuple[str, Optional[str], str]]:
        """
        Expand a signature's column_template into concrete (column_name, role, label) tuples.

        Handles templates like:
          - "Team_{signal}_Entropy"       -> [("Team_All_Entropy", None, "Team Entropy (All)")]
          - "{role}_{signal}_Entropy"     -> [("FOA_All_Entropy", "FOA", "FOA Entropy (All)"), ...]
          - "{role}_IBI"                  -> [("FOA_IBI", "FOA", "FOA IBI"), ...]
          - "{role}_alpha_{channel}"      -> [("FOA_alpha_Fp1", "FOA", "FOA Alpha Fp1"), ...]
        
        For fixed column_name signatures, returns that directly.
        """
        # Fixed column (summary signatures)
        if sig.column_name and not sig.column_template:
            return [(sig.column_name, None, sig.name)]

        if not sig.column_template:
            return []

        template = sig.column_template
        results = []
        use_roles = roles or (self.role_ids if '{role}' in template else [None])

        for role in use_roles:
            fmt = {}
            if role:
                fmt['role'] = role
            if '{signal}' in template:
                fmt['signal'] = signal_type
            if '{channel}' in template:
                ch = channel or (sig.channels[0] if sig.channels else 'Fp1')
                fmt['channel'] = ch

            col = template.format(**fmt)
            # Build human label
            parts = []
            if role:
                parts.append(role)
            parts.append(sig.name)
            if '{signal}' in template:
                parts.append(f"({signal_type})")
            if '{channel}' in template:
                parts.append(fmt.get('channel', ''))
            label = " ".join(parts)

            results.append((col, role, label))

        return results


# =============================================================================
# SUBTASK LOADER
# =============================================================================

class SubtaskLoader:
    """
    Loads subtask events from the Excel lookup table.
    Computes seconds-from-session-start from clock times.
    """

    def __init__(self, excel_path: Path):
        self.path = Path(excel_path)
        self._df: Optional[pd.DataFrame] = None

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self._df = self._load()
        return self._df

    def _load(self) -> pd.DataFrame:
        """Load and preprocess the subtask lookup table"""
        df = pd.read_excel(self.path, engine='openpyxl')
        # Columns: Team, Day, Scen, Run, Subtask, Start, End,
        #          Subtask_MR, Member, Index_Start, Index_End, Category
        # Start/End are datetime.time objects
        return df

    def get_subtasks(
        self,
        team: int,
        day: int,
        run: Optional[int] = None,
        session_start_epoch: Optional[float] = None
    ) -> List[SubtaskEvent]:
        """
        Get subtask events for a team/day, with times as seconds-from-session-start.

        If session_start_epoch is provided, we anchor times to it.
        Otherwise we use the earliest subtask Start time as t=0.
        """
        mask = (self.df['Team'] == team) & (self.df['Day'] == day)
        if run is not None:
            mask = mask & (self.df['Run'] == run)

        subset = self.df[mask].copy()
        if subset.empty:
            return []

        # Convert Start/End time objects to seconds-from-midnight
        def time_to_seconds(t) -> Optional[float]:
            if pd.isna(t):
                return None
            if hasattr(t, 'hour'):
                return t.hour * 3600 + t.minute * 60 + t.second
            return None

        subset['start_secs'] = subset['Start'].apply(time_to_seconds)
        subset['end_secs'] = subset['End'].apply(time_to_seconds)
        subset = subset.dropna(subset=['start_secs', 'end_secs'])

        if subset.empty:
            return []

        # Reference time: earliest subtask start
        ref_secs = subset['start_secs'].min()

        events = []
        for _, row in subset.iterrows():
            events.append(SubtaskEvent(
                subtask_id=int(row['Subtask']),
                subtask_mr=int(row['Subtask_MR']) if pd.notna(row.get('Subtask_MR')) else None,
                start_sec=float(row['start_secs'] - ref_secs),
                end_sec=float(row['end_secs'] - ref_secs),
                members=str(row.get('Member', 'Unknown')),
                category=int(row['Category']) if pd.notna(row.get('Category')) else None,
                label=f"ST{int(row['Subtask'])}" + (f" (MR{int(row['Subtask_MR'])})" if pd.notna(row.get('Subtask_MR')) else "")
            ))

        return sorted(events, key=lambda e: e.start_sec)


# =============================================================================
# UNIFIED DATA LOADER
# =============================================================================

class DataLoader:
    """
    Main data loading interface.
    
    Handles both entropy/AMI CSV and session parquet data,
    with a unified seconds-from-start time axis.
    """

    def __init__(self, data_dir: Path, registry: SignatureRegistry):
        self.data_dir = Path(data_dir)
        self.registry = registry
        self._entropy_df: Optional[pd.DataFrame] = None
        self._session_index: Optional[pd.DataFrame] = None
        self._subtask_loader: Optional[SubtaskLoader] = None

    # ---- Entropy/AMI data ----

    @property
    def entropy_csv_path(self) -> Path:
        return self.data_dir / 'team_entropy_ami.csv'

    @property
    def entropy_df(self) -> Optional[pd.DataFrame]:
        if self._entropy_df is None and self.entropy_csv_path.exists():
            df = pd.read_csv(self.entropy_csv_path, dtype={'Session': str})
            # Drop rows where Team is NaN (padding rows)
            df = df.dropna(subset=['Team'])
            df['Team'] = df['Team'].astype(int)
            df['Run'] = df['Run'].astype(int)
            self._entropy_df = df
        return self._entropy_df

    def has_entropy_data(self) -> bool:
        return self.entropy_csv_path.exists()

    # ---- Session data ----

    @property
    def session_root(self) -> Path:
        return self.data_dir / 'processed_sessions'

    def has_session_data(self) -> bool:
        return self.session_root.exists() and any(self.session_root.rglob('*.parquet'))

    @property
    def session_index(self) -> pd.DataFrame:
        """Build index of available session files by scanning the directory"""
        if self._session_index is None:
            rows = []
            if self.session_root.exists():
                for pq in self.session_root.rglob('*.parquet'):
                    if pq.name == 'sessions_index.parquet':
                        continue
                    # Parse: {DCE}/{Team}_{Day}_{Session}.parquet
                    dce = pq.parent.name
                    parts = pq.stem.split('_')
                    if len(parts) >= 3:
                        team = parts[0]     # "Team1"
                        day = parts[1]      # "Day1"
                        session = '_'.join(parts[2:])  # "Session2"
                        rows.append({
                            'dce': dce, 'team': team, 'day': day,
                            'session': session, 'path': str(pq)
                        })
            self._session_index = pd.DataFrame(rows)
        return self._session_index

    def load_session_df(
        self,
        dce: str, team: str, day: str, session: str,
        columns: Optional[List[str]] = None
    ) -> Optional[pd.DataFrame]:
        """Load a session parquet file"""
        path = self.session_root / dce / f"{team}_{day}_{session}.parquet"
        if not path.exists():
            return None
        try:
            meta_cols = ['dce', 'team', 'day', 'session', 'scenario_name',
                         'session_time', 'time_stamp']
            if columns:
                all_cols = meta_cols + [c for c in columns if c not in meta_cols]
                try:
                    return pd.read_parquet(path, columns=all_cols)
                except Exception:
                    df = pd.read_parquet(path)
                    avail = [c for c in all_cols if c in df.columns]
                    return df[avail]
            return pd.read_parquet(path)
        except Exception as e:
            print(f"Error loading session {path}: {e}")
            return None

    # ---- Subtask data ----

    @property
    def subtask_loader(self) -> Optional[SubtaskLoader]:
        if self._subtask_loader is None:
            # Look for subtask files
            for pattern in ['SubTask_LookupTable*.xlsx', 'subtask*.xlsx']:
                files = list(self.data_dir.glob(pattern))
                if files:
                    self._subtask_loader = SubtaskLoader(files[0])
                    break
        return self._subtask_loader

    def get_subtasks(
        self,
        team_id: str,
        day: Optional[str] = None,
        run: Optional[int] = None,
        session_start_epoch: Optional[float] = None
    ) -> List[SubtaskEvent]:
        """Get subtask events for a team"""
        if self.subtask_loader is None:
            return []
        try:
            team_int = int(team_id.replace('Team', ''))
            day_int = int(day.replace('Day', '')) if day else 1
            return self.subtask_loader.get_subtasks(
                team=team_int, day=day_int, run=run,
                session_start_epoch=session_start_epoch
            )
        except Exception as e:
            print(f"Error loading subtasks: {e}")
            return []

    # ---- Team/scenario discovery ----

    def get_available_teams_scenarios(self) -> List[TeamScenario]:
        """
        Get unified team/scenario combinations across both data sources.

        Returns one entry per (team, day, session) regardless of whether data
        exists in entropy/AMI, session physio, or both.  The unified
        TeamScenario carries all the fields needed by either loader:
          - scenario_id = "Day1_Session2" (canonical key)
          - entropy_run = Run number (for entropy CSV lookups)
          - dce / day / session (for session parquet lookups)
          - data_source = "both" | "entropy_ami" | "session_physio"
        """
        # Build lookup: (team_num_str, day_session_label) -> info dict
        unified: Dict[Tuple[str, str], dict] = {}

        # ---- Entropy/AMI ----
        if self.entropy_df is not None:
            for _, row in self.entropy_df[['Team', 'Run']].drop_duplicates().iterrows():
                tid = str(int(row['Team']))
                run = int(row['Run'])
                # Get Session column for day/session linking
                mask = (
                    (self.entropy_df['Team'] == int(row['Team'])) &
                    (self.entropy_df['Run'] == run)
                )
                session_vals = self.entropy_df.loc[mask, 'Session'].dropna().unique()
                session_label = str(session_vals[0]) if len(session_vals) > 0 else ""

                if session_label:
                    key = (tid, session_label)
                else:
                    # No Session column — use Run as fallback key
                    key = (tid, f"Run{run}")

                entry = unified.setdefault(key, {
                    'team_id': tid, 'session_label': session_label,
                    'has_entropy': False, 'has_session': False,
                    'entropy_run': None, 'dce': None, 'day': None, 'session': None,
                })
                entry['has_entropy'] = True
                entry['entropy_run'] = str(run)

                # Try to parse day/session from session_label (e.g. "Day1_Session2")
                if session_label and not entry['day']:
                    parts = session_label.split('_')
                    if len(parts) >= 2 and parts[0].startswith('Day'):
                        entry['day'] = parts[0]
                        entry['session'] = '_'.join(parts[1:])

        # ---- Session physio ----
        if not self.session_index.empty:
            for _, row in self.session_index.iterrows():
                tid = row['team'].replace('Team', '')
                day_session = f"{row['day']}_{row['session']}"
                key = (tid, day_session)

                entry = unified.setdefault(key, {
                    'team_id': tid, 'session_label': day_session,
                    'has_entropy': False, 'has_session': False,
                    'entropy_run': None, 'dce': None, 'day': None, 'session': None,
                })
                entry['has_session'] = True
                entry['dce'] = row['dce']
                entry['day'] = row['day']
                entry['session'] = row['session']
                # Keep session_label consistent
                if not entry['session_label'] or entry['session_label'].startswith('Run'):
                    entry['session_label'] = day_session

        # ---- Build unified TeamScenario list ----
        results = []
        for (tid, _label), info in sorted(unified.items()):
            session_label = info['session_label']
            scenario_id = session_label  # canonical: "Day1_Session2"

            # Determine data_source flag
            if info['has_entropy'] and info['has_session']:
                data_source = 'both'
            elif info['has_entropy']:
                data_source = 'entropy_ami'
            else:
                data_source = 'session_physio'

            # Build description
            parts = [f"Team {tid}"]
            if info['day']:
                parts.append(f"{info['day']} {info['session'] or ''}")
            if info['entropy_run']:
                parts.append(f"Run {info['entropy_run']}")
            if info['dce']:
                parts.append(f"({info['dce']})")
            # Source indicator
            src_tag = {'both': '📊🧠', 'entropy_ami': '📊', 'session_physio': '🧠'}
            parts.append(src_tag.get(data_source, ''))

            results.append(TeamScenario(
                team_id=tid,
                scenario_id=scenario_id,
                data_source=data_source,
                description=' '.join(parts).strip(),
                dce=info['dce'],
                day=info['day'],
                session=info['session'],
                session_label=session_label,
                entropy_run=info['entropy_run'],
            ))

        return results

    def get_entropy_run_for_scenario(self, team_id: str, scenario: TeamScenario) -> Optional[str]:
        """
        Resolve the entropy Run number for a unified scenario.
        Looks up the Run from the entropy CSV using the session label or day info.
        """
        if self.entropy_df is None:
            return None

        df = self.entropy_df[self.entropy_df['Team'] == int(team_id)]
        if df.empty:
            return None

        # Try matching via Session column
        if scenario.session_label and 'Session' in df.columns:
            match = df[df['Session'] == scenario.session_label]['Run'].unique()
            if len(match) > 0:
                return str(int(match[0]))

        # Try matching via day/session parsed from Session column
        if scenario.day and scenario.session and 'Session' in df.columns:
            target = f"{scenario.day}_{scenario.session}"
            match = df[df['Session'] == target]['Run'].unique()
            if len(match) > 0:
                return str(int(match[0]))

        return None

    def find_matching_session(self, team_id: str, entropy_session_label: str) -> Optional[TeamScenario]:
        """
        Given an entropy team+session label (e.g. "Day1_Session2"),
        find the matching session parquet.
        """
        if self.session_index.empty:
            return None
        parts = entropy_session_label.split('_')
        if len(parts) >= 2:
            day = parts[0]
            session = '_'.join(parts[1:])
            team_str = f"Team{team_id}"
            match = self.session_index[
                (self.session_index['team'] == team_str) &
                (self.session_index['day'] == day) &
                (self.session_index['session'] == session)
            ]
            if not match.empty:
                row = match.iloc[0]
                return TeamScenario(
                    team_id=team_id,
                    scenario_id=f"{day}_{session}",
                    data_source='session_physio',
                    dce=row['dce'], day=day, session=session,
                    description=f"Team {team_id} {day} {session}"
                )
        return None

    # ---- Timeseries loading ----

    def load_entropy_timeseries(
        self,
        sig: SignatureDefinition,
        team_id: str,
        run_id: str,
        roles: Optional[List[str]] = None,
        signal_type: str = "All",
        session_start_epoch: Optional[float] = None,
        window_duration: float = 1.0
    ) -> List[TimeseriesData]:
        """
        Load timeseries from entropy/AMI CSV.
        
        Returns list of TimeseriesData (one per expanded role, or one for team/summary).
        Time axis = Time_Window * window_duration (seconds from start).
        If session_start_epoch is known, we could anchor, but for now
        we just use relative seconds.
        """
        if self.entropy_df is None:
            return []

        df = self.entropy_df[
            (self.entropy_df['Team'] == int(team_id)) &
            (self.entropy_df['Run'] == int(run_id))
        ]
        if df.empty:
            return []

        # Time axis: Time_Window * window_duration
        timestamps = (df['Time_Window'] * window_duration).tolist()

        # Expand columns
        expanded = self.registry.expand_columns(
            sig, roles=roles, signal_type=signal_type
        )

        results = []
        for col, role, label in expanded:
            if col in df.columns:
                vals = df[col].tolist()
                color = self.registry.role_colors.get(role, sig.base_color) if role else sig.base_color
                results.append(TimeseriesData(
                    label=label,
                    timestamps=timestamps,
                    values=vals,
                    color=color,
                    unit=sig.unit,
                    role=role,
                    source_column=col
                ))

        return results

    def load_session_timeseries(
        self,
        sig: SignatureDefinition,
        team_scenario: TeamScenario,
        roles: Optional[List[str]] = None,
        channel: Optional[str] = None
    ) -> List[TimeseriesData]:
        """
        Load timeseries from session parquet.
        Time axis = seconds from session start (epoch - min_epoch).
        """
        if not team_scenario.dce or not team_scenario.day or not team_scenario.session:
            return []

        team_str = f"Team{team_scenario.team_id}"

        # Figure out which columns we need
        expanded = self.registry.expand_columns(
            sig, roles=roles, channel=channel
        )
        needed_cols = [col for col, _, _ in expanded]

        df = self.load_session_df(
            dce=team_scenario.dce,
            team=team_str,
            day=team_scenario.day,
            session=team_scenario.session,
            columns=needed_cols
        )
        if df is None or df.empty:
            return []

        # Time axis: seconds from start
        if 'time_stamp' in df.columns:
            t0 = df['time_stamp'].min()
            timestamps = (df['time_stamp'] - t0).tolist()
        else:
            timestamps = list(range(len(df)))

        results = []
        for col, role, label in expanded:
            if col in df.columns:
                vals = df[col].tolist()
                color = self.registry.role_colors.get(role, sig.base_color) if role else sig.base_color
                results.append(TimeseriesData(
                    label=label,
                    timestamps=timestamps,
                    values=vals,
                    color=color,
                    unit=sig.unit,
                    role=role,
                    source_column=col
                ))

        return results

    def load_timeseries(
        self,
        sig: SignatureDefinition,
        team_id: str,
        scenario_id: str,
        team_scenario: Optional[TeamScenario] = None,
        roles: Optional[List[str]] = None,
        signal_type: str = "All",
        channel: Optional[str] = None
    ) -> List[TimeseriesData]:
        """
        Unified timeseries loader. Routes to the appropriate source.
        Returns list of traces (one per role for role-level, or one for team/summary).
        """
        if sig.data_source == 'entropy_ami':
            # Resolve the entropy Run number from the unified scenario
            run_id = None
            if team_scenario and hasattr(team_scenario, 'entropy_run') and team_scenario.entropy_run:
                run_id = team_scenario.entropy_run
            if not run_id:
                # Try to resolve from the scenario
                run_id = self.get_entropy_run_for_scenario(team_id, team_scenario) if team_scenario else None
            if not run_id:
                # scenario_id might already be a Run number (legacy/direct)
                try:
                    int(scenario_id)
                    run_id = scenario_id
                except (ValueError, TypeError):
                    run_id = None
            if not run_id:
                return []
            return self.load_entropy_timeseries(
                sig, team_id, run_id,
                roles=roles, signal_type=signal_type
            )
        elif sig.data_source == 'session_physio':
            if team_scenario is None:
                # Try to find matching session
                ts_list = self.get_available_teams_scenarios()
                for ts in ts_list:
                    if (ts.team_id == team_id and
                        ts.scenario_id == scenario_id and
                        ts.data_source == 'session_physio'):
                        team_scenario = ts
                        break
                if team_scenario is None:
                    return []
            return self.load_session_timeseries(
                sig, team_scenario, roles=roles, channel=channel
            )
        return []

    def get_session_start_epoch(self, team_scenario: TeamScenario) -> Optional[float]:
        """Get the min timestamp for a session (for subtask alignment)"""
        if not team_scenario.dce:
            return None
        df = self.load_session_df(
            dce=team_scenario.dce,
            team=f"Team{team_scenario.team_id}",
            day=team_scenario.day,
            session=team_scenario.session,
            columns=[]  # just metadata
        )
        if df is not None and 'time_stamp' in df.columns:
            return float(df['time_stamp'].min())
        return None


# =============================================================================
# FACTORY
# =============================================================================

def create_data_loader(app_dir: Path) -> Tuple[Optional[SignatureRegistry], Optional[DataLoader]]:
    """Factory function to create registry and loader."""
    config_path = app_dir / 'config' / 'signatures.yaml'
    data_dir = app_dir / 'data'

    if not config_path.exists():
        return None, None
    try:
        registry = SignatureRegistry(config_path)
        loader = DataLoader(data_dir, registry)
        return registry, loader
    except Exception as e:
        print(f"Error creating data loader: {e}")
        return None, None
