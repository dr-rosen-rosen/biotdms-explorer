"""
Use Case 2: Analysis Results View

Layout:
  1. TEAM PANEL (top) — team-level signatures (entropy/AMI) with "All" subtask overlays
  2. ROLE PANELS — one per role, each with that role's data traces + role-specific subtask overlays
  3. Signature cards + evidence/interpretation

Subtask overlay logic:
  - "All" member subtasks appear on the team panel AND all role panels
  - Role-specific subtasks (e.g. "JTAC,Lead") appear only on matching role panels
"""

import streamlit as st
from typing import List, Optional, Dict, Tuple
from pathlib import Path
import sys

# Path setup
_current_file = Path(__file__).resolve() if '__file__' in dir() else Path.cwd() / 'views' / 'uc2_analysis.py'
_app_dir = _current_file.parent.parent
if str(_app_dir) not in sys.path:
    sys.path.insert(0, str(_app_dir))

from core.ontology import OntologyAccess, Measure
from views.uc2_signature_selection import SelectedSignature, TeamSelection

try:
    from core.data_loader import (
        SignatureRegistry, DataLoader, TimeseriesData,
        SubtaskEvent, TeamScenario, create_data_loader
    )
    DATA_LOADER_AVAILABLE = True
except ImportError:
    DATA_LOADER_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from core.subtask_construct_map import (
        SubtaskConstructMap, load_subtask_construct_map, SubtaskProfile
    )
    CONSTRUCT_MAP_AVAILABLE = True
except ImportError:
    CONSTRUCT_MAP_AVAILABLE = False

try:
    from core.measure_neighborhood import render_measure_neighborhood
    NEIGHBORHOOD_VIZ_AVAILABLE = True
except ImportError:
    NEIGHBORHOOD_VIZ_AVAILABLE = False


# =============================================================================
# CONSTANTS
# =============================================================================

SUBTASK_COLORS = {
    1: "rgba(59,130,246,0.15)",    # blue
    2: "rgba(16,185,129,0.15)",    # green
    3: "rgba(245,158,11,0.15)",    # amber
    4: "rgba(139,92,246,0.15)",    # purple
}
SUBTASK_BORDER = {
    1: "rgba(59,130,246,0.5)",
    2: "rgba(16,185,129,0.5)",
    3: "rgba(245,158,11,0.5)",
    4: "rgba(139,92,246,0.5)",
}

MODALITY_ICONS = {
    'cardiac': '❤️', 'ekg': '❤️',
    'ocular': '👁️', 'pupil': '👁️',
    'respiratory': '🌬️', 'resp': '🌬️',
    'eeg': '🧠',
    'entropy': '📊', 'ami': '🔗',
    'team': '👥', 'summary': '📊',
    'default': '📈'
}

ROLE_ICONS = {
    'FOA': '🎯', 'FOM': '🔥', 'FSO': '📡', 'JTAC': '✈️', 'Lead': '⭐'
}

CARD_COLORS = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899']


def get_icon(category: str) -> str:
    if not category:
        return MODALITY_ICONS['default']
    cat = category.lower()
    for key, icon in MODALITY_ICONS.items():
        if key in cat:
            return icon
    return MODALITY_ICONS['default']


def _is_entropy_ami(sig_def) -> bool:
    """Check if a signature is entropy or AMI (eligible for rolling stats)"""
    if sig_def is None:
        return False
    if getattr(sig_def, 'data_source', '') == 'entropy_ami':
        return True
    mt = (getattr(sig_def, 'measure_type', '') or '').lower()
    return mt in ('entropy', 'ami')


def _get_signal_types(sel_sig) -> List[str]:
    """Get the list of signal types from a SelectedSignature.
    
    Supports both the new signal_types list and legacy single signal_type.
    """
    # Try new multi-select field first
    types = getattr(sel_sig, 'signal_types', None)
    if not types and isinstance(sel_sig, dict):
        types = sel_sig.get('signal_types')
    if types and isinstance(types, list) and len(types) > 0:
        return types
    # Fall back to legacy single field
    st_ = getattr(sel_sig, 'signal_type', None)
    if not st_ and isinstance(sel_sig, dict):
        st_ = sel_sig.get('signal_type')
    return [st_ or "All"]


def _add_rolling_stats(
    fig: go.Figure,
    ts: 'TimeseriesData',
    window: int,
    n_sd: float,
    row: Optional[int] = None,
    col: Optional[int] = None
):
    """
    Add rolling mean ± SD band and exceedance markers for a single trace.
    
    Args:
        fig: Plotly figure to add to
        ts: TimeseriesData with timestamps and values
        window: Rolling window size in data points
        n_sd: Number of standard deviations for the band
        row/col: Subplot row/col (None for simple figures)
    """
    import numpy as np
    import pandas as pd

    series = pd.Series(ts.values, dtype=float)
    if len(series.dropna()) < window:
        return  # Not enough data for the window

    rolling_mean = series.rolling(window, center=True, min_periods=max(1, window // 4)).mean()
    rolling_std = series.rolling(window, center=True, min_periods=max(1, window // 4)).std()
    upper = rolling_mean + n_sd * rolling_std
    lower = rolling_mean - n_sd * rolling_std

    timestamps = list(ts.timestamps)

    # Parse the trace color to derive a band color
    band_color = _derive_band_color(ts.color, opacity=0.08)
    mean_color = _derive_band_color(ts.color, opacity=0.5)

    add_kwargs = dict(row=row, col=col) if row is not None else {}

    # Upper bound (invisible line)
    fig.add_trace(
        go.Scatter(
            x=timestamps, y=upper.tolist(),
            mode='lines', line=dict(width=0),
            showlegend=False, hoverinfo='skip'
        ),
        **add_kwargs
    )

    # Lower bound with fill to upper
    fig.add_trace(
        go.Scatter(
            x=timestamps, y=lower.tolist(),
            mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor=band_color,
            showlegend=False, hoverinfo='skip'
        ),
        **add_kwargs
    )

    # Rolling mean dashed line
    fig.add_trace(
        go.Scatter(
            x=timestamps, y=rolling_mean.tolist(),
            mode='lines', line=dict(color=mean_color, width=1.5, dash='dash'),
            name=f"μ ({ts.label})" if len(ts.label) < 20 else "Rolling μ",
            showlegend=False,
            hovertemplate="Rolling mean: %{y:.3f}<extra></extra>"
        ),
        **add_kwargs
    )

    # Exceedance markers
    exceed_mask = (series > upper) | (series < lower)
    exceed_idx = exceed_mask[exceed_mask].index
    if len(exceed_idx) > 0:
        ex_x = [timestamps[i] for i in exceed_idx if i < len(timestamps)]
        ex_y = [ts.values[i] for i in exceed_idx if i < len(ts.values)]
        if ex_x:
            fig.add_trace(
                go.Scatter(
                    x=ex_x, y=ex_y,
                    mode='markers',
                    marker=dict(color='red', size=4, symbol='circle', opacity=0.7),
                    name=f"Exceed ±{n_sd}σ",
                    showlegend=False,
                    hovertemplate=(
                        f"<b>Exceeds ±{n_sd}σ</b><br>"
                        "Time: %{x:.0f}s<br>"
                        "Value: %{y:.3f}<extra></extra>"
                    )
                ),
                **add_kwargs
            )


def _derive_band_color(hex_or_named: str, opacity: float = 0.1) -> str:
    """Convert a color to rgba with given opacity for the band fill"""
    color = hex_or_named or '#3b82f6'
    if color.startswith('#') and len(color) == 7:
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        return f'rgba({r},{g},{b},{opacity})'
    # Fallback
    return f'rgba(100,100,100,{opacity})'


def _render_rolling_stats_controls(key_prefix: str = "rolling") -> dict:
    """Render the rolling stats slider controls. Returns settings dict."""
    with st.expander("📏 Contextual Interpretation - Moving Window Settings", expanded=False):
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            enabled = st.checkbox(
                "Show rolling bands",
                value=True,
                key=f"{key_prefix}_enabled"
            )
        with c2:
            window_min = st.slider(
                "Window (minutes)",
                min_value=1, max_value=15, value=5, step=1,
                key=f"{key_prefix}_window",
                disabled=not enabled
            )
        with c3:
            n_sd = st.slider(
                "Threshold (±SD)",
                min_value=0.5, max_value=3.0, value=1.0, step=0.25,
                key=f"{key_prefix}_sd",
                disabled=not enabled
            )
    return {
        'enabled': enabled,
        'window': window_min * 60,  # convert minutes to data points (1Hz)
        'n_sd': n_sd
    }


# =============================================================================
# SUBTASK FILTERING
# =============================================================================

def filter_subtasks_for_team(subtasks: List[SubtaskEvent]) -> List[SubtaskEvent]:
    """Get subtasks where Member == 'All' (whole-team events)"""
    return [e for e in subtasks if e.members.strip().lower() == 'all']


def filter_subtasks_for_role(subtasks: List[SubtaskEvent], role: str) -> List[SubtaskEvent]:
    """
    Get subtasks relevant to a specific role:
    - 'All' subtasks always apply
    - Role-specific subtasks apply if role is in the member list
    """
    results = []
    for e in subtasks:
        members = e.members.strip()
        if members.lower() == 'all':
            results.append(e)
        else:
            member_list = [m.strip() for m in members.split(',')]
            if role in member_list:
                results.append(e)
    return results


def resolve_subtasks_for_panel(
    subtasks: List[SubtaskEvent],
    mode: str,
    panel_type: str = 'team',
    role: Optional[str] = None
) -> List[SubtaskEvent]:
    """
    Resolve which subtasks to display based on user-selected mode.

    Args:
        subtasks: All loaded subtask events
        mode: 'none', 'team', 'role', 'all'
        panel_type: 'team' or 'role' (which panel we're rendering)
        role: Role ID (only used when panel_type='role')

    Returns:
        Filtered list of subtask events for this panel
    """
    if mode == 'none' or not subtasks:
        return []

    if mode == 'team':
        # Show only team-wide (Member=All) subtasks on all panels
        return filter_subtasks_for_team(subtasks)

    if mode == 'role':
        if panel_type == 'team':
            # Team panel: show team-wide subtasks only
            return filter_subtasks_for_team(subtasks)
        else:
            # Role panel: show team-wide + this role's specific subtasks
            return filter_subtasks_for_role(subtasks, role) if role else []

    if mode == 'all':
        if panel_type == 'team':
            return list(subtasks)  # everything
        else:
            # Role panel: show team-wide + this role's specific subtasks
            return filter_subtasks_for_role(subtasks, role) if role else list(subtasks)

    return []


# =============================================================================
# MAIN ENTRY
# =============================================================================

def render_analysis_results(
    onto: OntologyAccess,
    team: Optional[TeamSelection],
    signatures: List[SelectedSignature]
):
    """Main analysis view with team + per-role layout"""

    st.header("📊 Performance Analysis")

    if team:
        st.success(f"🎯 **{team.description}**")
    else:
        st.info("📚 Evidence-only mode (no team data selected)")
        _render_evidence(onto, signatures)
        return

    st.markdown("---")

    config_path = _app_dir / "config" / "signatures.yaml"
    data_dir = _app_dir / "data"

    can_plot = (
        config_path.exists() and
        DATA_LOADER_AVAILABLE and
        PLOTLY_AVAILABLE and
        any(s.data_signature_id for s in signatures)
    )

    if can_plot:
        _render_full_analysis(config_path, data_dir, team, signatures, onto)
    else:
        _render_placeholder()
        _render_signature_cards(signatures)
        _render_evidence(onto, signatures)


# =============================================================================
# FULL ANALYSIS LAYOUT
# =============================================================================

def _render_full_analysis(
    config_path: Path,
    data_dir: Path,
    team: TeamSelection,
    signatures: List[SelectedSignature],
    onto: OntologyAccess
):
    """Team panel + per-role panels"""

    try:
        registry = SignatureRegistry(config_path)
        loader = DataLoader(data_dir, registry)
    except Exception as e:
        st.error(f"Error loading config: {e}")
        return

    # Resolve full TeamScenario
    team_scenario = _resolve_team_scenario(team, loader)

    # Load subtask events
    subtasks = _load_subtasks(team, team_scenario, loader)

    # =============================================
    # SUBTASK OVERLAY CONTROLS
    # =============================================
    subtask_mode = 'none'
    if subtasks:
        n_team = len(filter_subtasks_for_team(subtasks))
        n_role_specific = len(subtasks) - n_team
        st.caption(f"📌 {len(subtasks)} subtask events available "
                   f"({n_team} whole-team, {n_role_specific} role-specific)")

        subtask_mode = st.radio(
            "Subtask overlays",
            ['none', 'team', 'role', 'all'],
            format_func=lambda m: {
                'none': '🚫 None',
                'team': '👥 Team-wide only',
                'role': '👤 Role-specific (per panel)',
                'all': '📋 All subtasks',
            }[m],
            index=0,
            horizontal=True,
            key="subtask_overlay_mode"
        )

    # =============================================
    # SPEAKING ACTIVITY OVERLAY
    # =============================================
    speaking_overlay_data = None
    speaking_overlay_mode = 'none'
    if loader.has_speaking_data() and team_scenario and team_scenario.day and team_scenario.session:
        team_str = f"Team{team.team_id}"
        speaking_roles_present = loader.speaking_loader.available_roles(
            team_str, team_scenario.day, team_scenario.session
        )
        if speaking_roles_present:
            n_roles = len(speaking_roles_present)
            n_total = len(registry.role_ids)
            roles_missing = [r for r in registry.role_ids if r not in speaking_roles_present]
            missing_note = f" — missing: {', '.join(roles_missing)}" if roles_missing else ""
            st.caption(
                f"🎤 Speaking activity available ({n_roles}/{n_total} roles{missing_note})"
            )
            speaking_overlay_mode = st.radio(
                "Speaking activity overlay",
                ['none', 'stacked', 'heat', 'lanes'],
                format_func=lambda m: {
                    'none': '🚫 None',
                    'stacked': '📊 Stacked proportion (per-role, by minute)',
                    'heat': '🌡️ Heat strip (dominant speaker, by minute)',
                    'lanes': '🪧 Per-role lanes (spoke / didn\'t, by minute)',
                }[m],
                index=0,
                horizontal=True,
                key="speaking_overlay_mode",
            )
            if speaking_overlay_mode != 'none':
                speaking_overlay_data = loader.speaking_loader.load_session_overlay(
                    team=team_str,
                    day=team_scenario.day,
                    session=team_scenario.session,
                    roles=registry.role_ids,
                )

    # =============================================
    # CONSTRUCT DEMAND MAP
    # =============================================
    construct_map = None
    construct_view = 'none'
    if CONSTRUCT_MAP_AVAILABLE and subtasks:
        config_dir = config_path.parent
        construct_map = load_subtask_construct_map(config_dir)
        if construct_map:
            st.caption(f"🧠 Construct demand mapping available "
                       f"({len(construct_map.all_constructs())} constructs across "
                       f"{len([p for p in [construct_map.get_profile(i) for i in range(1,10)] if p])} subtasks)")
            construct_view = st.radio(
                "Construct demands",
                ['none', 'focused', 'full'],
                format_func=lambda m: {
                    'none': '🚫 None',
                    'focused': '🎯 Focused (signature-relevant)',
                    'full': '📋 Full (all constructs)',
                }[m],
                index=0,
                horizontal=True,
                key="construct_view_mode"
            )

    # Classify selected signatures
    data_sigs = [s for s in signatures if s.data_signature_id]
    team_sigs = []       # team-level + summary
    role_entropy_sigs = []  # role-level entropy/AMI (show in team panel, all roles overlaid)
    session_role_sigs = []  # role-level session physio (show in per-role panels)

    for sel in data_sigs:
        sig_def = registry.get_by_id(sel.data_signature_id)
        if sig_def is None:
            continue
        if sig_def.level in ('team', 'summary'):
            team_sigs.append((sel, sig_def))
        elif sig_def.level == 'role' and sig_def.data_source == 'entropy_ami':
            role_entropy_sigs.append((sel, sig_def))
        elif sig_def.level == 'role' and sig_def.data_source == 'session_physio':
            session_role_sigs.append((sel, sig_def))

    # =============================================
    # SECTION 1: TEAM-LEVEL PANEL
    # =============================================
    has_team_content = team_sigs or role_entropy_sigs
    if has_team_content:
        st.markdown("### 👥 Team-Level View")
        team_subtasks = resolve_subtasks_for_panel(
            subtasks, subtask_mode, panel_type='team'
        )
        if team_subtasks:
            st.caption(f"Showing {len(team_subtasks)} subtask overlay(s)")

        # Rolling stats controls (shared across all entropy/AMI plots)
        has_entropy_ami = any(
            _is_entropy_ami(sd) for _, sd in team_sigs + role_entropy_sigs
        )
        rolling_cfg = None
        if has_entropy_ami:
            rolling_cfg = _render_rolling_stats_controls(key_prefix="team_rolling")

        # Team / summary signatures — all signal types overlaid on one chart
        SIGNAL_TYPE_COLORS = {
            'All': '#3b82f6',    # blue
            'Neuro': '#8b5cf6',  # purple
            'Auto': '#10b981',   # green
        }

        for sel_sig, sig_def in team_sigs:
            sig_types = _get_signal_types(sel_sig)
            all_traces = []
            multi_signal = len(sig_types) > 1
            for st_ in sig_types:
                traces = loader.load_timeseries(
                    sig=sig_def,
                    team_id=team.team_id,
                    scenario_id=team.scenario_id,
                    team_scenario=team_scenario,
                    signal_type=st_
                )
                # When overlaying multiple signal types, give each a distinct color
                if multi_signal:
                    color = SIGNAL_TYPE_COLORS.get(st_, sig_def.base_color)
                    for t in traces:
                        t.color = color
                all_traces.extend(traces)
            if all_traces:
                icon = get_icon(sig_def.category)
                type_label = ", ".join(sig_types) if sig_types != ["All"] else ""
                st.markdown(f"#### {icon} {sig_def.name}")
                if type_label:
                    st.caption(f"Signal types: {type_label}")
                r_cfg = rolling_cfg if _is_entropy_ami(sig_def) else None
                fig = _build_plot(all_traces, sig_def, team_subtasks, rolling_cfg=r_cfg,
                                  construct_map=construct_map, construct_view=construct_view,
                                  signature_construct=sel_sig.construct,
                                  speaking_overlay_data=speaking_overlay_data,
                                  speaking_overlay_mode=speaking_overlay_mode,
                                  role_colors=registry.role_colors,
                                  role_order=registry.role_ids)
                st.plotly_chart(fig, use_container_width=True,
                                key=f"team_{sel_sig.data_signature_id}")
                _render_inline_evidence(sel_sig, onto)
            else:
                st.warning(f"No data for **{sig_def.name}**")

        # Role-level entropy/AMI (all roles overlaid in team panel)
        # Separate chart per signal type since each has 5 role traces
        for sel_sig, sig_def in role_entropy_sigs:
            sig_types = _get_signal_types(sel_sig)
            for st_ in sig_types:
                traces = loader.load_timeseries(
                    sig=sig_def,
                    team_id=team.team_id,
                    scenario_id=team.scenario_id,
                    team_scenario=team_scenario,
                    roles=registry.role_ids,
                    signal_type=st_
                )
                if traces:
                    icon = get_icon(sig_def.category)
                    st.markdown(f"#### {icon} {sig_def.name} — All Roles ({st_})")
                    fig = _build_plot(traces, sig_def, team_subtasks, rolling_cfg=rolling_cfg,
                                      construct_map=construct_map, construct_view=construct_view,
                                      signature_construct=sel_sig.construct,
                                      speaking_overlay_data=speaking_overlay_data,
                                      speaking_overlay_mode=speaking_overlay_mode,
                                      role_colors=registry.role_colors,
                                      role_order=registry.role_ids)
                    st.plotly_chart(fig, use_container_width=True,
                                    key=f"teamrole_{sel_sig.data_signature_id}_{st_}")
            # Evidence once per signature (after all signal-type charts)
            _render_inline_evidence(sel_sig, onto)

    # =============================================
    # SECTION 2: PER-ROLE PANELS
    # =============================================
    if session_role_sigs:
        st.markdown("---")
        st.markdown("### 👤 Individual Role Analysis")

        for role in registry.roles:
            role_subtasks = resolve_subtasks_for_panel(
                subtasks, subtask_mode, panel_type='role', role=role.id
            )

            # Gather traces for this role across all selected session sigs
            role_traces_by_sig: List[Tuple[str, SignatureDefinition, List[TimeseriesData]]] = []

            for sel_sig, sig_def in session_role_sigs:
                channel = sel_sig.channel
                traces = loader.load_timeseries(
                    sig=sig_def,
                    team_id=team.team_id,
                    scenario_id=team.scenario_id,
                    team_scenario=team_scenario,
                    roles=[role.id],
                    channel=channel
                )
                if traces:
                    role_traces_by_sig.append((sig_def.name, sig_def, traces))

            if not role_traces_by_sig:
                continue

            role_icon = ROLE_ICONS.get(role.id, '👤')
            with st.expander(
                f"{role_icon} **{role.id}** — {role.name}",
                expanded=True
            ):
                caption_parts = [f"{len(role_traces_by_sig)} signature(s)"]
                if role_subtasks:
                    n_all = len([s for s in role_subtasks if s.members.strip().lower() == 'all'])
                    n_role = len(role_subtasks) - n_all
                    caption_parts.append(
                        f"{len(role_subtasks)} subtask regions "
                        f"({n_all} team-wide + {n_role} role-specific)"
                    )
                st.caption(" · ".join(caption_parts))

                if len(role_traces_by_sig) == 1:
                    sig_name, sig_def, traces = role_traces_by_sig[0]
                    for t in traces:
                        t.color = role.color
                    # Find matching SelectedSignature for construct info and evidence
                    matching_sel = next(
                        (s for s, sd in session_role_sigs if sd.name == sig_name),
                        None
                    )
                    fig = _build_plot(traces, sig_def, role_subtasks, height=350,
                                      construct_map=construct_map, construct_view=construct_view,
                                      signature_construct=matching_sel.construct if matching_sel else None)
                    st.plotly_chart(fig, use_container_width=True,
                                   key=f"role_{role.id}_{sig_name}")
                    if matching_sel:
                        _render_inline_evidence(matching_sel, onto)
                else:
                    fig = _build_multi_sig_subplot(
                        role_traces_by_sig, role, role_subtasks
                    )
                    st.plotly_chart(fig, use_container_width=True,
                                   key=f"role_{role.id}_multi")
                    # Render evidence for each signature in the subplot
                    for sig_name, sig_def, _ in role_traces_by_sig:
                        matching_sel = next(
                            (s for s, sd in session_role_sigs if sd.name == sig_name),
                            None
                        )
                        if matching_sel:
                            _render_inline_evidence(matching_sel, onto)

                with st.expander("📊 Statistics", expanded=False):
                    all_traces = []
                    for _, _, traces in role_traces_by_sig:
                        all_traces.extend(traces)
                    _render_stats(all_traces)

    # =============================================
    # SECTION 3: SIGNATURE SUMMARY CARDS
    # =============================================
    st.markdown("---")
    st.markdown("### 📋 Selected Signatures")
    _render_signature_cards(signatures)


# =============================================================================
# HELPERS
# =============================================================================

def _resolve_team_scenario(team: TeamSelection, loader: DataLoader) -> Optional[TeamScenario]:
    """Resolve TeamSelection into full TeamScenario (with DCE/day/session)"""
    # Build a TeamScenario with all available fields from the unified selection
    if team.dce and team.data_source in ('session_physio', 'both'):
        return TeamScenario(
            team_id=team.team_id, scenario_id=team.scenario_id,
            data_source=team.data_source,
            dce=team.dce, day=team.day, session=team.session,
            session_label=team.session_label,
            entropy_run=getattr(team, 'entropy_run', None)
        )
    if team.session_label:
        # Try to find matching session parquet for physio data
        matched = loader.find_matching_session(team.team_id, team.session_label)
        if matched:
            # Carry over entropy_run if we have it
            if hasattr(team, 'entropy_run') and team.entropy_run:
                matched.entropy_run = team.entropy_run
            return matched
    # Entropy-only scenario (no session physio available)
    return TeamScenario(
        team_id=team.team_id, scenario_id=team.scenario_id,
        data_source=team.data_source or 'entropy_ami',
        day=team.day, session=team.session,
        session_label=team.session_label,
        entropy_run=getattr(team, 'entropy_run', None)
    )


def _load_subtasks(
    team: TeamSelection,
    team_scenario: Optional[TeamScenario],
    loader: DataLoader
) -> List[SubtaskEvent]:
    """Load subtask events using all available context"""
    day = team.day
    if not day and team_scenario:
        day = team_scenario.day
    if not day and team.session_label:
        parts = team.session_label.split('_')
        if parts and parts[0].startswith('Day'):
            day = parts[0]

    run = None
    try:
        if team.scenario_id and team.scenario_id.isdigit():
            run = int(team.scenario_id)
    except (ValueError, AttributeError):
        pass

    # Try with run filter first; if empty, fall back to day-only
    # (Run numbering may differ between entropy CSV and subtask Excel)
    events = loader.get_subtasks(team_id=team.team_id, day=day, run=run)
    if not events and run is not None:
        events = loader.get_subtasks(team_id=team.team_id, day=day, run=None)

    return events


# =============================================================================
# PLOTTING
# =============================================================================

def _add_subtask_overlays(fig, subtasks: List[SubtaskEvent], row=None, col=None):
    """
    Add subtask overlays to a figure:
    - Colored rect shapes (category-based color)
    - Annotation with subtask label + members

    Uses add_shape with explicit xref/yref for subplot compatibility,
    since add_vrect with row/col silently fails on make_subplots figures.
    """
    for evt in subtasks:
        cat = evt.category or 1
        fill = SUBTASK_COLORS.get(cat, "rgba(128,128,128,0.10)")
        border = SUBTASK_BORDER.get(cat, "rgba(128,128,128,0.3)")

        # Annotation text: label + members abbreviated
        members_short = evt.members.strip()
        if members_short.lower() == 'all':
            members_short = 'All'
        elif len(members_short) > 20:
            parts = [m.strip()[:3] for m in members_short.split(',')]
            members_short = ','.join(parts)
        ann_text = f"{evt.label} · {members_short}"

        # Determine axis references
        if row is not None and row > 1:
            xref = f'x{row}'
            yref = f'y{row}'
        else:
            xref = 'x'
            yref = 'y'

        # Add shape with explicit axis refs (works in both simple and subplot figures)
        fig.add_shape(
            type='rect',
            x0=float(evt.start_sec), x1=float(evt.end_sec),
            y0=0, y1=1,
            xref=xref, yref=f'{yref} domain',
            fillcolor=fill, layer='below',
            line=dict(width=1, color=border)
        )

        # Add annotation
        fig.add_annotation(
            x=float(evt.start_sec), y=1,
            xref=xref, yref=f'{yref} domain',
            text=ann_text,
            showarrow=False,
            xanchor='left', yanchor='top',
            font=dict(size=9, color='rgba(0,0,0,0.5)'),
        )


def _add_subtask_hover_markers(fig, subtasks: List[SubtaskEvent], y_position=None,
                               row=None, col=None,
                               construct_map: Optional['SubtaskConstructMap'] = None):
    """
    Add diamond markers at the top of each subtask region for hover tooltips.
    When a construct_map is available, the tooltip includes expected construct demands.
    """
    if not subtasks:
        return

    for evt in subtasks:
        cat = evt.category or 1
        border = SUBTASK_BORDER.get(cat, "rgba(128,128,128,0.6)")
        mid_x = (evt.start_sec + evt.end_sec) / 2
        duration = evt.end_sec - evt.start_sec

        # Build hover text
        hover_lines = [
            f"<b>{evt.label}</b>",
            f"Members: {evt.members}",
            f"Category: {evt.category}",
            f"Duration: {duration:.0f}s ({duration/60:.1f} min)",
            f"Time: {evt.start_sec:.0f}s – {evt.end_sec:.0f}s",
        ]

        # Enrich with construct demands if available
        if construct_map and CONSTRUCT_MAP_AVAILABLE:
            subtask_num = getattr(evt, 'subtask_number', None) or evt.category
            if subtask_num is not None:
                profile = construct_map.get_profile(int(subtask_num))
                if profile and profile.demands:
                    hover_lines.append("─────────────")
                    hover_lines.append("<b>Expected demands:</b>")
                    for d in profile.top_demands(5):
                        bar_len = int(d.weight * 8)
                        bar = "█" * bar_len + "░" * (8 - bar_len)
                        hover_lines.append(f"  {d.short_label}: {bar} ({d.weight:.1f})")
                    if profile.description:
                        hover_lines.append(f"<i>{profile.description[:80]}</i>")

        hover_text = "<br>".join(hover_lines) + "<extra></extra>"

        trace_kwargs = dict(
            x=[float(mid_x)],
            y=[y_position] if y_position else [0],
            mode='markers',
            marker=dict(size=10, symbol='diamond', color=border, opacity=0.7,
                        line=dict(width=1, color='white')),
            showlegend=False,
            hovertemplate=hover_text
        )

        if row is not None:
            fig.add_trace(go.Scatter(**trace_kwargs), row=row, col=col)
        else:
            fig.add_trace(go.Scatter(**trace_kwargs))


def _add_construct_heatmap(
    fig,
    subtasks: List[SubtaskEvent],
    construct_map: 'SubtaskConstructMap',
    construct_filter: Optional[List[str]] = None,
    row: int = 2,
    col: int = 1,
):
    """
    Add a construct demand heatmap as a subplot row.
    Each cell is a colored rectangle whose opacity reflects the demand weight.
    """
    if not subtasks or not construct_map or not construct_map.available:
        return

    labels, x_starts, x_ends, z_matrix = construct_map.build_heatmap_data(
        subtasks, constructs_to_show=construct_filter
    )
    if not labels or not x_starts:
        return

    n_constructs = len(labels)
    n_events = len(x_starts)

    # Draw filled rectangles for each cell
    for i, construct_label in enumerate(labels):
        construct_name = (construct_filter[i] if construct_filter and i < len(construct_filter)
                         else construct_map.all_constructs()[i] if i < len(construct_map.all_constructs())
                         else "unknown")
        base_color = construct_map.get_construct_color(construct_name)

        for j in range(n_events):
            weight = z_matrix[i][j]
            if weight <= 0:
                continue

            # Convert hex color to rgba with weight as opacity
            r, g, b = _hex_to_rgb(base_color)
            fill_color = f"rgba({r},{g},{b},{min(weight * 0.8 + 0.1, 0.9)})"

            # Add shape for this cell
            y0 = n_constructs - i - 1  # Invert so first construct is at top
            y1 = y0 + 1

            fig.add_shape(
                type='rect',
                x0=float(x_starts[j]), x1=float(x_ends[j]),
                y0=y0, y1=y1,
                xref=f'x{row}' if row > 1 else 'x',
                yref=f'y{row}' if row > 1 else 'y',
                fillcolor=fill_color,
                line=dict(width=0.5, color='rgba(255,255,255,0.5)'),
                layer='above',
            )

            # Add hover trace at cell center
            mid_x = (x_starts[j] + x_ends[j]) / 2
            mid_y = y0 + 0.5

            subtask_num = getattr(subtasks[j], 'subtask_number', None) or subtasks[j].category
            profile = construct_map.get_profile(int(subtask_num)) if subtask_num else None
            profile_label = profile.label if profile else f"Subtask {subtask_num}"

            fig.add_trace(
                go.Scatter(
                    x=[float(mid_x)], y=[mid_y],
                    mode='markers',
                    marker=dict(size=1, opacity=0),
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{construct_label}</b>: {weight:.1f}<br>"
                        f"{profile_label}<br>"
                        "<extra></extra>"
                    ),
                ),
                row=row, col=col,
            )

    # Configure the heatmap y-axis with construct labels
    yaxis_key = f'yaxis{row}' if row > 1 else 'yaxis'
    fig.update_layout(**{
        yaxis_key: dict(
            tickmode='array',
            tickvals=[n_constructs - i - 0.5 for i in range(n_constructs)],
            ticktext=labels,
            range=[0, n_constructs],
            showgrid=False,
            fixedrange=True,
            tickfont=dict(size=10),
        )
    })


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    h = hex_color.lstrip('#')
    if len(h) == 6:
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return 100, 100, 100


def _add_speaking_stacked(
    fig,
    overlay_data: dict,
    role_colors: Dict[str, str],
    role_order: List[str],
    row: int = 1,
    col: int = 1,
    silence_color: str = 'rgba(180,180,180,0.35)',
):
    """Add a stacked-area speaking strip with explicit silence band.

    Decomposes each minute into:
      - One band per role, height = (role's share of talk) × (team-union speaking
        proportion). This means total role-band height equals the fraction of
        the minute where ANY role was speaking — preserving overall density.
      - A silence band on top, height = 1 − team-union speaking proportion.
    Total stack always sums to 1.

    Hover surfaces both the role's share-of-talk and absolute speaking
    proportion; the silence band shows its own proportion.
    """
    import numpy as np
    timestamps = overlay_data.get('timestamps')
    role_props = overlay_data.get('role_props', {})
    union_prop = overlay_data.get('union_prop')
    if timestamps is None or len(timestamps) == 0 or not role_props or union_prop is None:
        return

    window_sec = float(overlay_data.get('window_sec', 60.0))
    x = list(timestamps)
    n_bins = len(x)

    role_arrays: Dict[str, np.ndarray] = {
        role: np.nan_to_num(role_props[role], nan=0.0)
        for role in role_order if role in role_props
    }
    if not role_arrays:
        return

    # Sum-of-roles is what we use to compute share-of-talk; can exceed union
    # when speech overlaps. Fall back gracefully when the sum is zero.
    sum_roles = np.zeros(n_bins, dtype=float)
    for arr in role_arrays.values():
        sum_roles = sum_roles + arr
    union = np.asarray(union_prop, dtype=float)
    # In silent minutes union may be 0 or NaN — clamp to [0, 1].
    union_safe = np.nan_to_num(np.clip(union, 0.0, 1.0), nan=0.0)
    silence = 1.0 - union_safe

    cumulative = np.zeros(n_bins, dtype=float)
    for role in role_order:
        if role not in role_arrays:
            continue
        raw = role_arrays[role]
        with np.errstate(invalid='ignore', divide='ignore'):
            share = np.where(sum_roles > 0, raw / sum_roles, 0.0)
        # Role band height = share-of-talk × union (so band area integrates to
        # the role's share of speaking time, scaled by overall density).
        band = share * union_safe
        next_cumulative = cumulative + band

        color = role_colors.get(role, '#888')
        r, g, b = _hex_to_rgb(color)
        fill_color = f'rgba({r},{g},{b},0.6)'
        line_color = f'rgba({r},{g},{b},0.9)'

        # customdata cols: 0 = raw role proportion, 1 = share-of-talk
        customdata = np.column_stack([raw, share])

        fig.add_trace(
            go.Scatter(
                x=x, y=cumulative.tolist(),
                mode='lines', line=dict(width=0),
                showlegend=False, hoverinfo='skip',
            ),
            row=row, col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=x, y=next_cumulative.tolist(),
                mode='lines',
                line=dict(width=0.5, color=line_color),
                fill='tonexty', fillcolor=fill_color,
                name=f"{role} speaking",
                legendgroup='speaking',
                hovertemplate=(
                    f"<b>{role}</b><br>"
                    "Window start: %{x:.0f}s<br>"
                    f"Spoke %{{customdata[0]:.0%}} of this {int(window_sec)}s window<br>"
                    "(%{customdata[1]:.0%} of talk time)"
                    "<extra></extra>"
                ),
                customdata=customdata,
            ),
            row=row, col=col,
        )
        cumulative = next_cumulative

    # Silence band on top — fills 1 − union
    next_cumulative = cumulative + silence
    fig.add_trace(
        go.Scatter(
            x=x, y=cumulative.tolist(),
            mode='lines', line=dict(width=0),
            showlegend=False, hoverinfo='skip',
        ),
        row=row, col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=next_cumulative.tolist(),
            mode='lines',
            line=dict(width=0.5, color='rgba(120,120,120,0.5)'),
            fill='tonexty', fillcolor=silence_color,
            name='Silence',
            legendgroup='speaking',
            hovertemplate=(
                "<b>Silence</b><br>"
                "Window start: %{x:.0f}s<br>"
                f"%{{customdata:.0%}} of this {int(window_sec)}s window"
                "<extra></extra>"
            ),
            customdata=silence.tolist(),
        ),
        row=row, col=col,
    )

    # Y-axis locked to [0, 1]
    yaxis_key = f'yaxis{row}' if row > 1 else 'yaxis'
    fig.update_layout(**{
        yaxis_key: dict(
            title=f'Speaking ({int(window_sec)}s)',
            range=[0, 1],
            tickvals=[0, 0.5, 1.0],
            ticktext=['0', '½', '1'],
            showgrid=False,
            tickfont=dict(size=9),
            fixedrange=True,
        )
    })


def _add_speaking_heat(
    fig,
    overlay_data: dict,
    role_colors: Dict[str, str],
    role_order: List[str],
    row: int = 1,
    col: int = 1,
    silence_threshold: float = 0.05,
):
    """Add a single-row heat strip showing dominant speaker per window.

    Each cell spans one window bin, colored by the role with the highest mean
    speaking proportion in that bin. Opacity scales with that proportion. Bins
    where no role exceeds `silence_threshold` are rendered as light gray.
    """
    import numpy as np
    timestamps = overlay_data.get('timestamps')
    role_props = overlay_data.get('role_props', {})
    if timestamps is None or len(timestamps) == 0 or not role_props:
        return

    window_sec = float(overlay_data.get('window_sec', 60.0))
    roles = [r for r in role_order if r in role_props]
    if not roles:
        return

    # Build a (n_roles, n_bins) matrix
    n_bins = len(timestamps)
    mat = np.full((len(roles), n_bins), np.nan, dtype=float)
    for i, role in enumerate(roles):
        vals = role_props[role]
        if len(vals) == n_bins:
            mat[i, :] = vals
        else:
            mat[i, :len(vals)] = vals
    mat_filled = np.nan_to_num(mat, nan=0.0)

    dominant_idx = np.argmax(mat_filled, axis=0)
    dominant_val = mat_filled[dominant_idx, np.arange(n_bins)]

    # Draw one rect per bin
    yref = f'y{row}' if row > 1 else 'y'
    xref = f'x{row}' if row > 1 else 'x'
    for j in range(n_bins):
        x0 = float(timestamps[j])
        x1 = x0 + window_sec
        val = float(dominant_val[j])
        if val < silence_threshold:
            fill_color = 'rgba(200,200,200,0.25)'
            label = 'silence'
        else:
            role = roles[int(dominant_idx[j])]
            r, g, b = _hex_to_rgb(role_colors.get(role, '#888'))
            opacity = float(min(0.9, 0.2 + val * 0.8))
            fill_color = f'rgba({r},{g},{b},{opacity})'
            label = role

        fig.add_shape(
            type='rect',
            x0=x0, x1=x1, y0=0, y1=1,
            xref=xref, yref=yref,
            fillcolor=fill_color,
            line=dict(width=0),
            layer='below',
        )
        # Hover marker at cell center
        fig.add_trace(
            go.Scatter(
                x=[x0 + window_sec / 2], y=[0.5],
                mode='markers', marker=dict(size=1, opacity=0),
                showlegend=False,
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    f"Window: %{{x:.0f}}s ± {int(window_sec/2)}s<br>"
                    f"Speaking proportion: {val:.2f}"
                    "<extra></extra>"
                ),
            ),
            row=row, col=col,
        )

    yaxis_key = f'yaxis{row}' if row > 1 else 'yaxis'
    fig.update_layout(**{
        yaxis_key: dict(
            title=f'Dominant ({int(window_sec)}s)',
            range=[0, 1],
            showgrid=False,
            showticklabels=False,
            fixedrange=True,
        )
    })


def _add_speaking_lanes(
    fig,
    overlay_data: dict,
    role_colors: Dict[str, str],
    role_order: List[str],
    row: int = 1,
    col: int = 1,
    speaking_threshold: float = 0.05,
):
    """Add per-role speaking/not-speaking swim-lanes (one row per role).

    Each cell is filled with the role color when that role spoke for at least
    `speaking_threshold` of the minute (default 5% = 3s of a 60s window),
    empty otherwise. Y-axis is labeled with role IDs.
    """
    import numpy as np
    timestamps = overlay_data.get('timestamps')
    role_props = overlay_data.get('role_props', {})
    if timestamps is None or len(timestamps) == 0 or not role_props:
        return

    window_sec = float(overlay_data.get('window_sec', 60.0))

    # Display order: top → bottom, mirroring role_order top-down
    lanes = [r for r in role_order if r in role_props]
    if not lanes:
        return
    n_lanes = len(lanes)

    yref = f'y{row}' if row > 1 else 'y'
    xref = f'x{row}' if row > 1 else 'x'

    for i, role in enumerate(lanes):
        # Place first role at the top
        y0 = n_lanes - i - 1
        y1 = y0 + 1
        r, g, b = _hex_to_rgb(role_colors.get(role, '#888'))
        fill_color = f'rgba({r},{g},{b},0.7)'
        empty_color = 'rgba(230,230,230,0.4)'

        vals = role_props[role]
        for j in range(len(timestamps)):
            x0 = float(timestamps[j])
            x1 = x0 + window_sec
            v = float(vals[j]) if not np.isnan(vals[j]) else 0.0
            spoke = v >= speaking_threshold

            fig.add_shape(
                type='rect',
                x0=x0, x1=x1, y0=y0, y1=y1,
                xref=xref, yref=yref,
                fillcolor=fill_color if spoke else empty_color,
                line=dict(width=0.3, color='rgba(255,255,255,0.6)'),
                layer='below',
            )

            # Hover marker at cell center
            fig.add_trace(
                go.Scatter(
                    x=[x0 + window_sec / 2], y=[y0 + 0.5],
                    mode='markers', marker=dict(size=1, opacity=0),
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{role}</b><br>"
                        f"Window: %{{x:.0f}}s ± {int(window_sec/2)}s<br>"
                        f"{'Spoke' if spoke else 'Did not speak'}"
                        f" ({v:.0%} of {int(window_sec)}s)"
                        "<extra></extra>"
                    ),
                ),
                row=row, col=col,
            )

    # Y-axis labels = role IDs
    yaxis_key = f'yaxis{row}' if row > 1 else 'yaxis'
    fig.update_layout(**{
        yaxis_key: dict(
            tickmode='array',
            tickvals=[n_lanes - i - 0.5 for i in range(n_lanes)],
            ticktext=lanes,
            range=[0, n_lanes],
            showgrid=False,
            fixedrange=True,
            tickfont=dict(size=10),
        )
    })


def _build_plot(
    traces: List[TimeseriesData],
    sig_def,
    subtasks: List[SubtaskEvent],
    title: Optional[str] = None,
    y_label: Optional[str] = None,
    height: int = 400,
    rolling_cfg: Optional[dict] = None,
    construct_map: Optional['SubtaskConstructMap'] = None,
    construct_view: str = 'none',
    signature_construct: Optional[str] = None,
    speaking_overlay_data: Optional[dict] = None,
    speaking_overlay_mode: str = 'none',
    role_colors: Optional[Dict[str, str]] = None,
    role_order: Optional[List[str]] = None,
) -> go.Figure:
    """
    Build a Plotly figure with traces + subtask overlays + optional rolling stats
    + optional construct heatmap strip.
    
    Args:
        construct_map: SubtaskConstructMap instance (None to skip heatmap)
        construct_view: 'none', 'focused', or 'full'
        signature_construct: The construct this signature measures (for focused view)
    """
    # Determine if we need the heatmap subplot
    show_heatmap = (
        construct_map is not None
        and construct_view != 'none'
        and subtasks
        and CONSTRUCT_MAP_AVAILABLE
    )

    if show_heatmap:
        # Determine construct filter for focused view
        construct_filter = None
        if construct_view == 'focused' and signature_construct:
            relevant = construct_map.constructs_for_signature(signature_construct)
            if relevant:
                construct_filter = relevant
            else:
                # Fall back to full if signature construct not in mapping
                construct_filter = None

        # How many rows in the heatmap?
        if construct_filter:
            n_heatmap_rows = len(construct_filter)
        else:
            n_heatmap_rows = len(construct_map.all_constructs())

        if n_heatmap_rows == 0:
            show_heatmap = False

    # Determine speaking overlay
    show_speaking = (
        speaking_overlay_mode in ('stacked', 'heat', 'lanes')
        and speaking_overlay_data is not None
        and speaking_overlay_data.get('role_props')
    )

    # Decide subplot layout (top → bottom: speaking, signal, construct heatmap)
    if show_heatmap or show_speaking:
        # Heights — keep signal panel dominant
        signal_height = height
        # Lanes mode needs taller strip (one mini-row per role); other modes
        # are single-row strips at fixed height.
        if not show_speaking:
            speaking_height = 0
        elif speaking_overlay_mode == 'lanes':
            n_lanes = sum(
                1 for r in (role_order or [])
                if r in (speaking_overlay_data or {}).get('role_props', {})
            ) or 1
            speaking_height = max(28 * n_lanes, 90)
        else:
            speaking_height = 90
        heatmap_height = max(25 * n_heatmap_rows, 60) if show_heatmap else 0
        total_height = signal_height + speaking_height + heatmap_height

        row_heights: List[float] = []
        subplot_titles: List[str] = []
        if show_speaking:
            row_heights.append(speaking_height / total_height)
            label = {
                'stacked': 'Speaking proportion (per role, per minute)',
                'heat': 'Dominant speaker (per minute)',
                'lanes': 'Spoke / did not speak (per role, per minute)',
            }.get(speaking_overlay_mode, 'Speaking activity')
            subplot_titles.append(label)
        row_heights.append(signal_height / total_height)
        subplot_titles.append("")
        if show_heatmap:
            row_heights.append(heatmap_height / total_height)
            subplot_titles.append("Construct Demands")

        fig = make_subplots(
            rows=len(row_heights), cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=row_heights,
            subplot_titles=subplot_titles,
        )
        speaking_row = 1 if show_speaking else None
        signal_row = (2 if show_speaking else 1)
        heatmap_row = signal_row + 1 if show_heatmap else None
    else:
        fig = go.Figure()
        speaking_row = None
        signal_row = None
        heatmap_row = None
        total_height = height
    # Add subtask overlays to signal panel
    add_kwargs = dict(row=signal_row, col=1) if signal_row else {}
    _add_subtask_overlays(fig, subtasks,
                          row=signal_row if signal_row else None,
                          col=1 if signal_row else None)

    for ts in traces:
        unit = ts.unit or (sig_def.unit if sig_def else "")
        fig.add_trace(go.Scatter(
            x=ts.timestamps, y=ts.values,
            mode='lines', name=ts.label,
            line=dict(color=ts.color, width=1.5),
            hovertemplate=(
                f"<b>{ts.label}</b><br>"
                "Time: %{x:.0f}s<br>"
                f"Value: %{{y:.3f}} {unit}<br>"
                "<extra></extra>"
            ),
            connectgaps=False
        ), **add_kwargs)

    # Rolling statistics bands (entropy/AMI only)
    if rolling_cfg and rolling_cfg.get('enabled'):
        for ts in traces:
            _add_rolling_stats(
                fig, ts,
                window=rolling_cfg['window'],
                n_sd=rolling_cfg['n_sd'],
                row=signal_row if signal_row else None,
                col=1 if signal_row else None,
            )

    # Add hover markers for subtasks near the top of the y-axis
    import numpy as np
    all_vals = []
    for ts in traces:
        all_vals.extend([v for v in ts.values if v is not None and not (isinstance(v, float) and np.isnan(v))])
    if all_vals and subtasks:
        y_top = np.nanpercentile(all_vals, 97)
        _add_subtask_hover_markers(fig, subtasks, y_position=y_top,
                                   row=signal_row if signal_row else None,
                                   col=1 if signal_row else None,
                                   construct_map=construct_map)

    # Add speaking overlay strip (top row) if enabled
    if show_speaking and speaking_row:
        _role_colors = role_colors or {}
        _role_order = role_order or list(_role_colors.keys())
        if speaking_overlay_mode == 'stacked':
            _add_speaking_stacked(
                fig, speaking_overlay_data,
                role_colors=_role_colors, role_order=_role_order,
                row=speaking_row, col=1,
            )
        elif speaking_overlay_mode == 'heat':
            _add_speaking_heat(
                fig, speaking_overlay_data,
                role_colors=_role_colors, role_order=_role_order,
                row=speaking_row, col=1,
            )
        elif speaking_overlay_mode == 'lanes':
            _add_speaking_lanes(
                fig, speaking_overlay_data,
                role_colors=_role_colors, role_order=_role_order,
                row=speaking_row, col=1,
            )

    # Add construct heatmap strip (bottom row) if enabled
    if show_heatmap and heatmap_row:
        _add_construct_heatmap(
            fig, subtasks, construct_map,
            construct_filter=construct_filter,
            row=heatmap_row, col=1,
        )

    unit_str = f" ({sig_def.unit})" if sig_def and sig_def.unit else ""
    y_title = y_label or (sig_def.y_label if sig_def else "Value")

    # Determine signal-row axis keys (axes are numbered by row in subplots)
    n_rows = (1 if speaking_row else 0) + 1 + (1 if heatmap_row else 0)
    bottom_row = (heatmap_row if heatmap_row else signal_row) or 1

    def _xaxis_key(r):
        return 'xaxis' if r == 1 else f'xaxis{r}'
    def _yaxis_key(r):
        return 'yaxis' if r == 1 else f'yaxis{r}'

    sig_xkey = _xaxis_key(signal_row or 1)
    sig_ykey = _yaxis_key(signal_row or 1)
    bottom_xkey = _xaxis_key(bottom_row)

    layout_kwargs = dict(
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=11)),
        hovermode='x unified', plot_bgcolor='white',
        height=total_height, margin=dict(l=60, r=20, t=50, b=50)
    )
    # Signal x-axis: only show title if it's the bottom row
    layout_kwargs[sig_xkey] = dict(
        title="Time (seconds from session start)" if signal_row == bottom_row else "",
        showgrid=True, gridcolor='rgba(0,0,0,0.06)',
    )
    layout_kwargs[sig_ykey] = dict(
        title=f"{y_title}{unit_str}",
        showgrid=True, gridcolor='rgba(0,0,0,0.06)',
    )
    # Bottom-row x-axis title (when bottom != signal row)
    if bottom_row != (signal_row or 1):
        layout_kwargs[bottom_xkey] = dict(
            title="Time (seconds from session start)",
            showgrid=True, gridcolor='rgba(0,0,0,0.06)',
        )

    if title:
        layout_kwargs['title'] = dict(text=str(title), font=dict(size=14))
    else:
        layout_kwargs['title'] = dict(text="")
        layout_kwargs['margin'] = dict(l=60, r=20, t=30, b=50)
    fig.update_layout(**layout_kwargs)

    return fig


def _build_multi_sig_subplot(
    traces_by_sig: List[Tuple[str, any, List[TimeseriesData]]],
    role,
    subtasks: List[SubtaskEvent]
) -> go.Figure:
    """Stacked subplots for multiple signatures on one role, shared x-axis"""

    n = len(traces_by_sig)
    subplot_titles = [name or "—" for name, _, _ in traces_by_sig]
    fig = make_subplots(
        rows=n, cols=1, shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=subplot_titles
    )

    for i, (sig_name, sig_def, traces) in enumerate(traces_by_sig):
        row = i + 1
        _add_subtask_overlays(fig, subtasks, row=row, col=1)

        for ts in traces:
            fig.add_trace(
                go.Scatter(
                    x=ts.timestamps, y=ts.values,
                    mode='lines', name=ts.label,
                    line=dict(color=role.color, width=1.5),
                    connectgaps=False, showlegend=(i == 0),
                    hovertemplate=(
                        f"<b>{ts.label}</b><br>"
                        "Time: %{x:.0f}s<br>"
                        "Value: %{y:.3f}<br><extra></extra>"
                    )
                ),
                row=row, col=1
            )

        # Hover markers for subtasks
        import numpy as np
        all_vals = []
        for ts in traces:
            all_vals.extend([v for v in ts.values if v is not None and not (isinstance(v, float) and np.isnan(v))])
        if all_vals and subtasks:
            y_top = np.nanpercentile(all_vals, 97)
            _add_subtask_hover_markers(fig, subtasks, y_position=y_top, row=row, col=1)

        y_label = (sig_def.y_label if sig_def and sig_def.y_label else sig_name) or "Value"
        fig.update_yaxes(title_text=y_label, row=row, col=1)

    fig.update_xaxes(title_text="Time (seconds from session start)", row=n, col=1)
    fig.update_layout(
        height=280 * n, plot_bgcolor='white', hovermode='x unified',
        margin=dict(l=60, r=20, t=40, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


# =============================================================================
# STATS / CARDS / EVIDENCE
# =============================================================================

def _render_stats(traces: List[TimeseriesData]):
    import pandas as pd
    import numpy as np
    rows = []
    for ts in traces:
        vals = [v for v in ts.values if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if vals:
            rows.append({'Trace': ts.label, 'Valid Pts': len(vals),
                         'Min': f"{min(vals):.3f}", 'Max': f"{max(vals):.3f}",
                         'Mean': f"{sum(vals)/len(vals):.3f}", 'Std': f"{np.std(vals):.3f}"})
        else:
            rows.append({'Trace': ts.label, 'Valid Pts': 0,
                         'Min': '—', 'Max': '—', 'Mean': '—', 'Std': '—'})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_placeholder():
    st.markdown("### 📈 Signature Timeline")
    st.markdown("""
    <div style="background:#f8fafc;border:2px dashed #cbd5e1;border-radius:8px;
                padding:2rem;text-align:center;color:#64748b;">
        📊 Select signatures with data mappings to see timeseries
    </div>
    """, unsafe_allow_html=True)


def _render_signature_cards(signatures: List[SelectedSignature]):
    cols = st.columns(min(len(signatures), 3))
    for idx, sig in enumerate(signatures):
        with cols[idx % 3]:
            icon = get_icon(sig.modality_category)
            color = CARD_COLORS[idx % len(CARD_COLORS)]
            extras = []
            if sig.channel:
                extras.append(f"Channel: {sig.channel}")
            if sig.signal_type and sig.signal_type != "All":
                extras.append(f"Signal: {sig.signal_type}")
            extra_html = "".join(
                f"<br><span style='color:#999;font-size:0.8em;'>{e}</span>" for e in extras
            )
            st.markdown(f"""
            <div style="border-left:4px solid {color};padding-left:12px;margin-bottom:16px;">
                <p style="margin:0;font-size:1.1em;"><b>{icon} {sig.label}</b></p>
                <p style="margin:4px 0;color:#666;font-size:0.9em;">{sig.modality_category}</p>
                {extra_html}
            </div>
            """, unsafe_allow_html=True)


def _render_inline_evidence(sig: SelectedSignature, onto: OntologyAccess):
    """Compact evidence block for placement directly under a plot."""
    with st.expander(f"💡 Evidence & Interpretation — {sig.label}", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Category:** {sig.modality_category or '—'}")
            st.markdown(f"**Technique:** {sig.technique or '—'}")
        with col2:
            if sig.construct:
                st.markdown(f"**Construct:** {sig.construct}")
            if sig.data_source:
                st.markdown(f"**Source:** `{sig.data_source}`")
        st.markdown("---")
        st.markdown("**Interpretation Guidance**")
        _render_guidance(sig)

        # Knowledge graph neighborhood visualization
        if NEIGHBORHOOD_VIZ_AVAILABLE and (sig.uri or sig.construct or sig.modality_category):
            st.markdown("---")
            st.markdown("**🌐 Knowledge Graph Context**")
            st.caption(
                "This signature's position in the ontology — "
                "related constructs, modalities, techniques, and measures from the literature."
            )
            try:
                fig = render_measure_neighborhood(
                    onto,
                    sig_uri=sig.uri or "",
                    sig_label=sig.label,
                    construct=sig.construct,
                    modality_category=sig.modality_category,
                    technique=sig.technique,
                    max_measures=10,
                    height=380,
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True,
                                   key=f"kg_{sig.uri or sig.label}")
                else:
                    st.caption("No ontology connections found for this signature.")
            except Exception as e:
                st.caption(f"Could not render knowledge graph: {e}")

        st.markdown("---")
        st.caption(
            "📚 Full evidence synthesis (effect sizes, study counts) "
            "will be available once the evidence layer is complete."
        )


def _render_evidence(onto: OntologyAccess, signatures: List[SelectedSignature]):
    for sig in signatures:
        icon = get_icon(sig.modality_category)
        with st.expander(f"{icon} **{sig.label}**", expanded=False):
            m1, m2 = st.columns(2)
            with m1:
                st.markdown(f"**Category:** {sig.modality_category}")
                st.markdown(f"**Measure type:** {sig.technique}")
            with m2:
                if sig.data_signature_id:
                    st.markdown(f"**Config ID:** `{sig.data_signature_id}`")
                if sig.data_source:
                    st.markdown(f"**Source:** `{sig.data_source}`")
            st.markdown("---")
            st.markdown("#### 💡 Interpretation Guidance")
            _render_guidance(sig)


def _render_guidance(sig: SelectedSignature):
    guidance = []
    cat = (sig.modality_category or "").lower()
    tech = (sig.technique or "").lower()

    if 'entropy' in tech or 'entropy' in cat:
        guidance += ["**Entropy**: Higher = more variability/unpredictability",
                     "Increases during novel/challenging situations"]
    if 'ami' in tech:
        guidance += ["**AMI**: Measures physiological coupling between members",
                     "Higher AMI = stronger synchronization"]
    if 'cardiac' in cat or 'ekg' in cat:
        guidance += ["**IBI**: Longer = slower heart rate (relaxed)",
                     "**HRV**: Higher variability = better autonomic flexibility"]
    if 'ocular' in cat or 'pupil' in cat:
        guidance += ["**Pupil dilation**: Larger = higher cognitive load"]
    if 'respiratory' in cat:
        guidance += ["**Resp rate**: Increases with stress",
                     "**RVT/Amplitude**: Breathing depth/effort"]
    if 'eeg' in cat:
        guidance += ["**Delta**: Deep processing", "**Theta**: Memory/learning",
                     "**Alpha**: Relaxed alertness", "**Beta**: Active focus"]
    if not guidance:
        guidance.append("Consider task context and baseline comparisons.")
    for g in guidance[:5]:
        st.markdown(f"- {g}")


def render_back_button() -> bool:
    c1, _ = st.columns([1, 5])
    with c1:
        return st.button("← Back to Selection", key="back_analysis")
