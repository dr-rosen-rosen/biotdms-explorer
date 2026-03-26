"""
Use Case 2: Signature Selection View

Two-panel interface:
  Left:  Team & scenario selection (from available data)
  Right: Signature browser with category/signal-type filters

Supports both entropy/AMI and session physio signatures.
Role-level signatures auto-expand for all roles.
"""

import streamlit as st
from typing import List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import sys

# Path setup
_current_file = Path(__file__).resolve() if '__file__' in dir() else Path.cwd() / 'views' / 'uc2_signature_selection.py'
_app_dir = _current_file.parent.parent
if str(_app_dir) not in sys.path:
    sys.path.insert(0, str(_app_dir))

from core.ontology import OntologyAccess, Measure

try:
    from core.data_loader import (
        SignatureRegistry, DataLoader, TeamScenario,
        SignatureDefinition, create_data_loader
    )
    DATA_LOADER_AVAILABLE = True
except ImportError:
    DATA_LOADER_AVAILABLE = False


# =============================================================================
# DATA CLASSES for passing between views
# =============================================================================

@dataclass
class SelectedSignature:
    """A signature selected for analysis - passed to the analysis view"""
    uri: str                    # Unique ID (sig_id from YAML or ontology URI)
    label: str
    modality: str
    modality_category: str
    technique: str
    construct: str
    data_signature_id: Optional[str] = None    # YAML sig id
    data_source: Optional[str] = None          # "entropy_ami" or "session_physio"
    signal_type: Optional[str] = None          # Legacy single: "All", "Neuro", "Auto"
    signal_types: Optional[List[str]] = None   # Multi-select: ["All", "Neuro"] etc.
    channel: Optional[str] = None              # EEG channel


@dataclass
class TeamSelection:
    """Selected team and scenario - passed to the analysis view"""
    team_id: str
    scenario_id: str
    description: str
    data_source: Optional[str] = None
    dce: Optional[str] = None
    day: Optional[str] = None
    session: Optional[str] = None
    session_label: Optional[str] = None
    entropy_run: Optional[str] = None


# =============================================================================
# HELPERS
# =============================================================================

def _get(obj, attr, default=None):
    """Get attribute from dict or dataclass"""
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


def _to_dict(obj) -> dict:
    """Convert to dict for session state storage"""
    if isinstance(obj, dict):
        return obj
    return asdict(obj)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def render_signature_selection(onto: OntologyAccess) -> Tuple[Optional[TeamSelection], List[SelectedSignature]]:
    """Render the signature selection interface"""

    st.header("📊 Select Signatures to Analyze")

    # Init state
    if 'selected_signatures' not in st.session_state:
        st.session_state.selected_signatures = []
    if 'selected_team' not in st.session_state:
        st.session_state.selected_team = None

    # Check for data config
    config_path = _app_dir / "config" / "signatures.yaml"
    data_dir = _app_dir / "data"
    has_data_config = config_path.exists() and DATA_LOADER_AVAILABLE

    if has_data_config:
        return render_data_driven_selection(config_path, data_dir, onto)
    else:
        return render_ontology_driven_selection(onto)


# =============================================================================
# DATA-DRIVEN SELECTION (primary path when YAML config exists)
# =============================================================================

def render_data_driven_selection(
    config_path: Path, data_dir: Path, onto: OntologyAccess
) -> Tuple[Optional[TeamSelection], List[SelectedSignature]]:
    """Selection using YAML-configured signatures with real data"""

    try:
        registry = SignatureRegistry(config_path)
        loader = DataLoader(data_dir, registry)
    except Exception as e:
        st.error(f"Error loading configuration: {e}")
        return None, []

    # Two-panel layout
    col_team, col_sigs = st.columns([1, 2], gap="large")

    # ---- LEFT: Team & Scenario ----
    with col_team:
        st.markdown("### 👥 Team & Scenario")
        available = loader.get_available_teams_scenarios()

        if not available:
            st.warning("No data found. Check data directory.")
        else:
            _render_team_selector(available)

    # ---- RIGHT: Signature Browser ----
    with col_sigs:
        st.markdown("### 📈 Signatures")
        _render_signature_browser(registry)

    return _render_selection_summary()


def _render_team_selector(available: List[TeamScenario]):
    """Render team/scenario selector from available data"""

    # Group by data source then team
    by_source = {}
    for ts in available:
        by_source.setdefault(ts.data_source, {}).setdefault(ts.team_id, []).append(ts)

    source_names = {
        'entropy_ami': '📊 Entropy / AMI',
        'session_physio': '🧠 Session Physiological'
    }

    # Auto-detect: use all available sources (no manual radio selector)
    # If both exist, merge their teams; store source per-scenario at selection time
    sources = list(by_source.keys())
    if len(sources) == 1:
        source = sources[0]
        st.caption(f"Source: {source_names.get(source, source)}")
        teams_dict = by_source[source]
    else:
        # Merge all sources into a single teams dict
        teams_dict = {}
        for src, td in by_source.items():
            for tid, scenarios in td.items():
                teams_dict.setdefault(tid, []).extend(scenarios)
        src_labels = [source_names.get(s, s) for s in sources]
        st.caption(f"Sources: {' + '.join(src_labels)}")
    if not teams_dict:
        st.info("No teams for this source.")
        return

    # Team selector
    team_ids = sorted(teams_dict.keys(), key=lambda x: int(x) if x.isdigit() else x)
    sel_team = st.selectbox(
        "Team",
        ["Select..."] + team_ids,
        key="team_sel"
    )

    if sel_team == "Select...":
        return

    # Scenario selector
    scenarios = teams_dict[sel_team]
    scenario_map = {ts.scenario_id: ts for ts in scenarios}
    sel_scenario = st.selectbox(
        "Scenario / Session",
        list(scenario_map.keys()),
        format_func=lambda sid: scenario_map[sid].description,
        key="scenario_sel"
    )

    if sel_scenario:
        ts = scenario_map[sel_scenario]
        st.session_state.selected_team = _to_dict(TeamSelection(
            team_id=ts.team_id,
            scenario_id=ts.scenario_id,
            description=ts.description,
            data_source=ts.data_source,
            dce=ts.dce,
            day=ts.day,
            session=ts.session,
            session_label=ts.session_label,
            entropy_run=getattr(ts, 'entropy_run', None)
        ))
        st.success(f"✓ {ts.description}")


def _render_signature_browser(registry: SignatureRegistry):
    """Render signature browser with filters"""

    all_sigs = list(registry.get_all_signatures().values())

    # ---- Filters ----
    categories = ["All Categories"] + registry.get_categories()
    sel_cat = st.selectbox("Category", categories, key="cat_filter")

    # Signal type is auto-set to "All" (no longer user-selectable)
    sel_signal = "All"

    # Filter
    filtered = all_sigs
    if sel_cat != "All Categories":
        filtered = [s for s in filtered if s.category == sel_cat]

    st.caption(f"**{len(filtered)} signature templates** (auto-expanded by role)")
    st.markdown("---")

    # ---- Signature list ----
    selected_ids = {_get(s, 'data_signature_id') for s in st.session_state.selected_signatures}

    for sig in filtered:
        is_selected = sig.id in selected_ids

        c_check, c_info = st.columns([0.07, 0.93])

        with c_check:
            new_sel = st.checkbox(
                "", value=is_selected, key=f"sig_{sig.id}",
                label_visibility="collapsed"
            )

            if new_sel and not is_selected:
                # Determine signal type from filter
                signal_type = sel_signal if sel_signal != "All" else "All"

                # Build the SelectedSignature
                sel = _to_dict(SelectedSignature(
                    uri=f"data:{sig.id}",
                    label=sig.name,
                    modality=sig.sensor or sig.measure_type or "unknown",
                    modality_category=sig.category,
                    technique=sig.measure_type,
                    construct="",
                    data_signature_id=sig.id,
                    data_source=sig.data_source,
                    signal_type="All",
                    signal_types=["All"] if sig.data_source == 'entropy_ami' else None,
                    channel=None
                ))
                st.session_state.selected_signatures.append(sel)
                st.rerun()

            elif not new_sel and is_selected:
                st.session_state.selected_signatures = [
                    s for s in st.session_state.selected_signatures
                    if _get(s, 'data_signature_id') != sig.id
                ]
                st.rerun()

        with c_info:
            # Level badge
            level_badge = {"team": "🌐", "role": "👤", "summary": "📊"}.get(sig.level, "")
            source_badge = "📈" if sig.data_source == "entropy_ami" else "🧠"

            st.markdown(f"**{sig.name}** {level_badge} {source_badge}")

            detail_parts = [f"`{sig.category}`"]
            if sig.level == 'role':
                detail_parts.append("auto-expands × 5 roles")
            if sig.column_template:
                detail_parts.append(f"col: `{sig.column_template}`")
            elif sig.column_name:
                detail_parts.append(f"col: `{sig.column_name}`")
            st.caption(" · ".join(detail_parts))

            # EEG channel selector (inline)
            if sig.channels and is_selected:
                ch = st.selectbox(
                    f"Channel for {sig.name}",
                    sig.channels,
                    key=f"ch_{sig.id}",
                    label_visibility="collapsed"
                )
                # Update the stored signature's channel
                for s in st.session_state.selected_signatures:
                    if _get(s, 'data_signature_id') == sig.id:
                        if isinstance(s, dict):
                            s['channel'] = ch
                        else:
                            s.channel = ch

            # Signal type multi-select for entropy/AMI sigs (inline)
            if sig.data_source == 'entropy_ami' and is_selected:
                available_signals = registry.signal_type_ids  # e.g. ["All", "Neuro", "Auto"]
                if available_signals:
                    selected_signals = st.multiselect(
                        f"Signals for {sig.name}",
                        available_signals,
                        default=["All"],
                        key=f"sigtype_{sig.id}",
                        label_visibility="collapsed"
                    )
                    if not selected_signals:
                        selected_signals = ["All"]
                    # Update the stored signature's signal_types
                    for s in st.session_state.selected_signatures:
                        if _get(s, 'data_signature_id') == sig.id:
                            if isinstance(s, dict):
                                s['signal_types'] = selected_signals
                                s['signal_type'] = selected_signals[0]
                            else:
                                s.signal_types = selected_signals
                                s.signal_type = selected_signals[0]

        st.markdown("---")


# =============================================================================
# ONTOLOGY-DRIVEN SELECTION (fallback)
# =============================================================================

def render_ontology_driven_selection(
    onto: OntologyAccess
) -> Tuple[Optional[TeamSelection], List[SelectedSignature]]:
    """Original ontology-only selection when no data config exists"""

    st.info("💡 No `config/signatures.yaml` found. Showing ontology signatures (no data loading).")

    col_team, col_sigs = st.columns([1, 2], gap="large")

    with col_team:
        st.markdown("### 👥 Select Team")
        st.warning("Add a `config/signatures.yaml` and data files to enable real data.")

    with col_sigs:
        st.markdown("### 📈 Ontology Signatures")

        with st.expander("🔍 Filters", expanded=True):
            f1, f2, f3 = st.columns(3)
            with f1:
                mod_cats = ["All"] + onto.get_modality_categories()
                sel_mod = st.selectbox("Modality", mod_cats, key="filter_mod")
            with f2:
                constructs = onto.get_all_constructs()
                const_opts = ["All"] + [c.label for c in constructs]
                sel_const = st.selectbox("Construct", const_opts, key="filter_const")
            with f3:
                tech_cats = ["All"] + onto.get_technique_categories()
                sel_tech = st.selectbox("Technique", tech_cats, key="filter_tech")

        measures = onto.get_all_measures()
        filtered = measures
        if sel_mod != "All":
            filtered = [m for m in filtered if m.modality_category and sel_mod.lower() in m.modality_category.lower()]
        if sel_const != "All":
            filtered = [m for m in filtered if m.construct and sel_const.lower() in m.construct.lower()]
        if sel_tech != "All":
            filtered = [m for m in filtered if m.technique and sel_tech.lower() in m.technique.lower()]

        st.caption(f"Showing {len(filtered)} of {len(measures)} signatures")
        st.markdown("---")

        selected_uris = {_get(s, 'uri') for s in st.session_state.selected_signatures}

        for measure in filtered[:50]:
            is_selected = measure.uri in selected_uris
            c_check, c_info = st.columns([0.07, 0.93])

            with c_check:
                new_sel = st.checkbox(
                    "", value=is_selected, key=f"sig_{measure.uri}",
                    label_visibility="collapsed"
                )
                if new_sel and not is_selected:
                    st.session_state.selected_signatures.append(_to_dict(SelectedSignature(
                        uri=measure.uri,
                        label=measure.label,
                        modality=measure.modality or "Unknown",
                        modality_category=measure.modality_category or "Unknown",
                        technique=measure.technique or "Unknown",
                        construct=measure.construct or "Unknown"
                    )))
                    st.rerun()
                elif not new_sel and is_selected:
                    st.session_state.selected_signatures = [
                        s for s in st.session_state.selected_signatures
                        if _get(s, 'uri') != measure.uri
                    ]
                    st.rerun()

            with c_info:
                st.markdown(f"**{measure.label}**")
                tags = []
                if measure.modality_category:
                    tags.append(f"🏷️ {measure.modality_category}")
                if measure.construct:
                    tags.append(f"🎯 {measure.construct}")
                if tags:
                    st.caption(" | ".join(tags))

            st.markdown("---")

    return _render_selection_summary()


# =============================================================================
# SELECTION SUMMARY & ACTION
# =============================================================================

def _render_selection_summary() -> Tuple[Optional[TeamSelection], List[SelectedSignature]]:
    """Show summary of selections and the Analyze button"""

    sigs = st.session_state.selected_signatures
    team = st.session_state.selected_team

    if not sigs and not team:
        return None, []

    st.markdown("---")
    st.markdown("### 📋 Selection Summary")

    s1, s2 = st.columns(2)

    with s1:
        if team:
            st.markdown("**Team:**")
            st.markdown(f"- {_get(team, 'description', 'Unknown')}")
        else:
            st.caption("No team selected (evidence-only mode)")

    with s2:
        if sigs:
            st.markdown(f"**Signatures ({len(sigs)}):**")
            for sig in sigs:
                label = _get(sig, 'label', 'Unknown')
                uri = _get(sig, 'uri', '')
                c1, c2 = st.columns([0.85, 0.15])
                with c1:
                    extras = []
                    ch = _get(sig, 'channel')
                    if ch:
                        extras.append(ch)
                    sig_types = _get(sig, 'signal_types')
                    if sig_types and isinstance(sig_types, list):
                        if sig_types != ["All"]:
                            extras.append(", ".join(sig_types))
                    extra_str = f" ({'; '.join(extras)})" if extras else ""
                    st.markdown(f"- {label}{extra_str}")
                with c2:
                    if st.button("✕", key=f"rm_{uri}", help="Remove"):
                        st.session_state.selected_signatures = [
                            s for s in sigs if _get(s, 'uri') != uri
                        ]
                        st.rerun()

    st.markdown("---")
    _, btn_col = st.columns([3, 1])
    with btn_col:
        can_analyze = len(sigs) > 0
        if st.button("Analyze →", type="primary", disabled=not can_analyze):
            # Build return objects
            team_obj = None
            if team:
                team_obj = TeamSelection(**team) if isinstance(team, dict) else team

            sig_objs = []
            for s in sigs:
                if isinstance(s, SelectedSignature):
                    sig_objs.append(s)
                elif isinstance(s, dict):
                    # Filter to only SelectedSignature fields
                    valid_fields = {f.name for f in SelectedSignature.__dataclass_fields__.values()}
                    filtered = {k: v for k, v in s.items() if k in valid_fields}
                    sig_objs.append(SelectedSignature(**filtered))

            return team_obj, sig_objs

    return None, []


def render_back_button() -> bool:
    c1, _ = st.columns([1, 5])
    with c1:
        return st.button("← Back", key="back_selection")
