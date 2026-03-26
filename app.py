"""
BioTDMS Explorer - Main Application

A system for exploring team performance signatures and their evidence base.
"""

import streamlit as st
from pathlib import Path
import sys
from views.data_management import render_data_panel

# Path setup
_current_file = Path(__file__).resolve() if '__file__' in dir() else Path.cwd() / 'app.py'
_app_dir = _current_file.parent
if str(_app_dir) not in sys.path:
    sys.path.insert(0, str(_app_dir))
_cwd = Path.cwd()
if str(_cwd) not in sys.path:
    sys.path.insert(0, str(_cwd))

# Local imports
from core.ontology import OntologyAccess
from views.landing import render_landing_page_styled
from views.uc1_measurement_strategy import (
    render_measurement_strategy, render_back_button as uc1_back
)
from views.uc2_signature_selection import (
    render_signature_selection, render_back_button as sig_back,
    SelectedSignature, TeamSelection
)
from views.uc2_analysis import render_analysis_results, render_back_button as analysis_back

# Page config
st.set_page_config(
    page_title="BioTDMS Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main { padding-top: 1rem; }
    h1, h2, h3 { color: #1f2937; }
    .stExpander { border: 1px solid #e5e7eb; border-radius: 8px; }
    .stButton > button { border-radius: 6px; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def init_state():
    defaults = {
        'current_view': 'landing',
        'selected_team': None,
        'selected_signatures': [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def reset_to_landing():
    st.session_state.current_view = 'landing'
    st.session_state.selected_team = None
    st.session_state.selected_signatures = []
    st.rerun()


@st.cache_resource
def load_ontology():
    paths = [
        _app_dir / "instances.ttl",
        _app_dir / "data" / "instances.ttl",
        _app_dir / "data" / "ontologies" / "instances.ttl",
        _cwd / "instances.ttl",
        Path("instances.ttl"),
    ]
    for path in paths:
        if path.exists():
            return OntologyAccess(path)
    raise FileNotFoundError(
        f"Could not find instances.ttl. Looked in: {[str(p) for p in paths]}"
    )


def _deserialize_team(raw) -> TeamSelection | None:
    """Convert session state team to TeamSelection"""
    if raw is None:
        return None
    if isinstance(raw, TeamSelection):
        return raw
    if isinstance(raw, dict):
        valid = {f for f in TeamSelection.__dataclass_fields__}
        filtered = {k: v for k, v in raw.items() if k in valid}
        return TeamSelection(**filtered)
    return raw


def _deserialize_sigs(raw_list) -> list[SelectedSignature]:
    """Convert session state signatures to SelectedSignature list"""
    results = []
    for s in raw_list:
        if isinstance(s, SelectedSignature):
            results.append(s)
        elif isinstance(s, dict):
            valid = {f for f in SelectedSignature.__dataclass_fields__}
            filtered = {k: v for k, v in s.items() if k in valid}
            results.append(SelectedSignature(**filtered))
    return results


def main():
    init_state()

    try:
        onto = load_ontology()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    # Header
    h1, h2 = st.columns([4, 1])
    with h1:
        st.title("BioTDMS Explorer")
    with h2:
        if st.session_state.current_view != 'landing':
            if st.button("🏠 Home"):
                reset_to_landing()

    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Ontology Stats")
        stats = onto.get_statistics()
        st.metric("Measures", stats['total_measures'])
        st.metric("Constructs", stats['total_constructs'])
        st.metric("Modalities", stats['total_modalities'])
        st.metric("Techniques", stats['total_techniques'])
        st.markdown("---")
        st.caption(f"Triples: {stats['total_triples']}")
    # Data management panel (sidebar)
    render_data_panel(_app_dir)
    
    # Routing
    view = st.session_state.current_view

    if view == 'landing':
        flow = render_landing_page_styled()
        if flow == 'measurement':
            st.session_state.current_view = 'uc1_competency'
            st.rerun()
        elif flow == 'performance':
            st.session_state.current_view = 'uc2_selection'
            st.rerun()

    elif view == 'uc1_competency':
        if uc1_back():
            reset_to_landing()
        render_measurement_strategy(onto)

    elif view == 'uc2_selection':
        if sig_back():
            reset_to_landing()

        team, signatures = render_signature_selection(onto)
        if signatures:
            st.session_state.selected_team = team
            st.session_state.selected_signatures = signatures
            st.session_state.current_view = 'uc2_analysis'
            st.rerun()

    elif view == 'uc2_analysis':
        if analysis_back():
            st.session_state.current_view = 'uc2_selection'
            st.rerun()

        team = _deserialize_team(st.session_state.selected_team)
        signatures = _deserialize_sigs(st.session_state.selected_signatures)
        render_analysis_results(onto, team, signatures)

    else:
        reset_to_landing()

    # Footer
    st.markdown("---")
    st.caption("BioTDMS Explorer v0.3 | Built with Streamlit")


if __name__ == "__main__":
    main()
