"""
UC1: Design a Measurement Strategy

Three-panel simultaneous filtering interface:
  Panel 1 — Select constructs (search + checkbox list)
  Panel 2 — Set context constraints (modality groups + level of analysis)
  Results  — Matching measures displayed as cards, table, or KG network

Modality grouping is fully ontology-driven via skos:broader.
Only modalities under the three signature parents (communication, physiology,
behavior) are shown.  Everything else is excluded automatically.
"""

import streamlit as st
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The three top-level modality parents that count as deployable signatures.
# Only modalities whose skos:broader matches one of these URIs will appear.
SIGNATURE_PARENT_URIS = {
    "http://example.org/ontology/teamMeasurement#communication",
    "http://example.org/ontology/teamMeasurement#physiology",
    "http://example.org/ontology/teamMeasurement#behavior",
}

# Human-friendly group names keyed by parent URI local name
PARENT_DISPLAY_NAMES = {
    "communication": "💬 Communication",
    "physiology": "🫀 Physiology",
    "behavior": "🏃 Behavior",
}

# Clean display labels for levels of analysis
LEVEL_DISPLAY: Dict[str, str] = {
    "individual": "Individual",
    "dyad": "Dyad",
    "dyads": "Dyads",
    "team": "Team",
    "individual_team": "Individual → Team",
    "cross_level": "Cross-Level",
    "group": "Group",
}


# ---------------------------------------------------------------------------
# Ontology-driven modality grouping
# ---------------------------------------------------------------------------

@dataclass
class ModalityInfo:
    """A modality instance from the ontology."""
    uri: str
    label: str
    parent_uri: Optional[str] = None
    parent_local: Optional[str] = None


def _extract_local(uri: str) -> str:
    """Extract local name from a URI."""
    s = str(uri)
    if "#" in s:
        return s.rsplit("#", 1)[-1]
    if "/" in s:
        return s.rsplit("/", 1)[-1]
    return s


def _build_modality_groups_from_ontology(onto) -> Dict[str, List[ModalityInfo]]:
    """
    Query the ontology for all modalities with skos:broader, then group them
    under the three signature parents.  Fully dynamic — new modalities added
    to the TTL with the right skos:broader appear automatically.

    Returns {group_display_name: [ModalityInfo, ...]} ordered by group then label.
    """
    query = """
    PREFIX meas: <http://example.org/ontology/teamMeasurement#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    SELECT ?uri ?label ?broader WHERE {
        ?uri a meas:Modality .
        ?uri skos:broader ?broader .
        OPTIONAL { ?uri rdfs:label ?label }
    }
    ORDER BY ?broader ?label
    """
    groups: Dict[str, List[ModalityInfo]] = {}
    allowed_parents = SIGNATURE_PARENT_URIS

    for row in onto.graph.query(query):
        broader_uri = str(row.broader)
        if broader_uri not in allowed_parents:
            continue

        parent_local = _extract_local(broader_uri)
        group_name = PARENT_DISPLAY_NAMES.get(parent_local, parent_local.replace("_", " ").title())

        info = ModalityInfo(
            uri=str(row.uri),
            label=str(row.label) if row.label else _extract_local(str(row.uri)),
            parent_uri=broader_uri,
            parent_local=parent_local,
        )

        groups.setdefault(group_name, []).append(info)

    return groups


def _get_signature_modality_uris(modality_groups: Dict[str, List[ModalityInfo]]) -> set:
    """Return the set of modality URIs that are valid signatures."""
    uris = set()
    for members in modality_groups.values():
        for m in members:
            uris.add(m.uri)
    return uris


# ---------------------------------------------------------------------------
# Measure helpers
# ---------------------------------------------------------------------------

def _level_local(measure) -> Optional[str]:
    """Get the level local name for a measure."""
    raw = measure.level
    if raw is None:
        return None
    s = str(raw)
    if "#" in s:
        s = s.rsplit("#", 1)[-1]
    if s.startswith("level_"):
        s = s[len("level_"):]
    return s.lower().replace(" ", "_")


def _build_modality_label_to_uri(modality_groups: Dict[str, List[ModalityInfo]]) -> Dict[str, str]:
    """Build a mapping from modality label (lowercase) to URI for matching."""
    result = {}
    for members in modality_groups.values():
        for m in members:
            result[m.label.lower()] = m.uri
            # Also index by local name from URI
            local = _extract_local(m.uri)
            result[local.lower()] = m.uri
            # Without modality_ prefix
            if local.lower().startswith("modality_"):
                result[local.lower()[len("modality_"):]] = m.uri
    return result


def _resolve_measure_modality_uri(measure, label_to_uri: Dict[str, str]) -> Optional[str]:
    """Resolve a measure's modality to a URI using the label-to-URI map."""
    raw = measure.modality
    if raw is None:
        return None
    # Try direct label match
    key = str(raw).lower()
    if key in label_to_uri:
        return label_to_uri[key]
    # Try with underscores instead of spaces
    key_under = key.replace(" ", "_").replace("-", "_")
    if key_under in label_to_uri:
        return label_to_uri[key_under]
    # Try modality_category
    cat = measure.modality_category
    if cat:
        cat_key = str(cat).lower()
        if cat_key in label_to_uri:
            return label_to_uri[cat_key]
        cat_under = cat_key.replace(" ", "_").replace("-", "_")
        if cat_under in label_to_uri:
            return label_to_uri[cat_under]
    return None


# ---------------------------------------------------------------------------
# Filtering logic
# ---------------------------------------------------------------------------

def filter_measures_multi(
    all_measures,
    selected_constructs: List[str],
    selected_modality_uris: set,
    selected_levels: List[str],
    signature_uris: set,
    label_to_uri: Dict[str, str],
) -> list:
    """
    Filter measures by multiple simultaneous selections.

    Logic:
      - Only measures whose modality is a valid signature are included
      - AND across dimensions (construct AND modality AND level)
      - OR within a dimension (matches any of the selected values)
      - Empty selection on a dimension = no filter on that dimension
    """
    results = []
    for m in all_measures:
        # Resolve modality to URI and check it's a signature type
        mod_uri = _resolve_measure_modality_uri(m, label_to_uri)
        if mod_uri is None or mod_uri not in signature_uris:
            continue

        # Construct filter
        if selected_constructs:
            m_construct = (m.construct or "").lower()
            if not any(c.lower() in m_construct for c in selected_constructs):
                continue

        # Modality filter
        if selected_modality_uris:
            if mod_uri not in selected_modality_uris:
                continue

        # Level filter
        if selected_levels:
            m_level = _level_local(m)
            if m_level is None:
                continue
            if m_level not in selected_levels:
                continue

        results.append(m)

    return results


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------

def _render_construct_panel(constructs) -> List[str]:
    """Panel 1: Construct selection with search + checkboxes."""
    st.markdown("#### 🎯 Constructs")
    st.caption("What team dynamics do you want to measure?")

    # Search filter
    search = st.text_input(
        "Search constructs",
        key="uc1_construct_search",
        placeholder="Type to filter…",
        label_visibility="collapsed",
    )

    # Filter construct list
    search_lower = search.strip().lower()
    visible = [
        c for c in constructs
        if not search_lower or search_lower in c.label.lower()
    ]

    # Select all / clear
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Select all", key="uc1_sel_all_constructs", use_container_width=True):
            for c in visible:
                st.session_state[f"uc1_c_{c.label}"] = True
    with col_b:
        if st.button("Clear", key="uc1_clr_constructs", use_container_width=True):
            for c in constructs:
                st.session_state[f"uc1_c_{c.label}"] = False

    # Checkboxes
    selected = []
    for c in visible:
        key = f"uc1_c_{c.label}"
        checked = st.checkbox(c.label, key=key)
        if checked:
            selected.append(c.label)

    n = len(selected)
    st.caption(f"**{n}** construct{'s' if n != 1 else ''} selected")
    return selected


def _render_constraint_panel(
    modality_groups: Dict[str, List[ModalityInfo]],
    available_levels: List[str],
) -> Tuple[set, List[str]]:
    """Panel 2: Context constraints — ontology-driven modality groups + level of analysis."""
    st.markdown("#### 🔬 Available Modalities")
    st.caption("What data sources can you collect?")

    selected_modality_uris = set()

    # Render groups in a stable order matching PARENT_DISPLAY_NAMES
    group_order = list(PARENT_DISPLAY_NAMES.values())
    ordered_groups = []
    for gn in group_order:
        if gn in modality_groups:
            ordered_groups.append((gn, modality_groups[gn]))
    # Any groups not in the predefined order (future-proofing)
    for gn, members in modality_groups.items():
        if gn not in group_order:
            ordered_groups.append((gn, members))

    for group_name, members in ordered_groups:
        with st.expander(f"**{group_name}** ({len(members)})", expanded=True):
            for mod in members:
                # Use the ontology label directly, clean up for display
                display = mod.label
                # Title-case if it's all lowercase (likely a local name)
                if display == display.lower():
                    display = display.replace("_", " ").title()
                key = f"uc1_m_{mod.uri}"
                if st.checkbox(display, key=key):
                    selected_modality_uris.add(mod.uri)

    st.markdown("---")
    st.markdown("#### 📐 Level of Analysis")
    st.caption("At what level do you need measurements?")

    selected_levels = []
    for level_local in available_levels:
        display = LEVEL_DISPLAY.get(level_local, level_local.replace("_", " ").title())
        key = f"uc1_l_{level_local}"
        if st.checkbox(display, key=key):
            selected_levels.append(level_local)

    return selected_modality_uris, selected_levels


def _render_results_cards(measures):
    """Card view of matching measures."""
    for m in measures:
        with st.expander(f"**{m.label}**  —  {m.construct or 'No construct'}"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Modality:** {m.modality or '—'}")
            with col2:
                st.markdown(f"**Level:** {m.level or '—'}")
            with col3:
                st.markdown(f"**Technique:** {m.technique or '—'}")

            if m.description:
                desc = m.description
                if len(desc) > 500:
                    desc = desc[:500] + "…"
                st.markdown(f"**Description:** {desc}")

            # Evidence placeholder
            st.markdown("---")
            st.markdown("📚 **Evidence Summary**")
            st.info(
                "Evidence synthesis is in progress. "
                "Effect sizes, study counts, and forest plots will appear here "
                "once the evidence layer is complete."
            )


def _render_results_table(measures):
    """Table view of matching measures."""
    import pandas as pd

    if not measures:
        return

    rows = []
    for m in measures:
        rows.append({
            "Measure": m.label,
            "Construct": m.construct or "—",
            "Modality": m.modality or "—",
            "Level": m.level or "—",
            "Technique": m.technique or "—",
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Measure": st.column_config.TextColumn(width="large"),
            "Construct": st.column_config.TextColumn(width="medium"),
            "Modality": st.column_config.TextColumn(width="medium"),
            "Level": st.column_config.TextColumn(width="small"),
            "Technique": st.column_config.TextColumn(width="medium"),
        },
    )


def _render_kg_network(measures, onto):
    """KG network view of filtered measures and their relationships."""
    try:
        from pyvis.network import Network
        import tempfile
        import streamlit.components.v1 as components
    except ImportError:
        st.warning("pyvis is required for network visualization. Install with: pip install pyvis")
        return

    if not measures:
        return

    net = Network(
        height="600px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#333333",
        directed=False,
    )
    net.set_options("""
    {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -40,
                "centralGravity": 0.005,
                "springLength": 150,
                "springConstant": 0.04
            },
            "solver": "forceAtlas2Based",
            "stabilization": {"iterations": 150}
        },
        "nodes": {
            "font": {"size": 14, "face": "arial"}
        },
        "edges": {
            "smooth": {"type": "continuous"},
            "color": {"opacity": 0.5}
        }
    }
    """)

    # Color scheme
    COLORS = {
        "Measure": "#4F46E5",     # indigo
        "Construct": "#059669",   # emerald
        "Modality": "#D97706",    # amber
        "Technique": "#7C3AED",   # violet
    }

    added_nodes = set()

    def add_node(node_id, label, group):
        if node_id not in added_nodes:
            net.add_node(
                node_id,
                label=label,
                title=f"{group}: {label}",
                color=COLORS.get(group, "#6B7280"),
                shape="dot" if group == "Measure" else "diamond",
                size=12 if group == "Measure" else 18,
            )
            added_nodes.add(node_id)

    for m in measures:
        m_id = m.uri
        add_node(m_id, m.label, "Measure")

        if m.construct:
            c_id = f"construct:{m.construct}"
            add_node(c_id, m.construct, "Construct")
            net.add_edge(m_id, c_id)

        if m.modality:
            mod_id = f"modality:{m.modality}"
            add_node(mod_id, m.modality, "Modality")
            net.add_edge(m_id, mod_id)

        if m.technique:
            t_id = f"technique:{m.technique}"
            add_node(t_id, m.technique, "Technique")
            net.add_edge(m_id, t_id)

    # Render to temp HTML and embed
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        net.write_html(f.name, notebook=False, open_browser=False)
        f.seek(0)
        html_content = open(f.name, "r").read()

    components.html(html_content, height=620, scrolling=True)

    # Legend
    legend_cols = st.columns(4)
    for i, (group, color) in enumerate(COLORS.items()):
        with legend_cols[i]:
            st.markdown(
                f'<span style="color:{color}; font-weight:600;">● {group}</span>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def render_measurement_strategy(onto) -> None:
    """Render the UC1 Measurement Strategy view."""

    st.markdown("## 🎯 Design a Measurement Strategy")
    st.markdown(
        "Select the constructs you want to measure and the data sources available "
        "to you. Matching signatures are shown below."
    )
    st.markdown("---")

    # ----- Load data from ontology -----
    all_constructs = onto.get_all_constructs()
    all_measures = onto.get_all_measures()

    # Deduplicate measures by URI (the SPARQL query can return duplicates
    # when a measure has multiple modalities/techniques)
    seen_uris = set()
    unique_measures = []
    for m in all_measures:
        if m.uri not in seen_uris:
            seen_uris.add(m.uri)
            unique_measures.append(m)
    all_measures = unique_measures

    # Build modality groups dynamically from ontology skos:broader
    modality_groups = _build_modality_groups_from_ontology(onto)
    signature_uris = _get_signature_modality_uris(modality_groups)
    label_to_uri = _build_modality_label_to_uri(modality_groups)

    # Available levels from actual data
    available_levels = set()
    for m in all_measures:
        ll = _level_local(m)
        if ll:
            available_levels.add(ll)
    available_levels = sorted(available_levels)

    # Count total signature measures (for the summary bar)
    total_signatures = sum(
        1 for m in all_measures
        if _resolve_measure_modality_uri(m, label_to_uri) in signature_uris
    )

    # ----- Layout: two filter panels side-by-side, results below -----
    panel1, panel2 = st.columns([1, 1])

    with panel1:
        selected_constructs = _render_construct_panel(all_constructs)

    with panel2:
        selected_modality_uris, selected_levels = _render_constraint_panel(
            modality_groups, available_levels
        )

    # ----- Filter -----
    filtered = filter_measures_multi(
        all_measures,
        selected_constructs,
        selected_modality_uris,
        selected_levels,
        signature_uris,
        label_to_uri,
    )

    # ----- Results -----
    st.markdown("---")

    n_filtered = len(filtered)

    active_filters = []
    if selected_constructs:
        active_filters.append(f"{len(selected_constructs)} construct{'s' if len(selected_constructs) != 1 else ''}")
    if selected_modality_uris:
        active_filters.append(f"{len(selected_modality_uris)} modalit{'ies' if len(selected_modality_uris) != 1 else 'y'}")
    if selected_levels:
        active_filters.append(f"{len(selected_levels)} level{'s' if len(selected_levels) != 1 else ''}")

    if active_filters:
        filter_desc = ", ".join(active_filters)
        st.markdown(f"### Matching Signatures: **{n_filtered}** of {total_signatures}  \n*Filtering by {filter_desc}*")
    else:
        st.markdown(f"### All Signatures ({total_signatures})")
        st.caption("Use the panels above to narrow results.")

    # Tabbed results
    if n_filtered == 0 and active_filters:
        st.warning("No measures match the current filter combination. Try broadening your selections.")
    else:
        tab_cards, tab_table, tab_network = st.tabs(["📋 Cards", "📊 Table", "🌐 Network"])

        with tab_cards:
            _render_results_cards(filtered)

        with tab_table:
            _render_results_table(filtered)

        with tab_network:
            if n_filtered > 100:
                st.warning(
                    f"Showing network for {n_filtered} measures may be slow. "
                    "Consider narrowing filters for a clearer visualization."
                )
            _render_kg_network(filtered, onto)


def render_back_button() -> bool:
    """Render a back button; returns True if clicked."""
    col1, _ = st.columns([1, 5])
    with col1:
        return st.button("← Back", key="uc1_back")
