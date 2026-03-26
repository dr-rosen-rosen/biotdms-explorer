"""
Measure Neighborhood Graph

Queries the ontology for a signature's conceptual neighborhood
and renders it as a Plotly node-link diagram for embedding in
the UC2 inline evidence expanders.

Supports two query strategies:
  1. URI-based: exact measure lookup + 1-hop traversal (when sig.uri
     matches an ontology measure)
  2. Metadata-based: search by construct, modality, and technique
     keywords from the SelectedSignature fields (always works)

The metadata strategy produces richer results for aggregate signatures
(e.g., "Team Entropy across all signals") that don't map 1:1 to a
single ontology measure.
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
import math

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GraphNode:
    id: str
    label: str
    group: str  # "focal", "construct", "modality", "technique", "measure", "evidence_stub"
    description: Optional[str] = None


@dataclass
class GraphEdge:
    source: str
    target: str
    label: str


@dataclass
class NeighborhoodGraph:
    focal_id: str
    nodes: List[GraphNode] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)

    def node_by_id(self, node_id: str) -> Optional[GraphNode]:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None


# ---------------------------------------------------------------------------
# Colors and styling
# ---------------------------------------------------------------------------

GROUP_STYLES = {
    "focal":          {"color": "#dc2626", "size": 18, "symbol": "star"},
    "construct":      {"color": "#059669", "size": 14, "symbol": "diamond"},
    "modality":       {"color": "#d97706", "size": 12, "symbol": "square"},
    "technique":      {"color": "#7c3aed", "size": 12, "symbol": "triangle-up"},
    "measure":        {"color": "#6366f1", "size": 9,  "symbol": "circle"},
    "evidence_stub":  {"color": "#9ca3af", "size": 8,  "symbol": "circle"},
}

GROUP_LABELS = {
    "focal": "This Signature",
    "construct": "Construct",
    "modality": "Modality",
    "technique": "Technique",
    "measure": "Related Measure",
    "evidence_stub": "Evidence (coming)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_local(uri: str) -> str:
    s = str(uri)
    if "#" in s:
        return s.rsplit("#", 1)[-1]
    if "/" in s:
        return s.rsplit("/", 1)[-1]
    return s


def _clean_label(label: str, max_len: int = 35) -> str:
    """Clean up a label for display."""
    label = label.strip()
    if len(label) > max_len:
        return label[:max_len - 1] + "…"
    return label


# ---------------------------------------------------------------------------
# URI-based query (original, for when sig.uri is a real ontology URI)
# ---------------------------------------------------------------------------

def _try_uri_query(onto, measure_uri: str) -> Optional[NeighborhoodGraph]:
    """
    Attempt a direct URI lookup. Returns None if the URI isn't found
    in the ontology (i.e., it's a YAML config ID, not an ontology URI).
    """
    check = """
    PREFIX meas: <http://example.org/ontology/teamMeasurement#>
    ASK { <%s> a meas:Measure }
    """ % measure_uri
    try:
        result = bool(onto.graph.query(check))
        if not result:
            return None
    except Exception:
        return None

    # URI exists — proceed with direct neighborhood query
    return _query_uri_neighborhood(onto, measure_uri)


def _query_uri_neighborhood(onto, measure_uri: str, max_siblings: int = 8) -> NeighborhoodGraph:
    """Full URI-based neighborhood query."""
    graph = NeighborhoodGraph(focal_id=measure_uri)
    added: Set[str] = set()

    def add_node(nid, label, group, desc=None):
        if nid not in added:
            graph.nodes.append(GraphNode(nid, label, group, desc))
            added.add(nid)

    def add_edge(src, tgt, label):
        graph.edges.append(GraphEdge(src, tgt, label))

    query = """
    PREFIX meas: <http://example.org/ontology/teamMeasurement#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?label ?desc ?construct ?cLabel ?modality ?mLabel ?technique ?tLabel
    WHERE {
        BIND(<%s> AS ?measure)
        ?measure a meas:Measure .
        OPTIONAL { ?measure rdfs:label ?label }
        OPTIONAL { ?measure meas:hasDescription ?desc }
        OPTIONAL { ?measure meas:measuresConstruct ?construct . OPTIONAL { ?construct rdfs:label ?cLabel } }
        OPTIONAL { ?measure meas:includesModality ?modality . OPTIONAL { ?modality rdfs:label ?mLabel } }
        OPTIONAL { ?measure meas:usesAnalyticTechnique ?technique . OPTIONAL { ?technique rdfs:label ?tLabel } }
    }
    """ % measure_uri

    construct_uris = set()
    for row in onto.graph.query(query):
        add_node(measure_uri, str(row.label) if row.label else _extract_local(measure_uri),
                 "focal", str(row.desc)[:200] if row.desc else None)
        if row.construct:
            c = str(row.construct)
            add_node(c, str(row.cLabel) if row.cLabel else _extract_local(c), "construct")
            add_edge(measure_uri, c, "measuresConstruct")
            construct_uris.add(c)
        if row.modality:
            m = str(row.modality)
            add_node(m, str(row.mLabel) if row.mLabel else _extract_local(m), "modality")
            add_edge(measure_uri, m, "includesModality")
        if row.technique:
            t = str(row.technique)
            add_node(t, str(row.tLabel) if row.tLabel else _extract_local(t), "technique")
            add_edge(measure_uri, t, "usesAnalyticTechnique")

    # Siblings sharing constructs
    for c_uri in construct_uris:
        sib_query = """
        PREFIX meas: <http://example.org/ontology/teamMeasurement#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT DISTINCT ?s ?sLabel WHERE {
            ?s a meas:Measure . ?s meas:measuresConstruct <%s> .
            FILTER(?s != <%s>)
            OPTIONAL { ?s rdfs:label ?sLabel }
        } LIMIT %d
        """ % (c_uri, measure_uri, max_siblings)
        for row in onto.graph.query(sib_query):
            s = str(row.s)
            add_node(s, str(row.sLabel) if row.sLabel else _extract_local(s), "measure")
            add_edge(s, c_uri, "measuresConstruct")

    return graph


# ---------------------------------------------------------------------------
# Metadata-based query (broadened, works from signature fields)
# ---------------------------------------------------------------------------

def query_by_metadata(
    onto,
    sig_label: str,
    construct: Optional[str] = None,
    modality_category: Optional[str] = None,
    technique: Optional[str] = None,
    max_measures: int = 12,
) -> NeighborhoodGraph:
    """
    Build a neighborhood graph from signature metadata rather than a URI.

    Strategy — cascading search with union of results:
      1. Create a virtual focal node from the signature fields
      2. Find measures by construct match (if construct provided)
      3. Find measures by technique keyword match (e.g., "entropy", "shannon")
      4. Find measures by label keyword match (e.g., signature name terms)
      5. For all found measures, show their constructs, modalities, and techniques

    This produces rich graphs even for aggregate signatures like "Team Entropy"
    that span multiple ontology constructs and modalities.
    """
    focal_id = f"sig:{sig_label}"
    graph = NeighborhoodGraph(focal_id=focal_id)
    added: Set[str] = set()
    found_measure_uris: Set[str] = set()

    def add_node(nid, label, group, desc=None):
        if nid not in added:
            graph.nodes.append(GraphNode(nid, label, group, desc))
            added.add(nid)

    def add_edge(src, tgt, label):
        graph.edges.append(GraphEdge(src, tgt, label))

    def _add_measure_with_context(row, connect_to_focal=False):
        """Helper to add a measure and its construct/modality/technique from a query row."""
        m_uri = str(row.m)
        if m_uri in found_measure_uris and not connect_to_focal:
            return  # Already added
        m_label = str(row.mLabel) if row.mLabel else _extract_local(m_uri)
        found_measure_uris.add(m_uri)
        add_node(m_uri, m_label, "measure")

        if row.construct:
            c_uri = str(row.construct)
            c_label = str(row.cLabel) if row.cLabel else _extract_local(c_uri)
            add_node(c_uri, c_label, "construct")
            add_edge(m_uri, c_uri, "measuresConstruct")
        if row.mod:
            mod_uri = str(row.mod)
            mod_label = str(row.modLabel) if row.modLabel else _extract_local(mod_uri)
            add_node(mod_uri, mod_label, "modality")
            add_edge(m_uri, mod_uri, "includesModality")
        if row.tech:
            tech_uri = str(row.tech)
            tech_label = str(row.techLabel) if row.techLabel else _extract_local(tech_uri)
            add_node(tech_uri, tech_label, "technique")
            add_edge(m_uri, tech_uri, "usesAnalyticTechnique")

    # Focal node (virtual — represents the UC2 signature)
    add_node(focal_id, sig_label, "focal",
             f"Construct: {construct or '—'}\nModality: {modality_category or '—'}")

    # Build search keywords from all available signature metadata
    search_keywords = set()
    for field in [sig_label, construct, modality_category, technique]:
        if field:
            for word in field.lower().replace("_", " ").replace("/", " ").replace("-", " ").split():
                if len(word) > 3 and word not in {'team', 'role', 'level', 'data', 'signal', 'type', 'none'}:
                    search_keywords.add(word)

    # ---- Strategy A: Construct match ----
    construct_uris = set()
    if construct:
        construct_lower = construct.lower().replace("_", " ")
        q_constructs = """
        PREFIX meas: <http://example.org/ontology/teamMeasurement#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT DISTINCT ?c ?cLabel WHERE {
            ?c a meas:Construct .
            OPTIONAL { ?c rdfs:label ?cLabel }
        }
        """
        for row in onto.graph.query(q_constructs):
            c_uri = str(row.c)
            c_label = str(row.cLabel) if row.cLabel else _extract_local(c_uri)
            c_lower = c_label.lower().replace("_", " ")
            local_lower = _extract_local(c_uri).lower().replace("_", " ")
            if (construct_lower in c_lower or c_lower in construct_lower
                    or construct_lower in local_lower or local_lower in construct_lower):
                add_node(c_uri, c_label, "construct")
                add_edge(focal_id, c_uri, "measuresConstruct")
                construct_uris.add(c_uri)

    # Fetch measures by construct
    for c_uri in construct_uris:
        if len(found_measure_uris) >= max_measures:
            break
        q = """
        PREFIX meas: <http://example.org/ontology/teamMeasurement#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT DISTINCT ?m ?mLabel ?construct ?cLabel ?mod ?modLabel ?tech ?techLabel WHERE {
            ?m a meas:Measure . ?m meas:measuresConstruct <%s> .
            BIND(<%s> AS ?construct) .
            OPTIONAL { <%s> rdfs:label ?cLabel }
            OPTIONAL { ?m rdfs:label ?mLabel }
            OPTIONAL { ?m meas:includesModality ?mod . OPTIONAL { ?mod rdfs:label ?modLabel } }
            OPTIONAL { ?m meas:usesAnalyticTechnique ?tech . OPTIONAL { ?tech rdfs:label ?techLabel } }
        } LIMIT %d
        """ % (c_uri, c_uri, c_uri, max_measures)
        for row in onto.graph.query(q):
            if len(found_measure_uris) >= max_measures:
                break
            _add_measure_with_context(row)

    # ---- Strategy B: Technique keyword match ----
    # This catches entropy measures across ALL constructs
    if search_keywords and len(found_measure_uris) < max_measures:
        for keyword in search_keywords:
            if len(found_measure_uris) >= max_measures:
                break
            q_tech = """
            PREFIX meas: <http://example.org/ontology/teamMeasurement#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT DISTINCT ?m ?mLabel ?construct ?cLabel ?mod ?modLabel ?tech ?techLabel WHERE {
                ?m a meas:Measure .
                ?m meas:usesAnalyticTechnique ?tech .
                ?tech rdfs:label ?techLabel .
                FILTER(CONTAINS(LCASE(STR(?techLabel)), "%s"))
                OPTIONAL { ?m rdfs:label ?mLabel }
                OPTIONAL { ?m meas:measuresConstruct ?construct . OPTIONAL { ?construct rdfs:label ?cLabel } }
                OPTIONAL { ?m meas:includesModality ?mod . OPTIONAL { ?mod rdfs:label ?modLabel } }
            } LIMIT %d
            """ % (keyword, max_measures - len(found_measure_uris))
            for row in onto.graph.query(q_tech):
                if len(found_measure_uris) >= max_measures:
                    break
                _add_measure_with_context(row)

    # ---- Strategy C: Label keyword match ----
    # Finds measures whose name contains relevant terms
    if search_keywords and len(found_measure_uris) < max_measures:
        for keyword in search_keywords:
            if len(found_measure_uris) >= max_measures:
                break
            q_label = """
            PREFIX meas: <http://example.org/ontology/teamMeasurement#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT DISTINCT ?m ?mLabel ?construct ?cLabel ?mod ?modLabel ?tech ?techLabel WHERE {
                ?m a meas:Measure .
                ?m rdfs:label ?mLabel .
                FILTER(CONTAINS(LCASE(STR(?mLabel)), "%s"))
                OPTIONAL { ?m meas:measuresConstruct ?construct . OPTIONAL { ?construct rdfs:label ?cLabel } }
                OPTIONAL { ?m meas:includesModality ?mod . OPTIONAL { ?mod rdfs:label ?modLabel } }
                OPTIONAL { ?m meas:usesAnalyticTechnique ?tech . OPTIONAL { ?tech rdfs:label ?techLabel } }
            } LIMIT %d
            """ % (keyword, max_measures - len(found_measure_uris))
            for row in onto.graph.query(q_label):
                if len(found_measure_uris) >= max_measures:
                    break
                _add_measure_with_context(row)

    # ---- Strategy D: Modality keyword match (last resort) ----
    if not found_measure_uris and modality_category:
        modality_lower = modality_category.lower()
        q_mod = """
        PREFIX meas: <http://example.org/ontology/teamMeasurement#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT DISTINCT ?m ?mLabel ?construct ?cLabel ?mod ?modLabel ?tech ?techLabel WHERE {
            ?m a meas:Measure .
            ?m meas:includesModality ?mod .
            ?mod rdfs:label ?modLabel .
            FILTER(CONTAINS(LCASE(STR(?modLabel)), "%s"))
            OPTIONAL { ?m rdfs:label ?mLabel }
            OPTIONAL { ?m meas:measuresConstruct ?construct . OPTIONAL { ?construct rdfs:label ?cLabel } }
            OPTIONAL { ?m meas:usesAnalyticTechnique ?tech . OPTIONAL { ?tech rdfs:label ?techLabel } }
        } LIMIT %d
        """ % (modality_lower, max_measures)
        for row in onto.graph.query(q_mod):
            _add_measure_with_context(row)

    # Connect focal node to constructs discovered through measures
    # (constructs found via technique/label match that weren't in Strategy A)
    for node in graph.nodes:
        if node.group == "construct" and node.id not in construct_uris:
            add_edge(focal_id, node.id, "relatedConstruct")

    # Evidence stub
    stub_id = f"evidence_stub_{focal_id}"
    add_node(stub_id, "Evidence\n(coming soon)", "evidence_stub")
    add_edge(focal_id, stub_id, "hasEvidence")

    return graph


# ---------------------------------------------------------------------------
# Combined query: try URI first, fall back to metadata
# ---------------------------------------------------------------------------

def query_signature_neighborhood(
    onto,
    sig_uri: str,
    sig_label: str = "",
    construct: Optional[str] = None,
    modality_category: Optional[str] = None,
    technique: Optional[str] = None,
    max_measures: int = 12,
) -> NeighborhoodGraph:
    """
    Build the best possible neighborhood graph for a signature.

    Tries URI-based lookup first (fast, exact).
    Falls back to metadata-based search (broader, always produces results
    if the construct or modality exists in the ontology).
    """
    # Try exact URI match first
    if sig_uri and "://" in sig_uri:
        result = _try_uri_query(onto, sig_uri)
        if result and len(result.nodes) > 2:  # More than just focal + evidence stub
            return result

    # Fall back to metadata-based search
    return query_by_metadata(
        onto,
        sig_label=sig_label or sig_uri or "Unknown Signature",
        construct=construct,
        modality_category=modality_category,
        technique=technique,
        max_measures=max_measures,
    )


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def _layout_neighborhood(graph: NeighborhoodGraph) -> Dict[str, Tuple[float, float]]:
    """
    Radial layout: focal at center, constructs/modalities/techniques
    in inner ring, measures in outer ring clustered by construct.
    """
    positions: Dict[str, Tuple[float, float]] = {}

    # Focal at origin
    positions[graph.focal_id] = (0.0, 0.0)

    # Inner ring: constructs, modalities, techniques, evidence stubs
    inner = [n for n in graph.nodes if n.group in ("construct", "modality", "technique", "evidence_stub")]
    inner_r = 2.0
    for i, node in enumerate(inner):
        angle = (2 * math.pi * i) / max(len(inner), 1) + math.pi / 2
        positions[node.id] = (inner_r * math.cos(angle), inner_r * math.sin(angle))

    # Outer ring: measures, clustered near their construct
    measures = [n for n in graph.nodes if n.group == "measure"]
    if measures:
        # Group measures by which construct they connect to
        construct_positions = {n.id: positions[n.id] for n in graph.nodes
                               if n.group == "construct" and n.id in positions}

        # Build measure → construct mapping from edges
        measure_to_construct = {}
        for edge in graph.edges:
            if edge.label == "measuresConstruct":
                if edge.source in {m.id for m in measures}:
                    measure_to_construct[edge.source] = edge.target
                elif edge.target in {m.id for m in measures}:
                    measure_to_construct[edge.target] = edge.source

        # Place measures around their construct
        by_construct: Dict[str, List[GraphNode]] = {}
        unlinked = []
        for m in measures:
            c = measure_to_construct.get(m.id)
            if c and c in construct_positions:
                by_construct.setdefault(c, []).append(m)
            else:
                unlinked.append(m)

        outer_r = 1.3
        for c_id, c_measures in by_construct.items():
            cx, cy = construct_positions[c_id]
            base_angle = math.atan2(cy, cx)
            spread = min(math.pi * 0.7, math.pi * len(c_measures) / 8)
            for i, m in enumerate(c_measures):
                a = base_angle - spread / 2 + spread * i / max(len(c_measures) - 1, 1)
                positions[m.id] = (cx + outer_r * math.cos(a), cy + outer_r * math.sin(a))

        # Place unlinked measures in a separate arc
        if unlinked:
            base = -math.pi / 2
            spread = math.pi * 0.5
            for i, m in enumerate(unlinked):
                a = base - spread / 2 + spread * i / max(len(unlinked) - 1, 1)
                positions[m.id] = (3.0 * math.cos(a), 3.0 * math.sin(a))

    return positions


# ---------------------------------------------------------------------------
# Plotly rendering
# ---------------------------------------------------------------------------

def render_measure_neighborhood(
    onto,
    sig_uri: str,
    sig_label: str = "",
    construct: Optional[str] = None,
    modality_category: Optional[str] = None,
    technique: Optional[str] = None,
    max_measures: int = 10,
    height: int = 400,
) -> Optional['go.Figure']:
    """
    Query the ontology and render a Plotly node-link diagram.

    Tries URI-based query first, falls back to metadata-based search.
    Returns a go.Figure or None if Plotly unavailable or no results.
    """
    if not PLOTLY_AVAILABLE:
        return None

    graph = query_signature_neighborhood(
        onto, sig_uri,
        sig_label=sig_label,
        construct=construct,
        modality_category=modality_category,
        technique=technique,
        max_measures=max_measures,
    )

    # Need more than just focal + evidence stub to be worth showing
    if len(graph.nodes) <= 2:
        return None

    positions = _layout_neighborhood(graph)

    fig = go.Figure()

    # Draw edges
    for edge in graph.edges:
        if edge.source in positions and edge.target in positions:
            x0, y0 = positions[edge.source]
            x1, y1 = positions[edge.target]
            fig.add_trace(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode='lines',
                line=dict(width=1, color='rgba(150,150,150,0.4)'),
                hoverinfo='skip', showlegend=False,
            ))

    # Draw nodes by group
    groups_present = {}
    for node in graph.nodes:
        groups_present.setdefault(node.group, []).append(node)

    for group, nodes in groups_present.items():
        style = GROUP_STYLES.get(group, GROUP_STYLES["measure"])
        xs, ys, texts, hovers = [], [], [], []
        for node in nodes:
            if node.id in positions:
                x, y = positions[node.id]
                xs.append(x)
                ys.append(y)
                texts.append(_clean_label(node.label, 25))
                hover_parts = [f"<b>{node.label}</b>",
                               f"Type: {GROUP_LABELS.get(group, group)}"]
                if node.description:
                    hover_parts.append(node.description[:150])
                hovers.append("<br>".join(hover_parts) + "<extra></extra>")

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode='markers+text',
            marker=dict(
                size=style["size"], color=style["color"],
                symbol=style["symbol"],
                line=dict(width=1, color='white'),
            ),
            text=texts, textposition="top center",
            textfont=dict(size=9, color='#374151'),
            hovertemplate=hovers,
            name=GROUP_LABELS.get(group, group),
            legendgroup=group, showlegend=True,
        ))

    fig.update_layout(
        height=height,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False,
                   scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0,
                    xanchor="center", x=0.5, font=dict(size=10)),
        hovermode='closest',
    )

    return fig
