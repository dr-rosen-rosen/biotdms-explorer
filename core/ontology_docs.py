"""
ontology_docs.py
================

Generate documentation artifacts (figures + tables) directly from the BioTDMS
ontology TTL files. Used both for manuscript figures and (later) for an
overview view in the BioTDMS Explorer.

Outputs:
    - Top-level schema diagram (SVG/PNG) showing core classes and the object
      properties that connect them.
    - Hierarchy tree diagrams (one per major branch: Construct, Modality,
      analyticTechnique, levelOfAnalysis).
    - Summary statistics table (Markdown).
    - Class glossary table (Markdown).

Architecture:
    1. `OntologyModel` — intermediate representation extracted from the graph.
       This is the "shared extraction" layer that a future interactive renderer
       (pyvis/Plotly for Streamlit) can also consume.
    2. Renderers — functions that take an OntologyModel and produce static
       Graphviz / Markdown outputs.

Usage (programmatic):
    from core.ontology_docs import build_model, render_all
    model = build_model([Path("teamMeasurement.ttl"), Path("instances.ttl")])
    render_all(model, out_dir=Path("docs/ontology_overview"))

Usage (CLI):
    python -m core.ontology_docs \\
        --ttl data/ontologies/teamMeasurement.ttl \\
        --ttl data/ontologies/instances.ttl \\
        --out docs/ontology_overview

Requires:
    rdflib, graphviz (Python bindings + the `dot` binary on PATH)
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Namespaces
# ---------------------------------------------------------------------------

NAMESPACES: Dict[str, str] = {
    "meas": "http://example.org/ontology/teamMeasurement#",
    "evid": "http://example.org/ontology/evidence#",
    "inst": "http://example.org/ontology/instances#",
}

# Core schema classes that anchor the top-level diagram. These are the classes
# that appear as domain/range for the major object properties and that readers
# need to see in the "one figure to understand the whole thing" view.
CORE_CLASSES_MEAS: Tuple[str, ...] = (
    "Measure",
    "Manipulation",
    "Construct",
    "Modality",
    "Method",
    "analyticTechnique",
    "levelOfAnalysis",
)
CORE_CLASSES_EVID: Tuple[str, ...] = (
    "Publication",
    "EffectSize",
    "primaryStudy",
    "metaAnalysis",
)

# Branches we want to render as separate hierarchy trees. Keyed by display
# title; value is the local name of the root class in the meas: namespace.
HIERARCHY_BRANCHES_MEAS: Dict[str, str] = {
    "Construct hierarchy": "Construct",
    "Modality hierarchy": "Modality",
    "Analytic technique hierarchy": "analyticTechnique",
    "Level of analysis hierarchy": "levelOfAnalysis",
    "Method hierarchy": "Method",
}

# ---------------------------------------------------------------------------
# Intermediate representation
# ---------------------------------------------------------------------------


@dataclass
class ClassInfo:
    """Information about a single OWL/RDFS class."""

    uri: str
    local_name: str
    namespace_prefix: str  # 'meas', 'evid', 'inst', or '' if unknown
    label: Optional[str] = None
    comment: Optional[str] = None
    parents: List[str] = field(default_factory=list)  # parent URIs
    children: List[str] = field(default_factory=list)  # child URIs
    instance_count: int = 0

    @property
    def display_label(self) -> str:
        """Label used in figures. Falls back to local name if no rdfs:label."""
        return self.label if self.label else self.local_name

    @property
    def qname(self) -> str:
        """Compact name like 'meas:Construct' for display."""
        return f"{self.namespace_prefix}:{self.local_name}" if self.namespace_prefix else self.local_name


@dataclass
class PropertyInfo:
    """Information about an object or datatype property."""

    uri: str
    local_name: str
    namespace_prefix: str
    label: Optional[str] = None
    comment: Optional[str] = None
    is_object_property: bool = True  # False => datatype property
    domains: List[str] = field(default_factory=list)  # class URIs (flattened if union)
    ranges: List[str] = field(default_factory=list)  # class URIs or literal type URIs
    sub_property_of: List[str] = field(default_factory=list)

    @property
    def display_label(self) -> str:
        return self.label if self.label else self.local_name

    @property
    def qname(self) -> str:
        return f"{self.namespace_prefix}:{self.local_name}" if self.namespace_prefix else self.local_name


@dataclass
class OntologyModel:
    """Intermediate representation built from one or more TTL files.

    This is the shared structure that both static (Graphviz) and interactive
    (Streamlit) renderers consume.
    """

    classes: Dict[str, ClassInfo] = field(default_factory=dict)  # by URI
    object_properties: Dict[str, PropertyInfo] = field(default_factory=dict)
    datatype_properties: Dict[str, PropertyInfo] = field(default_factory=dict)
    source_files: List[str] = field(default_factory=list)
    triple_count: int = 0

    # --- Lookup helpers --------------------------------------------------

    def class_by_local(self, local_name: str, prefix: str = "meas") -> Optional[ClassInfo]:
        """Find a class by its local name within a namespace."""
        target_uri = NAMESPACES[prefix] + local_name
        return self.classes.get(target_uri)

    def classes_in_namespace(self, prefix: str) -> List[ClassInfo]:
        """All classes whose URI is in the given namespace, sorted by local name."""
        ns = NAMESPACES[prefix]
        return sorted(
            (c for c in self.classes.values() if c.uri.startswith(ns)),
            key=lambda c: c.local_name.lower(),
        )

    def descendants(self, root_uri: str) -> Set[str]:
        """All transitive descendants of a class (not including the root itself)."""
        out: Set[str] = set()
        stack = list(self.classes.get(root_uri, ClassInfo("", "", "")).children)
        while stack:
            current = stack.pop()
            if current in out:
                continue
            out.add(current)
            child = self.classes.get(current)
            if child:
                stack.extend(child.children)
        return out

    def subtree(self, root_uri: str) -> Set[str]:
        """Root plus all descendants."""
        return {root_uri} | self.descendants(root_uri)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def _split_uri(uri: str) -> Tuple[str, str]:
    """Return (namespace_prefix, local_name) for a known namespace, else ('', uri)."""
    for prefix, ns_uri in NAMESPACES.items():
        if uri.startswith(ns_uri):
            return prefix, uri[len(ns_uri):]
    # Standard vocabularies
    if uri.startswith(str(OWL)):
        return "owl", uri[len(str(OWL)):]
    if uri.startswith(str(RDFS)):
        return "rdfs", uri[len(str(RDFS)):]
    if uri.startswith(str(RDF)):
        return "rdf", uri[len(str(RDF)):]
    # XSD or unknown
    if "#" in uri:
        return "", uri.rsplit("#", 1)[-1]
    return "", uri.rsplit("/", 1)[-1]


def _resolve_union_domain_range(graph: Graph, node) -> List[str]:
    """Domain/range may be a single URI or an owl:unionOf list. Return flat URIs."""
    if isinstance(node, URIRef):
        return [str(node)]
    # Blank node: look for owl:unionOf
    union_list = list(graph.objects(node, OWL.unionOf))
    if not union_list:
        return []
    head = union_list[0]
    # Walk RDF list
    items: List[str] = []
    while head and head != RDF.nil:
        first = next(graph.objects(head, RDF.first), None)
        if isinstance(first, URIRef):
            items.append(str(first))
        head = next(graph.objects(head, RDF.rest), None)
    return items


def build_model(ttl_paths: Iterable[Path]) -> OntologyModel:
    """Parse TTL files and build the OntologyModel."""
    g = Graph()
    paths = list(ttl_paths)
    for p in paths:
        logger.info("Parsing %s", p)
        g.parse(p, format="turtle")

    model = OntologyModel(
        source_files=[str(p) for p in paths],
        triple_count=len(g),
    )

    # --- Classes -----------------------------------------------------------
    class_uris: Set[URIRef] = set()
    for s, _p, _o in g.triples((None, RDF.type, OWL.Class)):
        if isinstance(s, URIRef):
            class_uris.add(s)
    # Also include classes that appear as parents but aren't explicitly typed
    for _s, _p, o in g.triples((None, RDFS.subClassOf, None)):
        if isinstance(o, URIRef):
            class_uris.add(o)

    for cls_uri in class_uris:
        uri_str = str(cls_uri)
        prefix, local = _split_uri(uri_str)
        # Skip classes outside our project namespaces
        if prefix not in NAMESPACES:
            continue
        label_node = next(g.objects(cls_uri, RDFS.label), None)
        comment_node = next(g.objects(cls_uri, RDFS.comment), None)
        parents: List[str] = []
        for parent in g.objects(cls_uri, RDFS.subClassOf):
            if isinstance(parent, URIRef):
                parents.append(str(parent))
        model.classes[uri_str] = ClassInfo(
            uri=uri_str,
            local_name=local,
            namespace_prefix=prefix,
            label=str(label_node) if label_node else None,
            comment=str(comment_node) if comment_node else None,
            parents=parents,
        )

    # Populate children (reverse edges)
    for cls in model.classes.values():
        for parent_uri in cls.parents:
            parent = model.classes.get(parent_uri)
            if parent:
                parent.children.append(cls.uri)

    # Sort children for stable rendering
    for cls in model.classes.values():
        cls.children.sort(key=lambda u: model.classes[u].local_name.lower() if u in model.classes else u)

    # --- Properties --------------------------------------------------------
    for prop_type, container, is_obj in [
        (OWL.ObjectProperty, model.object_properties, True),
        (OWL.DatatypeProperty, model.datatype_properties, False),
    ]:
        for s, _p, _o in g.triples((None, RDF.type, prop_type)):
            if not isinstance(s, URIRef):
                continue
            uri_str = str(s)
            prefix, local = _split_uri(uri_str)
            if prefix not in NAMESPACES:
                continue
            label_node = next(g.objects(s, RDFS.label), None)
            comment_node = next(g.objects(s, RDFS.comment), None)

            # Domain/range — may be union classes
            domains: List[str] = []
            for d in g.objects(s, RDFS.domain):
                domains.extend(_resolve_union_domain_range(g, d))
            ranges: List[str] = []
            for r in g.objects(s, RDFS.range):
                ranges.extend(_resolve_union_domain_range(g, r))

            sub_of: List[str] = []
            for sp in g.objects(s, RDFS.subPropertyOf):
                if isinstance(sp, URIRef):
                    sub_of.append(str(sp))

            container[uri_str] = PropertyInfo(
                uri=uri_str,
                local_name=local,
                namespace_prefix=prefix,
                label=str(label_node) if label_node else None,
                comment=str(comment_node) if comment_node else None,
                is_object_property=is_obj,
                domains=domains,
                ranges=ranges,
                sub_property_of=sub_of,
            )

    # --- Instance counts ---------------------------------------------------
    # For each individual (rdf:type X where X is in our class set), bump count.
    for s, _p, o in g.triples((None, RDF.type, None)):
        if not isinstance(s, URIRef) or not isinstance(o, URIRef):
            continue
        # Skip the schema-level type declarations themselves
        if o in (OWL.Class, OWL.ObjectProperty, OWL.DatatypeProperty, OWL.Ontology):
            continue
        target = model.classes.get(str(o))
        if target:
            target.instance_count += 1

    logger.info(
        "Built model: %d classes, %d object props, %d datatype props, %d triples",
        len(model.classes),
        len(model.object_properties),
        len(model.datatype_properties),
        model.triple_count,
    )
    return model


# ---------------------------------------------------------------------------
# Renderers — Graphviz figures
# ---------------------------------------------------------------------------

# Visual style constants
NS_COLORS: Dict[str, str] = {
    "meas": "#4ECDC4",  # teal — measurement layer
    "evid": "#45B7D1",  # blue — evidence layer
    "inst": "#DDA0DD",  # plum — instances
}
NS_FILL: Dict[str, str] = {
    "meas": "#E8F8F7",
    "evid": "#E3F2F8",
    "inst": "#F5EAF5",
}
SUBCLASS_EDGE_STYLE = {"arrowhead": "empty", "color": "#666666"}
OBJECT_PROP_EDGE_STYLE = {"arrowhead": "vee", "color": "#333333", "fontsize": "10"}


def _node_label_html(info: ClassInfo) -> str:
    """HTML-like label: bold display name on top, smaller qname underneath."""
    display = info.display_label
    return (
        f"<<TABLE BORDER='0' CELLBORDER='0' CELLSPACING='0'>"
        f"<TR><TD><B>{_xml_escape(display)}</B></TD></TR>"
        f"<TR><TD><FONT POINT-SIZE='9' COLOR='#555555'>{_xml_escape(info.qname)}</FONT></TD></TR>"
        f"</TABLE>>"
    )


def _xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _safe_id(uri: str) -> str:
    """Make a Graphviz-safe node id from a URI.

    Graphviz interprets ':' as a port separator, so raw URIs like
    'http://example.org/...#Foo' break edges. We replace non-alphanumeric chars
    with underscores; the mapping is deterministic so edges stay consistent.
    """
    out: List[str] = []
    for ch in uri:
        if ch.isalnum() or ch == "_":
            out.append(ch)
        else:
            out.append("_")
    safe = "".join(out)
    # Graphviz requires IDs not to start with a digit
    if safe and safe[0].isdigit():
        safe = "n_" + safe
    return safe


def _try_import_graphviz():
    try:
        import graphviz  # type: ignore
        return graphviz
    except ImportError as exc:
        raise RuntimeError(
            "graphviz Python package is required. Install with: pip install graphviz\n"
            "Also ensure the `dot` binary is on PATH (e.g., apt-get install graphviz)."
        ) from exc


def render_top_level_schema(
    model: OntologyModel,
    out_path: Path,
    formats: Tuple[str, ...] = ("svg", "png"),
) -> List[Path]:
    """Top-level schema: core classes + object properties as labeled edges.

    Shows the headline architecture: Measure --measuresConstruct--> Construct,
    Measure --includesModality--> Modality, etc.
    """
    graphviz = _try_import_graphviz()
    dot = graphviz.Digraph(
        "top_level_schema",
        graph_attr={"rankdir": "LR", "splines": "true", "nodesep": "0.6", "ranksep": "1.0"},
        node_attr={"shape": "box", "style": "rounded,filled", "fontname": "Helvetica"},
        edge_attr={"fontname": "Helvetica"},
    )

    # Collect URIs of core classes that actually exist in the model
    core_uris: List[str] = []
    for local in CORE_CLASSES_MEAS:
        info = model.class_by_local(local, "meas")
        if info:
            core_uris.append(info.uri)
    for local in CORE_CLASSES_EVID:
        info = model.class_by_local(local, "evid")
        if info:
            core_uris.append(info.uri)

    # Nodes
    for uri in core_uris:
        info = model.classes[uri]
        dot.node(
            _safe_id(uri),
            label=_node_label_html(info),
            color=NS_COLORS.get(info.namespace_prefix, "#888888"),
            fillcolor=NS_FILL.get(info.namespace_prefix, "#EEEEEE"),
        )

    core_set = set(core_uris)

    # Object property edges — only when both domain and range are in our core set
    for prop in model.object_properties.values():
        for dom in prop.domains:
            for rng in prop.ranges:
                if dom in core_set and rng in core_set:
                    dot.edge(
                        _safe_id(dom),
                        _safe_id(rng),
                        label=prop.display_label,
                        **OBJECT_PROP_EDGE_STYLE,
                    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    produced: List[Path] = []
    for fmt in formats:
        rendered = dot.render(
            filename=out_path.stem,
            directory=str(out_path.parent),
            format=fmt,
            cleanup=True,
        )
        produced.append(Path(rendered))
        logger.info("Wrote %s", rendered)
    return produced


def render_hierarchy_tree(
    model: OntologyModel,
    root_local: str,
    title: str,
    out_path: Path,
    namespace: str = "meas",
    formats: Tuple[str, ...] = ("svg", "png"),
    max_depth: Optional[int] = None,
) -> List[Path]:
    """Render a subClassOf tree rooted at the given class."""
    graphviz = _try_import_graphviz()
    root = model.class_by_local(root_local, namespace)
    if root is None:
        logger.warning("Root class %s:%s not found; skipping %s", namespace, root_local, title)
        return []

    dot = graphviz.Digraph(
        f"hierarchy_{root_local}",
        graph_attr={
            "rankdir": "TB",
            "splines": "ortho",
            "nodesep": "0.3",
            "ranksep": "0.5",
            "label": title,
            "labelloc": "t",
            "fontname": "Helvetica",
            "fontsize": "14",
        },
        node_attr={"shape": "box", "style": "rounded,filled", "fontname": "Helvetica"},
        edge_attr={"fontname": "Helvetica"},
    )

    # BFS over descendants, respecting max_depth
    queue: List[Tuple[str, int]] = [(root.uri, 0)]
    seen: Set[str] = set()
    while queue:
        uri, depth = queue.pop(0)
        if uri in seen:
            continue
        seen.add(uri)
        info = model.classes.get(uri)
        if info is None:
            continue
        dot.node(
            _safe_id(uri),
            label=_node_label_html(info),
            color=NS_COLORS.get(info.namespace_prefix, "#888888"),
            fillcolor=NS_FILL.get(info.namespace_prefix, "#EEEEEE"),
        )
        if max_depth is None or depth < max_depth:
            for child_uri in info.children:
                queue.append((child_uri, depth + 1))
                dot.edge(_safe_id(child_uri), _safe_id(uri), **SUBCLASS_EDGE_STYLE)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    produced: List[Path] = []
    for fmt in formats:
        rendered = dot.render(
            filename=out_path.stem,
            directory=str(out_path.parent),
            format=fmt,
            cleanup=True,
        )
        produced.append(Path(rendered))
        logger.info("Wrote %s", rendered)
    return produced


# ---------------------------------------------------------------------------
# Renderers — Markdown tables
# ---------------------------------------------------------------------------


def render_summary_stats(model: OntologyModel, out_path: Path) -> Path:
    """Summary statistics table: counts per branch + property/instance totals."""
    lines: List[str] = []
    lines.append("# Ontology Summary Statistics\n")
    lines.append(f"_Source files:_ " + ", ".join(f"`{Path(p).name}`" for p in model.source_files))
    lines.append(f"\n_Total triples:_ {model.triple_count}\n")

    # Top-level: classes per namespace
    lines.append("## Classes by layer\n")
    lines.append("| Namespace | Classes | Total instances |")
    lines.append("|---|---:|---:|")
    for prefix in ("meas", "evid"):
        classes = model.classes_in_namespace(prefix)
        total_instances = sum(c.instance_count for c in classes)
        lines.append(f"| `{prefix}:` | {len(classes)} | {total_instances} |")

    # Per major branch under meas:
    lines.append("\n## Classes per major branch (measurement layer)\n")
    lines.append("| Branch | Direct subclasses | All descendants | Instances (subtree) |")
    lines.append("|---|---:|---:|---:|")
    for title, root_local in HIERARCHY_BRANCHES_MEAS.items():
        root = model.class_by_local(root_local, "meas")
        if root is None:
            lines.append(f"| {title} | — | — | — |")
            continue
        descendants = model.descendants(root.uri)
        subtree = {root.uri} | descendants
        instances = sum(model.classes[u].instance_count for u in subtree if u in model.classes)
        lines.append(
            f"| {title} (`meas:{root_local}`) | {len(root.children)} | {len(descendants)} | {instances} |"
        )

    # Properties
    lines.append("\n## Properties\n")
    lines.append("| Type | Count |")
    lines.append("|---|---:|")
    lines.append(f"| Object properties | {len(model.object_properties)} |")
    lines.append(f"| Datatype properties | {len(model.datatype_properties)} |")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", out_path)
    return out_path


def render_class_glossary(model: OntologyModel, out_path: Path) -> Path:
    """Full class glossary: one row per class with parent, definition, instance count."""
    lines: List[str] = []
    lines.append("# Class Glossary\n")

    for prefix, layer_title in (("meas", "Measurement layer (`meas:`)"), ("evid", "Evidence layer (`evid:`)")):
        classes = model.classes_in_namespace(prefix)
        if not classes:
            continue
        lines.append(f"\n## {layer_title}\n")
        lines.append("| Class | Label | Parent(s) | Definition | Instances |")
        lines.append("|---|---|---|---|---:|")
        for cls in classes:
            parents = ", ".join(
                model.classes[p].qname if p in model.classes else _split_uri(p)[1]
                for p in cls.parents
            ) or "—"
            comment = (cls.comment or "").replace("\n", " ").replace("|", "\\|")
            label = cls.label or ""
            lines.append(
                f"| `{cls.qname}` | {label} | {parents} | {comment} | {cls.instance_count} |"
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", out_path)
    return out_path


def render_property_glossary(model: OntologyModel, out_path: Path) -> Path:
    """Properties table: object + datatype properties with domain/range."""
    lines: List[str] = []
    lines.append("# Property Glossary\n")

    def fmt_uri_list(uris: List[str]) -> str:
        out: List[str] = []
        for u in uris:
            if u in model.classes:
                out.append(f"`{model.classes[u].qname}`")
            else:
                prefix, local = _split_uri(u)
                out.append(f"`{prefix}:{local}`" if prefix else f"`{local}`")
        return ", ".join(out) or "—"

    for container, title in (
        (model.object_properties, "Object properties"),
        (model.datatype_properties, "Datatype properties"),
    ):
        if not container:
            continue
        lines.append(f"\n## {title}\n")
        lines.append("| Property | Label | Domain | Range | Definition |")
        lines.append("|---|---|---|---|---|")
        for prop in sorted(container.values(), key=lambda p: (p.namespace_prefix, p.local_name.lower())):
            comment = (prop.comment or "").replace("\n", " ").replace("|", "\\|")
            label = prop.label or ""
            lines.append(
                f"| `{prop.qname}` | {label} | {fmt_uri_list(prop.domains)} | "
                f"{fmt_uri_list(prop.ranges)} | {comment} |"
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def render_all(
    model: OntologyModel,
    out_dir: Path,
    formats: Tuple[str, ...] = ("svg", "png"),
) -> Dict[str, List[Path]]:
    """Render every artifact and return a manifest of produced files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, List[Path]] = {}

    # Top-level schema
    manifest["top_level_schema"] = render_top_level_schema(
        model, out_dir / "schema_top_level", formats=formats
    )

    # Hierarchy trees
    for title, root_local in HIERARCHY_BRANCHES_MEAS.items():
        key = f"hierarchy_{root_local}"
        manifest[key] = render_hierarchy_tree(
            model,
            root_local=root_local,
            title=title,
            out_path=out_dir / key,
            namespace="meas",
            formats=formats,
        )

    # Tables
    manifest["summary_stats"] = [render_summary_stats(model, out_dir / "summary_stats.md")]
    manifest["class_glossary"] = [render_class_glossary(model, out_dir / "class_glossary.md")]
    manifest["property_glossary"] = [render_property_glossary(model, out_dir / "property_glossary.md")]

    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate ontology documentation (figures + tables) from TTL files."
    )
    parser.add_argument(
        "--ttl",
        action="append",
        required=True,
        help="Path to a TTL file. Repeat to include multiple files (e.g. schema + instances).",
    )
    parser.add_argument(
        "--out",
        default="docs/ontology_overview",
        help="Output directory (default: docs/ontology_overview)",
    )
    parser.add_argument(
        "--format",
        action="append",
        default=None,
        help="Graphviz output format (svg, png, pdf). Repeat for multiple. Default: svg + png.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    ttl_paths = [Path(p) for p in args.ttl]
    for p in ttl_paths:
        if not p.exists():
            parser.error(f"TTL file not found: {p}")

    formats = tuple(args.format) if args.format else ("svg", "png")
    out_dir = Path(args.out)

    model = build_model(ttl_paths)
    manifest = render_all(model, out_dir=out_dir, formats=formats)

    print(f"\nGenerated artifacts in {out_dir}:")
    for key, paths in manifest.items():
        for p in paths:
            print(f"  [{key}] {p}")
    print(
        f"\nModel summary: {len(model.classes)} classes, "
        f"{len(model.object_properties)} object props, "
        f"{len(model.datatype_properties)} datatype props, "
        f"{model.triple_count} triples."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
