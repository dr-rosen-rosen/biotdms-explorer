"""
Subtask → Construct Mapping

Loads the CTA-derived mapping from subtask numbers to expected construct demands.
Used by the UC2 analysis view to:
  1. Enrich subtask hover tooltips with construct demand info
  2. Render a construct heatmap strip below timeseries plots
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import yaml


@dataclass
class ConstructDemand:
    """A single construct's expected demand within a subtask."""
    construct: str
    weight: float
    color: str = "#6b7280"
    short_label: str = ""


@dataclass
class SubtaskProfile:
    """The full construct demand profile for one subtask."""
    number: int
    label: str
    description: str
    demands: List[ConstructDemand] = field(default_factory=list)

    def get_demand(self, construct: str) -> float:
        for d in self.demands:
            if d.construct.lower() == construct.lower():
                return d.weight
        return 0.0

    def top_demands(self, n: int = 3) -> List[ConstructDemand]:
        return sorted(self.demands, key=lambda d: d.weight, reverse=True)[:n]

    def hover_text(self) -> str:
        if not self.demands:
            return ""
        lines = ["<b>Expected demands:</b>"]
        for d in sorted(self.demands, key=lambda x: x.weight, reverse=True):
            bar_len = int(d.weight * 8)
            bar = "\u2588" * bar_len + "\u2591" * (8 - bar_len)
            label = d.short_label or d.construct
            lines.append(f"  {label}: {bar} ({d.weight:.1f})")
        return "<br>".join(lines)


class SubtaskConstructMap:
    """
    Loads and queries subtask-to-construct demand mappings.

    Usage:
        scm = SubtaskConstructMap(Path("config/subtask_constructs.yaml"))
        profile = scm.get_profile(subtask_number=2)
        hover = profile.hover_text()
        all_c = scm.all_constructs()
        focused = scm.constructs_for_signature("workload")
    """

    def __init__(self, yaml_path: Path):
        self._path = Path(yaml_path)
        self._profiles: Dict[int, SubtaskProfile] = {}
        self._construct_display: Dict[str, dict] = {}
        self._all_constructs: Optional[List[str]] = None
        self._load()

    def _load(self):
        if not self._path.exists():
            return
        with open(self._path, 'r') as f:
            data = yaml.safe_load(f)
        if not data:
            return

        self._construct_display = data.get('construct_display', {})

        for num_key, info in data.get('subtasks', {}).items():
            num = int(num_key)
            demands = []
            for construct_key, weight in info.get('constructs', {}).items():
                display = self._construct_display.get(construct_key, {})
                demands.append(ConstructDemand(
                    construct=construct_key,
                    weight=float(weight),
                    color=display.get('color', '#6b7280'),
                    short_label=display.get('short', construct_key),
                ))
            self._profiles[num] = SubtaskProfile(
                number=num,
                label=info.get('label', f"Subtask {num}"),
                description=info.get('description', ''),
                demands=demands,
            )

    @property
    def available(self) -> bool:
        return len(self._profiles) > 0

    def get_profile(self, subtask_number: int) -> Optional[SubtaskProfile]:
        return self._profiles.get(subtask_number)

    def all_constructs(self) -> List[str]:
        if self._all_constructs is None:
            constructs = set()
            for profile in self._profiles.values():
                for d in profile.demands:
                    constructs.add(d.construct)
            self._all_constructs = sorted(constructs)
        return self._all_constructs

    def get_construct_color(self, construct: str) -> str:
        return self._construct_display.get(construct, {}).get('color', '#6b7280')

    def get_construct_short(self, construct: str) -> str:
        return self._construct_display.get(construct, {}).get('short', construct)

    def constructs_for_signature(self, signature_construct: str) -> List[str]:
        """For focused view: constructs relevant to a signature's construct."""
        if not signature_construct:
            return self.all_constructs()
        key = signature_construct.lower().replace(" ", "_")
        matches = [c for c in self.all_constructs() if c.lower() == key]
        return matches if matches else self.all_constructs()

    def build_heatmap_data(
        self,
        subtask_events,
        constructs_to_show: Optional[List[str]] = None,
    ) -> Tuple[List[float], List[float], List[str], List[List[float]]]:
        """
        Build arrays for a Plotly heatmap aligned to subtask windows.

        Returns (x_starts, x_ends, y_labels, z_matrix)
        where z_matrix[construct_idx][event_idx] = demand weight.
        """
        if not subtask_events or not self.available:
            return [], [], [], []

        constructs = constructs_to_show or self.all_constructs()
        if not constructs:
            return [], [], [], []

        x_starts, x_ends = [], []
        z_matrix = [[] for _ in constructs]

        for evt in subtask_events:
            subtask_num = getattr(evt, 'subtask_number', None)
            if subtask_num is None:
                subtask_num = getattr(evt, 'category', None)
            if subtask_num is None:
                continue

            profile = self.get_profile(int(subtask_num))
            x_starts.append(evt.start_sec)
            x_ends.append(evt.end_sec)

            for i, construct in enumerate(constructs):
                weight = profile.get_demand(construct) if profile else 0.0
                z_matrix[i].append(weight)

        y_labels = [self.get_construct_short(c) for c in constructs]
        return y_labels, x_starts, x_ends, z_matrix


def load_subtask_construct_map(config_dir: Path) -> Optional[SubtaskConstructMap]:
    """Convenience loader. Returns None if config missing (graceful degradation)."""
    yaml_path = config_dir / "subtask_constructs.yaml"
    if yaml_path.exists():
        scm = SubtaskConstructMap(yaml_path)
        return scm if scm.available else None
    return None
