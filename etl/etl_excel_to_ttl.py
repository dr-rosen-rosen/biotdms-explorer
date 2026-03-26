#!/usr/bin/env python3
"""
etl_excel_to_ttl.py — Full Excel-to-TTL ETL for BioTDMS ontology

Processes all sheets: publications, studies, measures, effects, manipulations
Handles:
  - Ontology-driven URI matching (reads existing TTL instances)
  - Construct normalization via curated mapping
  - New construct/modality/technique/level auto-creation
  - Evidence instances (publications, studies, effects) with full property coverage
  - Effect category split: evid:hasSignificanceCategory + evid:hasEffectDomain
  - Manipulation instances linked to effects as independent variables
  - Data quality reporting (orphan refs, unmapped values, exclusions)

USAGE:
  python etl_excel_to_ttl.py --excel coding.xlsx --team-ttl teamMeasurement.ttl --evidence-ttl evidence.ttl [--old-instances instances.ttl] --out-ttl output.ttl [--merge]

  --merge: also writes merged_instances.ttl combining old-instances + new output
"""

import argparse
import re
import sys
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import pandas as pd
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL, SKOS, XSD

# ── Namespaces ──────────────────────────────────────────────────────
MEAS = Namespace("http://example.org/ontology/teamMeasurement#")
EVID = Namespace("http://example.org/ontology/evidence#")
INST = Namespace("http://example.org/ontology/instances#")


# ── Construct normalization map (curated 2026-03-25) ────────────────
# Maps raw Excel measuresConstruct values -> inst: local name
# Decisions tracked in DECISIONS_LOG.md
CONSTRUCT_MAP = {
    "Adaptation": "construct_adaptation",
    "Adaptation?": "construct_adaptation",
    "Phase transitions in collaborative problem-solving; communication dynamics; team cognition dynamics": "construct_adaptation",
    "Communication: quality": "construct_communication",
    "Coordination: cognitive": "construct_coordination",
    "Coordination: physical": "construct_coordination",
    "Coordination: Complementarity": "construct_coordination",
    "Coordination: Imitation": "construct_coordination",
    "Coordination: cognitive and physical": "construct_coordination",
    "Coordination?": "construct_coordination",
    "Team coordination (behavioral)": "construct_coordination",
    "Coordination: cognitive & Resilience": "construct_coordination",
    "Cooperation": "construct_coordination",
    "Resilience": "construct_resilience",
    "Shared mental model": "construct_shared_mental_model",
    "Workload": "construct_workload",
    "Team workload": "construct_workload",
    "Composition: roles": "construct_team_composition",
    "Composition: experience": "construct_team_composition",
    "Performance: outcome": "construct_task_outcome",
    "Performance Outcome": "construct_task_outcome",
    "Performance: Outcome": "construct_task_outcome",
    "Performance: outcome & efficiency": "construct_task_outcome",
    "Team performance effectiveness (accuracy)": "construct_task_outcome",
    "Team performance effectiveness (safety)": "construct_task_outcome",
    "Team Performance": "construct_task_outcome",
    "Performance: efficiency": "construct_task_outcome",
    "Team performance effectiveness (efficiency)": "construct_task_outcome",
    "Positive/negative emotional valence": "construct_team_affect",
    "Empathy": "construct_team_affect",
    "Anxiety": "construct_team_affect",
    "Emotional Intelligence": "construct_team_affect",
    "Teamwork quality": "construct_teamwork",
    "Team Orientation / cohesion": "construct_team_cohesion",
    "Team orientation / Cohesion": "construct_team_cohesion",
    "Team orientation/cohesion": "construct_team_cohesion",
    "Team Orientation / cohesion/Social Impairment": "construct_team_cohesion",
    "Leadership": "construct_leadership",
    "Physiological compliance / synchrony": "construct_physiological_synchrony",
    "Physiological Compliance": "construct_physiological_synchrony",
    "Social Psychophysiological Compliance": "construct_physiological_synchrony",
    "Social  Psychophysiological Compliance": "construct_physiological_synchrony",
    "Physiological synchrony": "construct_physiological_synchrony",
    "Physiological synchrony (timing)": "construct_physiological_synchrony",
    "Physiological Stability": "construct_physiological_synchrony",
    "Team synchrony": "construct_physiological_synchrony",
    "Individual Behavior: Consistency": "construct_individual_behavior",
    "Team engagement": "construct_team_engagement",
    "Decision making exploration": "construct_decision_making",
    "Team prganization": "construct_team_organization",
    
    # Manipulation constructs (from manipulations sheet)
    "Communication quality": "construct_communication",
    "communication quality": "construct_communication",
    "team member familiarity": "construct_team_composition",
    "task type or context": "construct_task_outcome",
    "context descriptor": "construct_task_outcome",
    "shared mental model": "construct_shared_mental_model",
    "time index": "construct_task_outcome",
    "Social feedback / perceptual access to teammate behavior": "construct_communication",
}

# Labels for NEW constructs (not already in the TTL)
NEW_CONSTRUCT_LABELS = {
    "construct_team_cohesion": "Team cohesion",
    "construct_leadership": "Leadership",
    "construct_physiological_synchrony": "Physiological synchrony",
    "construct_individual_behavior": "Individual behavior",
    "construct_team_engagement": "Team engagement",
    "construct_decision_making": "Decision making",
    "construct_team_organization": "Team organization",
}

# ── Effect category classification ──────────────────────────────────
SIGNIFICANCE_CATEGORIES = {"NS", "Significantly positive", "Significantly negative"}

# If the Category value is in SIGNIFICANCE_CATEGORIES, it goes to
# evid:hasSignificanceCategory. Otherwise it goes to evid:hasEffectDomain.


# ── Utilities ───────────────────────────────────────────────────────

def normalize_whitespace(text):
    """Replace non-breaking spaces, collapse whitespace."""
    if pd.isna(text):
        return None
    s = str(text).replace('\xa0', ' ').replace('\u200b', '')
    s = re.sub(r'\s+', ' ', s).strip()
    return s if s else None


def slugify(label):
    """Create URL-safe local name fragment."""
    s = re.sub(r'[^a-zA-Z0-9]+', '_', str(label)).strip('_')
    return s.lower()


def norm_label(label):
    """Normalize label for display/matching."""
    s = normalize_whitespace(label)
    if not s:
        return ""
    s = s.replace('\u2013', '-').replace('\u2014', '-').replace('\u2010', '-')
    s = re.sub(r'^(dsa)\s*-\s*', 'DSA - ', s, flags=re.I)
    return s


def safe_float(val):
    """Try to parse a float, return None on failure."""
    if pd.isna(val):
        return None
    try:
        s = str(val).strip()
        # Handle scientific notation from Excel like "3.1E-2"
        return float(s)
    except (ValueError, TypeError):
        return None


def safe_int(val):
    """Try to parse an int, return None on failure."""
    f = safe_float(val)
    if f is None:
        return None
    return int(f)


def infer_coarse_modality(label_lower):
    """Infer the skos:broader parent for a modality label."""
    ll = label_lower
    if any(k in ll for k in ('physiol', 'ibi', 'cardiac', 'eda', 'eeg', 'heart rate',
                              'electrodermal', 'breathing', 'respiration', 'postural')):
        return 'physiology'
    if any(k in ll for k in ('survey', 'interview', 'questionnaire', 'self report',
                              'team rating')):
        return 'survey'
    if any(k in ll for k in ('observ', 'ethnograph')):
        return 'observation'
    if any(k in ll for k in ('communicat', 'language', 'speech', 'text', 'verbal',
                              'paralinguistic')):
        return 'communication'
    if any(k in ll for k in ('behavior', 'movement', 'system', 'log', 'motor',
                              'interaction', 'activity')):
        return 'behavior'
    if any(k in ll for k in ('task outcome', 'accuracy', 'time on task', 'simulation',
                              'class grade', 'performance', 'count')):
        return 'behavior'  # task outcomes group under behavior per existing TTL
    return None


def normalize_measure_id(raw_id):
    """Normalize measure ID variants to a consistent format.
    
    Handles: meas_001, Meas_064, measure_5001 -> consistent inst: local name
    The TTL uses inst:measure_meas_NNN pattern.
    """
    s = str(raw_id).strip()
    # Strip trailing junk like " + ", " +", etc.
    s = re.sub(r'\s*\+.*$', '', s).strip()
    # Remove any remaining URI-unsafe characters
    s = re.sub(r'[^a-zA-Z0-9_]', '_', s).strip('_')
    s_lower = s.lower()
    
    if s_lower.startswith('measure_'):
        return f"measure_{s_lower.split('measure_')[1]}"
    elif s_lower.startswith('meas_'):
        num_part = s_lower.split('meas_')[1]
        return f"measure_meas_{num_part}"
    else:
        return f"measure_{slugify(s)}"


def normalize_effect_metric(raw_metric):
    """Normalize effect size metric names to reduce variation."""
    if pd.isna(raw_metric):
        return None
    s = str(raw_metric).strip()
    sl = s.lower()
    
    # Collapse Pearson r variants
    if sl in ('pearson r', "pearson's r", "pearson's r product-moment correlation",
              'pearson correlation', 'r', 'correlation coefficient rs.',
              'correlation'):
        return 'Pearson r'
    if sl in ('r (partial correlation)',):
        return 'partial r'
    if 'beta' in sl or sl == '\u03b2' or 'regression beta' in sl:
        return 'regression beta'
    if sl in ('t', 't-test', 'independent t-test', 't-test of means',
              't-test of mean k', 't-test of  means'):
        return 't-test'
    if sl in ('f-test', 'f-statistic', 'one way anova', 'anova simple effects',
              'anovas/chi-square'):
        return 'F-test / ANOVA'
    if sl in ('chi-square', 'chi-square test'):
        return 'chi-square'
    if sl in ("cohen's d",):
        return "Cohen's d"
    if sl in ('eta-squared',):
        return 'eta-squared'
    if sl in ('odds ratio',):
        return 'odds ratio'
    if sl in ('no', 'qualitative', 'descriptive comparison'):
        return None  # not a quantitative metric
    
    return s  # keep as-is if no match


# ── Main ETL class ──────────────────────────────────────────────────

class BioTDMS_ETL:
    def __init__(self):
        self.graph = Graph()
        self.graph.bind("meas", MEAS)
        self.graph.bind("evid", EVID)
        self.graph.bind("inst", INST)
        self.graph.bind("rdfs", RDFS)
        self.graph.bind("owl", OWL)
        self.graph.bind("skos", SKOS)
        self.graph.bind("xsd", XSD)
        
        # Tracking
        self.report = {
            'publications': {'created': 0, 'excluded': 0},
            'studies': {'created': 0, 'orphan_pub_refs': []},
            'measures': {'created': 0, 'updated': 0},
            'effects': {'created': 0, 'skipped': 0, 'orphan_iv': [], 'orphan_dv': []},
            'manipulations': {'created': 0},
            'constructs': {'new': [], 'existing_used': set()},
            'modalities': {'new': [], 'existing_used': set()},
            'techniques': {'new': [], 'existing_used': set()},
            'unmapped_constructs': [],
            'warnings': [],
        }
        
        # Caches built from existing TTL
        self._existing_modalities = {}   # normalized_label -> URI
        self._existing_techniques = {}
        self._existing_constructs = {}
        self._existing_levels = {}
        self._existing_measures = {}     # measure local_name -> URI
        self._known_pub_ids = set()
        self._known_study_ids = set()
        self._known_measure_ids = set()  # Excel measure_id -> inst: local
        self._known_manip_ids = set()
    
    def load_base_ttls(self, paths):
        """Load base TTL files and build lookup caches."""
        for p in paths:
            if not p:
                continue
            pth = Path(p)
            if not pth.exists():
                print(f"[WARN] Skipping missing: {p}")
                continue
            try:
                self.graph.parse(str(pth), format='turtle')
                print(f"[OK] Loaded {pth.name} ({len(self.graph)} triples total)")
            except Exception as e:
                print(f"[ERROR] Failed to parse {p}: {e}")
                raise
        
        self._build_caches()
    
    def _build_caches(self):
        """Build lookup caches from loaded graph."""
        # Modalities: label -> URI
        for s in self.graph.subjects(RDF.type, MEAS.Modality):
            label = str(self.graph.value(s, RDFS.label, default=""))
            if label:
                self._existing_modalities[label.lower().strip()] = s
        
        # Techniques: label -> URI
        for s in self.graph.subjects(RDF.type, MEAS.analyticTechnique):
            label = str(self.graph.value(s, RDFS.label, default=""))
            if label:
                self._existing_techniques[label.lower().strip()] = s
        
        # Constructs: local_name -> URI
        for s in self.graph.subjects(RDF.type, MEAS.Construct):
            local = str(s).split('#')[-1]
            self._existing_constructs[local] = s
        
        # Levels: label -> URI
        for s in self.graph.subjects(RDF.type, MEAS.levelOfAnalysis):
            label = str(self.graph.value(s, RDFS.label, default=""))
            if label:
                self._existing_levels[label.lower().strip()] = s
        
        # Measures: local_name -> URI
        for s in self.graph.subjects(RDF.type, MEAS.Measure):
            local = str(s).split('#')[-1]
            self._existing_measures[local] = s
        
        print(f"[CACHE] {len(self._existing_modalities)} modalities, "
              f"{len(self._existing_techniques)} techniques, "
              f"{len(self._existing_constructs)} constructs, "
              f"{len(self._existing_levels)} levels, "
              f"{len(self._existing_measures)} measures")
    
    # ── Get-or-create helpers ───────────────────────────────────────
    
    def _get_or_create_modality(self, raw_label):
        """Resolve modality text to URI, creating if needed."""
        label = norm_label(raw_label)
        if not label:
            return None
        
        key = label.lower()
        if key in self._existing_modalities:
            self.report['modalities']['existing_used'].add(key)
            return self._existing_modalities[key]
        
        # Create new
        uri = INST[f"modality_{slugify(label)}"]
        if (uri, RDF.type, MEAS.Modality) not in self.graph:
            self.graph.add((uri, RDF.type, MEAS.Modality))
            self.graph.add((uri, RDFS.label, Literal(label)))
            coarse = infer_coarse_modality(key)
            if coarse:
                self.graph.add((uri, SKOS.broader, MEAS[coarse]))
            self.report['modalities']['new'].append(label)
            self._existing_modalities[key] = uri
        
        return uri
    
    def _get_or_create_technique(self, raw_label):
        """Resolve technique text to URI, creating if needed."""
        label = norm_label(raw_label)
        if not label:
            return None
        
        key = label.lower()
        if key in self._existing_techniques:
            self.report['techniques']['existing_used'].add(key)
            return self._existing_techniques[key]
        
        uri = INST[f"tech_{slugify(label)}"]
        if (uri, RDF.type, MEAS.analyticTechnique) not in self.graph:
            self.graph.add((uri, RDF.type, MEAS.analyticTechnique))
            self.graph.add((uri, RDFS.label, Literal(label)))
            self.report['techniques']['new'].append(label)
            self._existing_techniques[key] = uri
        
        return uri
    
    def _resolve_construct(self, raw_label):
        """Resolve construct text using the curated normalization map."""
        if not raw_label:
            return None
        
        cleaned = normalize_whitespace(raw_label)
        if not cleaned:
            return None
        
        # Look up in curated map
        local_name = CONSTRUCT_MAP.get(cleaned)
        if not local_name:
            self.report['unmapped_constructs'].append(cleaned)
            return None
        
        # Check if it exists in graph already
        if local_name in self._existing_constructs:
            self.report['constructs']['existing_used'].add(local_name)
            return self._existing_constructs[local_name]
        
        # Create new construct
        uri = INST[local_name]
        if (uri, RDF.type, MEAS.Construct) not in self.graph:
            label = NEW_CONSTRUCT_LABELS.get(local_name, cleaned)
            self.graph.add((uri, RDF.type, MEAS.Construct))
            self.graph.add((uri, RDFS.label, Literal(label)))
            self.report['constructs']['new'].append(local_name)
            self._existing_constructs[local_name] = uri
        
        return uri
    
    def _resolve_level(self, raw_label):
        """Resolve level of analysis text to URI."""
        label = norm_label(raw_label)
        if not label:
            return None
        
        key = label.lower()
        if key in self._existing_levels:
            return self._existing_levels[key]
        
        # Create new
        uri = INST[f"level_{slugify(label)}"]
        if (uri, RDF.type, MEAS.levelOfAnalysis) not in self.graph:
            self.graph.add((uri, RDF.type, MEAS.levelOfAnalysis))
            self.graph.add((uri, RDFS.label, Literal(label)))
            self._existing_levels[key] = uri
        
        return uri
    
    def _resolve_measure_uri(self, excel_id):
        """Resolve an Excel measure ID to its inst: URI."""
        if not excel_id or pd.isna(excel_id):
            return None
        
        raw = str(excel_id).strip()
        # Sanitize: remove trailing junk like " + " 
        raw = re.sub(r'\s*\+\s*$', '', raw).strip()
        
        local = normalize_measure_id(raw)
        
        # Check cache
        if local in self._existing_measures:
            return self._existing_measures[local]
        
        # May not exist yet (will be created in map_measures)
        uri = INST[local]
        return uri
    
    def _resolve_iv_uri(self, excel_id):
        """Resolve an IV reference — could be a measure or manipulation."""
        if not excel_id or pd.isna(excel_id):
            return None
        
        raw = str(excel_id).strip()
        
        # Check if it's a manipulation reference
        if raw.lower().startswith('manip_'):
            return INST[raw.lower()]
        
        # Check for free-text (not a clean ID)
        if '(' in raw or len(raw) > 30:
            self.report['warnings'].append(f"Free-text IV skipped: {raw[:50]}")
            return None
        
        # Check for compound manipulation references like "manip_2006 + 2007 +role"
        if 'manip' in raw.lower():
            # Take the first manip ID
            match = re.search(r'(manip_\d+)', raw, re.I)
            if match:
                return INST[match.group(1).lower()]
            return None
        
        return self._resolve_measure_uri(raw)
    
    # ── Sheet processors ────────────────────────────────────────────
    
    def map_publications(self, df):
        """Process publications sheet."""
        excl_col = next((c for c in df.columns if 'exclude' in c.lower()), None)
        
        for _, row in df.iterrows():
            pub_id = str(row.get('publication_id', '')).strip()
            if not pub_id:
                continue
            
            # Check exclusion
            if excl_col:
                excl_val = str(row.get(excl_col, '')).strip().upper()
                if excl_val in ('Y', 'Y?'):
                    self.report['publications']['excluded'] += 1
                    continue
            
            # Sanitize pub_id for URI safety
            safe_pub_id = re.sub(r'[^a-zA-Z0-9_]', '_', pub_id).strip('_')
            if safe_pub_id != pub_id:
                self.report['warnings'].append(
                    f"Sanitized pub_id: '{pub_id}' -> '{safe_pub_id}'")
            
            uri = INST[safe_pub_id]
            self.graph.add((uri, RDF.type, EVID.Publication))
            self.graph.add((uri, RDFS.label, Literal(pub_id)))
            
            if pd.notna(row.get('DOI')):
                doi = str(row['DOI']).strip().replace('\\n', '')
                self.graph.add((uri, EVID.hasDOI, Literal(doi, datatype=XSD.string)))
            
            if pd.notna(row.get('pubYear')):
                self.graph.add((uri, EVID.hasPubYear,
                                Literal(str(int(row['pubYear'])), datatype=XSD.string)))
            
            if pd.notna(row.get('firstAuthor')):
                self.graph.add((uri, EVID.hasFirstAuthor,
                                Literal(str(row['firstAuthor']).strip(), datatype=XSD.string)))
            
            self._known_pub_ids.add(pub_id)
            self.report['publications']['created'] += 1
    
    def map_studies(self, df):
        """Process studies sheet."""
        for _, row in df.iterrows():
            study_id = str(row.get('study_id', '')).strip()
            pub_id = str(row.get('publication_id', '')).strip()
            
            # Sanitize IDs for URI safety
            study_id = re.sub(r'[^a-zA-Z0-9_]', '_', study_id).strip('_')
            pub_id = re.sub(r'[^a-zA-Z0-9_]', '_', pub_id).strip('_')
            
            if not study_id:
                continue
            
            uri = INST[study_id]
            
            # Type assertion
            study_type = str(row.get('studyType', '')).strip().lower()
            if study_type == 'meta-analysis':
                self.graph.add((uri, RDF.type, EVID.metaAnalysis))
            else:
                self.graph.add((uri, RDF.type, EVID.primaryStudy))
            
            self.graph.add((uri, RDFS.label, Literal(study_id)))
            
            # Link to publication
            if pub_id:
                pub_uri = INST[pub_id]
                self.graph.add((pub_uri, EVID.reportsStudy, uri))
                if pub_id not in self._known_pub_ids:
                    self.report['studies']['orphan_pub_refs'].append(
                        f"{study_id} -> {pub_id}")
            
            # Study population
            if pd.notna(row.get('hasStudyPopulation')):
                self.graph.add((uri, EVID.hasStudyPopulation,
                                Literal(str(row['hasStudyPopulation']).strip(),
                                        datatype=XSD.string)))
            
            # Team size (store as note on the study for now)
            if pd.notna(row.get('teamSize')):
                ts = safe_int(row['teamSize'])
                if ts:
                    self.graph.add((uri, RDFS.comment,
                                    Literal(f"Team size: {ts}")))
            
            self._known_study_ids.add(study_id)
            self.report['studies']['created'] += 1
    
    def map_measures(self, df):
        """Process measures sheet."""
        for _, row in df.iterrows():
            raw_id = row.get('measure_id')
            if pd.isna(raw_id):
                continue
            
            raw_id_str = str(raw_id).strip()
            local = normalize_measure_id(raw_id_str)
            uri = INST[local]
            
            is_new = local not in self._existing_measures
            
            self.graph.add((uri, RDF.type, MEAS.Measure))
            
            # Label
            name = normalize_whitespace(row.get('hasName'))
            if name:
                # Use rdfs:label (the standard the app queries use)
                if (uri, RDFS.label, None) not in self.graph:
                    self.graph.add((uri, RDFS.label, Literal(name)))
                # Also add meas:hasName for backward compat
                self.graph.add((uri, MEAS.hasName, Literal(name)))
            
            # Description
            desc = normalize_whitespace(row.get('hasDescription'))
            if desc:
                self.graph.add((uri, MEAS.hasDescription, Literal(desc)))
            
            # Modality
            mod_val = normalize_whitespace(row.get('includesModality'))
            if mod_val:
                mod_uri = self._get_or_create_modality(mod_val)
                if mod_uri:
                    self.graph.add((uri, MEAS.includesModality, mod_uri))
            
            # Analytic technique
            tech_val = normalize_whitespace(row.get('usesAnalyticTechnique'))
            if tech_val:
                # May be comma-separated
                for t in tech_val.split(','):
                    t = t.strip()
                    if t:
                        tech_uri = self._get_or_create_technique(t)
                        if tech_uri:
                            self.graph.add((uri, MEAS.usesAnalyticTechnique, tech_uri))
            
            # Construct (using curated map)
            construct_val = normalize_whitespace(row.get('measuresConstruct'))
            if construct_val:
                cons_uri = self._resolve_construct(construct_val)
                if cons_uri:
                    self.graph.add((uri, MEAS.measuresConstruct, cons_uri))
            
            # Level of analysis
            level_val = normalize_whitespace(row.get('hasLevelOfAnalysis'))
            if level_val:
                level_uri = self._resolve_level(level_val)
                if level_uri:
                    self.graph.add((uri, MEAS.hasLevelOfAnalysis, level_uri))
            
            # Scale
            scale_val = normalize_whitespace(row.get('hasScale'))
            if scale_val:
                self.graph.add((uri, MEAS.hasScale, Literal(scale_val)))
            
            # Interpretation
            interp_val = normalize_whitespace(row.get('hasInterpretation'))
            if interp_val:
                self.graph.add((uri, MEAS.hasInterpretation, Literal(interp_val)))
            
            # Track the Excel ID -> local name mapping
            self._known_measure_ids.add(raw_id_str.lower())
            self._existing_measures[local] = uri
            
            if is_new:
                self.report['measures']['created'] += 1
            else:
                self.report['measures']['updated'] += 1
    
    def map_manipulations(self, df):
        """Process manipulations sheet."""
        for _, row in df.iterrows():
            manip_id = str(row.get('manip_id', '')).strip()
            if not manip_id:
                continue
            
            uri = INST[manip_id.lower()]
            self.graph.add((uri, RDF.type, MEAS.Manipulation))
            
            name = normalize_whitespace(row.get('hasName'))
            if name:
                self.graph.add((uri, RDFS.label, Literal(name)))
            
            desc = normalize_whitespace(row.get('hasDescription'))
            if desc:
                self.graph.add((uri, MEAS.hasDescription, Literal(desc)))
            
            # Link to construct
            cons_val = normalize_whitespace(row.get('manipulatesConstruct'))
            if cons_val:
                cons_uri = self._resolve_construct(cons_val)
                if cons_uri:
                    self.graph.add((uri, MEAS.measuresConstruct, cons_uri))
            
            # Link to study
            study_id = str(row.get('study_id', '')).strip()
            if study_id:
                # Some have typos like "tudy_2004" or use pub_ prefix
                if not study_id.startswith('study_') and not study_id.startswith('pub_'):
                    if study_id.startswith('tudy_'):
                        study_id = 's' + study_id  # fix typo
                
                study_uri = INST[study_id]
                self.graph.add((study_uri, RDFS.comment,
                                Literal(f"Uses manipulation: {manip_id}")))
            
            self._known_manip_ids.add(manip_id.lower())
            self.report['manipulations']['created'] += 1
    
    def map_effects(self, df):
        """Process effects sheet — the core evidence layer."""
        # Find the Category column (may have trailing space)
        cat_col = next((c for c in df.columns if c.strip().lower() == 'category'), None)
        
        for _, row in df.iterrows():
            effect_id = str(row.get('effect_id', '')).strip()
            study_id = str(row.get('study_id', '')).strip()
            
            if not effect_id:
                continue
            
            uri = INST[f"effect_{effect_id.replace('effect_', '')}"]
            self.graph.add((uri, RDF.type, EVID.EffectSize))
            self.graph.add((uri, RDFS.label, Literal(effect_id)))
            
            # Link to study
            if study_id:
                # Normalize: some use pub_ prefix instead of study_
                study_id = re.sub(r'[^a-zA-Z0-9_]', '_', study_id).strip('_')
                study_uri = INST[study_id]
                self.graph.add((study_uri, EVID.reportsEffectSize, uri))
            
            # Independent variable
            iv_raw = normalize_whitespace(row.get('independentVariable'))
            if iv_raw:
                iv_uri = self._resolve_iv_uri(iv_raw)
                if iv_uri:
                    self.graph.add((uri, EVID.hasIndependentVariable, iv_uri))
                else:
                    self.report['effects']['orphan_iv'].append(
                        f"{effect_id}: {iv_raw[:40]}")
            
            # Dependent variable
            dv_raw = normalize_whitespace(row.get('dependentVariable'))
            if dv_raw:
                dv_uri = self._resolve_measure_uri(dv_raw)
                if dv_uri:
                    self.graph.add((uri, EVID.hasDependentVariable, dv_uri))
                else:
                    self.report['effects']['orphan_dv'].append(
                        f"{effect_id}: {dv_raw[:40]}")
            
            # Effect size value
            es_val = safe_float(row.get('hasEffectSizeValue'))
            if es_val is not None:
                self.graph.add((uri, EVID.hasEffectSizeValue,
                                Literal(es_val, datatype=XSD.float)))
            
            # Effect size metric (normalized)
            metric = normalize_effect_metric(row.get('usesEffectSizeMetric'))
            if metric:
                self.graph.add((uri, EVID.usesEffectSizeMetric,
                                Literal(metric, datatype=XSD.string)))
            
            # P-value
            pval = safe_float(row.get('hasPValue'))
            if pval is not None:
                self.graph.add((uri, EVID.hasPValue,
                                Literal(pval, datatype=XSD.float)))
            
            # Standard error
            se = safe_float(row.get('hasSE'))
            if se is not None:
                self.graph.add((uri, EVID.hasSE,
                                Literal(se, datatype=XSD.float)))
            
            # Confidence intervals
            lower_ci = safe_float(row.get('hasLowerCI'))
            if lower_ci is not None:
                self.graph.add((uri, EVID.hasLowerCI,
                                Literal(lower_ci, datatype=XSD.float)))
            
            upper_ci = safe_float(row.get('hasUpperCI'))
            if upper_ci is not None:
                self.graph.add((uri, EVID.hasUpperCI,
                                Literal(upper_ci, datatype=XSD.float)))
            
            # Sample sizes
            team_n = safe_float(row.get('teamSampleSize'))
            if team_n is not None:
                self.graph.add((uri, EVID.hasTeamSampleSize,
                                Literal(team_n, datatype=XSD.float)))
            
            indiv_n = safe_float(row.get('individualSampleSize'))
            if indiv_n is not None:
                self.graph.add((uri, EVID.hasIndividualSampleSize,
                                Literal(indiv_n, datatype=XSD.float)))
            
            # Perturbation phase
            perturb = normalize_whitespace(row.get('perturbationPhase'))
            if perturb:
                self.graph.add((uri, EVID.perturbationPhase,
                                Literal(perturb, datatype=XSD.string)))
            
            # Effect level
            effect_level = normalize_whitespace(row.get('hasEffectLevel'))
            if effect_level and effect_level.lower() not in ('significant', 'null', 'marginal'):
                # Filter out values that are actually significance categories, not levels
                self.graph.add((uri, EVID.hasEffectLevel,
                                Literal(effect_level, datatype=XSD.string)))
            
            # Description / notes
            desc = normalize_whitespace(row.get('description'))
            if desc:
                self.graph.add((uri, EVID.hasNotes,
                                Literal(desc, datatype=XSD.string)))
            
            notes = normalize_whitespace(row.get('notes'))
            if notes:
                existing_notes = self.graph.value(uri, EVID.hasNotes)
                if existing_notes:
                    combined = f"{existing_notes}; {notes}"
                    self.graph.set((uri, EVID.hasNotes,
                                   Literal(combined, datatype=XSD.string)))
                else:
                    self.graph.add((uri, EVID.hasNotes,
                                    Literal(notes, datatype=XSD.string)))
            
            # Category split: significance vs domain
            if cat_col:
                cat_val = normalize_whitespace(row.get(cat_col))
                if cat_val:
                    if cat_val in SIGNIFICANCE_CATEGORIES:
                        self.graph.add((uri, EVID.hasSignificanceCategory,
                                        Literal(cat_val, datatype=XSD.string)))
                    else:
                        self.graph.add((uri, EVID.hasEffectDomain,
                                        Literal(cat_val, datatype=XSD.string)))
            
            self.report['effects']['created'] += 1
    
    # ── Schema updates ──────────────────────────────────────────────
    
    def add_schema_extensions(self):
        """Add new properties to the evidence schema for category split."""
        # evid:hasSignificanceCategory
        sig_prop = EVID.hasSignificanceCategory
        if (sig_prop, RDF.type, OWL.DatatypeProperty) not in self.graph:
            self.graph.add((sig_prop, RDF.type, OWL.DatatypeProperty))
            self.graph.add((sig_prop, RDFS.domain, EVID.EffectSize))
            self.graph.add((sig_prop, RDFS.range, XSD.string))
            self.graph.add((sig_prop, RDFS.label,
                            Literal("has significance category")))
            self.graph.add((sig_prop, RDFS.comment,
                            Literal("Categorical significance: NS, Significantly positive, Significantly negative")))
        
        # evid:hasEffectDomain
        dom_prop = EVID.hasEffectDomain
        if (dom_prop, RDF.type, OWL.DatatypeProperty) not in self.graph:
            self.graph.add((dom_prop, RDF.type, OWL.DatatypeProperty))
            self.graph.add((dom_prop, RDFS.domain, EVID.EffectSize))
            self.graph.add((dom_prop, RDFS.range, XSD.string))
            self.graph.add((dom_prop, RDFS.label,
                            Literal("has effect domain")))
            self.graph.add((dom_prop, RDFS.comment,
                            Literal("Semantic domain of the effect (e.g., Performance effect, Coordination effect)")))
        
        # meas:Manipulation class (if not present)
        if (MEAS.Manipulation, RDF.type, OWL.Class) not in self.graph:
            self.graph.add((MEAS.Manipulation, RDF.type, OWL.Class))
            self.graph.add((MEAS.Manipulation, RDFS.label,
                            Literal("Manipulation")))
            self.graph.add((MEAS.Manipulation, RDFS.comment,
                            Literal("Experimental manipulation used as independent variable")))
    
    # ── Main process ────────────────────────────────────────────────
    
    def process_excel(self, excel_path):
        """Process all sheets from the Excel workbook."""
        print(f"\n{'='*60}")
        print(f"Processing: {excel_path}")
        print(f"{'='*60}")
        
        # Add schema extensions first
        self.add_schema_extensions()
        
        # Load sheets
        xf = pd.ExcelFile(excel_path)
        
        # 1. Publications (filter exclusions)
        if 'publication' in xf.sheet_names:
            print("\n[1/5] Publications...")
            df = pd.read_excel(excel_path, sheet_name='publication')
            self.map_publications(df)
        
        # 2. Studies
        if 'studies' in xf.sheet_names:
            print("[2/5] Studies...")
            df = pd.read_excel(excel_path, sheet_name='studies')
            self.map_studies(df)
        
        # 3. Measures (must come before effects for URI resolution)
        if 'measures' in xf.sheet_names:
            print("[3/5] Measures...")
            df = pd.read_excel(excel_path, sheet_name='measures')
            self.map_measures(df)
        
        # 4. Manipulations (must come before effects)
        if 'manipulations' in xf.sheet_names:
            print("[4/5] Manipulations...")
            df = pd.read_excel(excel_path, sheet_name='manipulations')
            self.map_manipulations(df)
        
        # 5. Effects
        if 'effects' in xf.sheet_names:
            print("[5/5] Effects...")
            df = pd.read_excel(excel_path, sheet_name='effects')
            self.map_effects(df)
        
        self._print_report()
    
    def _print_report(self):
        """Print processing summary."""
        r = self.report
        print(f"\n{'='*60}")
        print("ETL REPORT")
        print(f"{'='*60}")
        print(f"Publications:  {r['publications']['created']} created, "
              f"{r['publications']['excluded']} excluded")
        print(f"Studies:       {r['studies']['created']} created")
        print(f"Measures:      {r['measures']['created']} new, "
              f"{r['measures']['updated']} updated")
        print(f"Manipulations: {r['manipulations']['created']} created")
        print(f"Effects:       {r['effects']['created']} created")
        
        if r['constructs']['new']:
            print(f"\nNew constructs created ({len(r['constructs']['new'])}):")
            for c in r['constructs']['new']:
                print(f"  + {c}")
        
        if r['modalities']['new']:
            print(f"\nNew modalities created ({len(r['modalities']['new'])}):")
            for m in r['modalities']['new']:
                print(f"  + {m}")
        
        if r['techniques']['new']:
            print(f"\nNew techniques created ({len(r['techniques']['new'])}):")
            for t in r['techniques']['new']:
                print(f"  + {t}")
        
        if r['unmapped_constructs']:
            unique_unmapped = sorted(set(r['unmapped_constructs']))
            print(f"\nUNMAPPED constructs ({len(unique_unmapped)}):")
            for u in unique_unmapped:
                print(f"  ? {u}")
        
        if r['studies']['orphan_pub_refs']:
            print(f"\nOrphan study->pub refs ({len(r['studies']['orphan_pub_refs'])}):")
            for o in r['studies']['orphan_pub_refs'][:10]:
                print(f"  ! {o}")
        
        if r['effects']['orphan_iv']:
            print(f"\nOrphan effect IV refs ({len(r['effects']['orphan_iv'])}):")
            for o in r['effects']['orphan_iv'][:10]:
                print(f"  ! {o}")
        
        if r['warnings']:
            print(f"\nWarnings ({len(r['warnings'])}):")
            for w in r['warnings'][:10]:
                print(f"  ~ {w}")
        
        print(f"\nTotal triples: {len(self.graph)}")
    
    def save(self, output_path):
        """Save the graph to TTL."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.graph.serialize(str(out), format='turtle')
        print(f"\nSaved: {out} ({len(self.graph)} triples)")
    
    def save_report(self, output_path):
        """Save the report as JSON for downstream use."""
        r = self.report.copy()
        # Convert sets to lists for JSON
        r['constructs']['existing_used'] = sorted(r['constructs']['existing_used'])
        r['modalities']['existing_used'] = sorted(r['modalities']['existing_used'])
        r['techniques']['existing_used'] = sorted(r['techniques']['existing_used'])
        
        out = Path(output_path)
        with open(out, 'w') as f:
            json.dump(r, f, indent=2)
        print(f"Report saved: {out}")


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="BioTDMS Excel-to-TTL ETL")
    ap.add_argument("--excel", required=True, help="Path to Excel coding workbook")
    ap.add_argument("--team-ttl", required=True, help="Path to teamMeasurement.ttl")
    ap.add_argument("--evidence-ttl", required=True, help="Path to evidence.ttl")
    ap.add_argument("--old-instances", default=None,
                    help="Optional existing instances.ttl to load")
    ap.add_argument("--out-ttl", required=True,
                    help="Output TTL path (standalone)")
    ap.add_argument("--merge", action="store_true",
                    help="Also write a merged file combining old instances + new")
    args = ap.parse_args()
    
    etl = BioTDMS_ETL()
    etl.load_base_ttls([args.team_ttl, args.evidence_ttl, args.old_instances])
    etl.process_excel(args.excel)
    etl.save(args.out_ttl)
    
    # Save report
    report_path = Path(args.out_ttl).with_suffix('.report.json')
    etl.save_report(report_path)
    
    if args.merge and args.old_instances:
        merged_path = Path(args.out_ttl).parent / "merged_instances.ttl"
        print(f"\nMerged output: {merged_path}")
        etl.save(str(merged_path))


if __name__ == "__main__":
    main()
