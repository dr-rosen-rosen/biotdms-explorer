"""
BioTDMS Ontology Access Layer

Provides clean interfaces for querying the RDF/Turtle ontology.
All UI components should use this module rather than direct SPARQL queries.
"""

from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from rdflib import Graph, Namespace

# Namespace definitions
MEAS = Namespace("http://example.org/ontology/teamMeasurement#")
INST = Namespace("http://example.org/ontology/instances#")
EVID = Namespace("http://example.org/ontology/evidence#")
RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")


@dataclass
class Measure:
    """Represents a measurement/signature from the ontology"""
    uri: str
    label: str
    description: Optional[str]
    modality: Optional[str]
    modality_category: Optional[str]
    technique: Optional[str]
    construct: Optional[str]
    level: Optional[str]


@dataclass 
class Construct:
    """Represents a team construct from the ontology"""
    uri: str
    label: str
    description: Optional[str] = None


@dataclass
class Modality:
    """Represents a measurement modality"""
    uri: str
    label: str
    category: Optional[str] = None


@dataclass
class AnalyticTechnique:
    """Represents an analytic technique"""
    uri: str
    label: str
    category: Optional[str] = None


class OntologyAccess:
    """Main interface for ontology queries."""
    
    def __init__(self, ontology_path: str | Path):
        self.ontology_path = Path(ontology_path)
        self._graph: Optional[Graph] = None
    
    @property
    def graph(self) -> Graph:
        """Lazy-load the graph"""
        if self._graph is None:
            self._graph = Graph()
            self._graph.parse(self.ontology_path, format="turtle")
        return self._graph
    
    def _extract_local_name(self, uri: str) -> str:
        """Extract local name from URI"""
        if '#' in uri:
            return uri.split('#')[-1]
        return uri.split('/')[-1]
    
    def get_all_constructs(self) -> List[Construct]:
        """Get all team constructs defined in the ontology"""
        query = """
        PREFIX meas: <http://example.org/ontology/teamMeasurement#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT DISTINCT ?uri ?label ?desc WHERE {
            ?uri a meas:Construct .
            OPTIONAL { ?uri rdfs:label ?label }
            OPTIONAL { ?uri rdfs:comment ?desc }
        }
        ORDER BY ?label
        """
        results = []
        for row in self.graph.query(query):
            results.append(Construct(
                uri=str(row.uri),
                label=str(row.label) if row.label else self._extract_local_name(str(row.uri)),
                description=str(row.desc) if row.desc else None
            ))
        return results
    
    def get_all_modalities(self) -> List[Modality]:
        """Get all modalities with their parent categories"""
        query = """
        PREFIX meas: <http://example.org/ontology/teamMeasurement#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        SELECT DISTINCT ?uri ?label ?broader WHERE {
            ?uri a meas:Modality .
            OPTIONAL { ?uri rdfs:label ?label }
            OPTIONAL { ?uri skos:broader ?broader }
        }
        ORDER BY ?broader ?label
        """
        results = []
        for row in self.graph.query(query):
            results.append(Modality(
                uri=str(row.uri),
                label=str(row.label) if row.label else self._extract_local_name(str(row.uri)),
                category=self._extract_local_name(str(row.broader)) if row.broader else None
            ))
        return results
    
    def get_modality_categories(self) -> List[str]:
        """Get unique top-level modality categories"""
        modalities = self.get_all_modalities()
        categories = set()
        for m in modalities:
            if m.category:
                categories.add(m.category)
            else:
                categories.add(m.label)
        return sorted(categories)
    
    def get_all_techniques(self) -> List[AnalyticTechnique]:
        """Get all analytic techniques"""
        query = """
        PREFIX meas: <http://example.org/ontology/teamMeasurement#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT DISTINCT ?uri ?label WHERE {
            ?uri a meas:analyticTechnique .
            OPTIONAL { ?uri rdfs:label ?label }
        }
        ORDER BY ?label
        """
        results = []
        for row in self.graph.query(query):
            label = str(row.label) if row.label else self._extract_local_name(str(row.uri))
            category = None
            if ' - ' in label:
                category = label.split(' - ')[0].strip()
            results.append(AnalyticTechnique(
                uri=str(row.uri),
                label=label,
                category=category
            ))
        return results
    
    def get_technique_categories(self) -> List[str]:
        """Get unique technique categories"""
        techniques = self.get_all_techniques()
        categories = set()
        for t in techniques:
            if t.category:
                categories.add(t.category)
        return sorted(categories)
    
    def get_all_levels(self) -> List[str]:
        """Get all levels of analysis"""
        query = """
        PREFIX meas: <http://example.org/ontology/teamMeasurement#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT DISTINCT ?label WHERE {
            ?uri a meas:levelOfAnalysis .
            OPTIONAL { ?uri rdfs:label ?label }
        }
        """
        return [str(row.label) for row in self.graph.query(query) if row.label]
    
    def get_all_measures(self) -> List[Measure]:
        """Get all measures/signatures from the ontology"""
        query = """
        PREFIX meas: <http://example.org/ontology/teamMeasurement#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        SELECT ?uri ?label ?desc ?modality ?modLabel ?modBroader ?technique ?techLabel ?construct ?constLabel ?level ?levelLabel
        WHERE {
            ?uri a meas:Measure .
            OPTIONAL { ?uri rdfs:label ?label }
            OPTIONAL { ?uri meas:hasDescription ?desc }
            OPTIONAL { 
                ?uri meas:includesModality ?modality .
                OPTIONAL { ?modality rdfs:label ?modLabel }
                OPTIONAL { ?modality skos:broader ?modBroader }
            }
            OPTIONAL { 
                ?uri meas:usesAnalyticTechnique ?technique .
                OPTIONAL { ?technique rdfs:label ?techLabel }
            }
            OPTIONAL { 
                ?uri meas:measuresConstruct ?construct .
                OPTIONAL { ?construct rdfs:label ?constLabel }
            }
            OPTIONAL { 
                ?uri meas:hasLevelOfAnalysis ?level .
                OPTIONAL { ?level rdfs:label ?levelLabel }
            }
        }
        ORDER BY ?label
        """
        results = []
        for row in self.graph.query(query):
            mod_category = None
            if row.modBroader:
                mod_category = self._extract_local_name(str(row.modBroader))
            elif row.modality:
                mod_category = str(row.modLabel) if row.modLabel else self._extract_local_name(str(row.modality))
            
            results.append(Measure(
                uri=str(row.uri),
                label=str(row.label) if row.label else self._extract_local_name(str(row.uri)),
                description=str(row.desc) if row.desc else None,
                modality=str(row.modLabel) if row.modLabel else (self._extract_local_name(str(row.modality)) if row.modality else None),
                modality_category=mod_category,
                technique=str(row.techLabel) if row.techLabel else (self._extract_local_name(str(row.technique)) if row.technique else None),
                construct=str(row.constLabel) if row.constLabel else (self._extract_local_name(str(row.construct)) if row.construct else None),
                level=str(row.levelLabel) if row.levelLabel else (self._extract_local_name(str(row.level)) if row.level else None)
            ))
        return results
    
    def get_measures_by_construct(self, construct_label: str) -> List[Measure]:
        """Get measures that assess a specific construct"""
        all_measures = self.get_all_measures()
        return [m for m in all_measures if m.construct and construct_label.lower() in m.construct.lower()]
    
    def get_measures_by_modality_category(self, category: str) -> List[Measure]:
        """Get measures within a modality category"""
        all_measures = self.get_all_measures()
        return [m for m in all_measures if m.modality_category and category.lower() in m.modality_category.lower()]
    
    def get_measure_by_uri(self, uri: str) -> Optional[Measure]:
        """Get a specific measure by URI"""
        all_measures = self.get_all_measures()
        for m in all_measures:
            if m.uri == uri:
                return m
        return None
    
    def filter_measures(
        self,
        construct: Optional[str] = None,
        modality_category: Optional[str] = None,
        technique_category: Optional[str] = None,
        level: Optional[str] = None
    ) -> List[Measure]:
        """Filter measures by multiple criteria"""
        measures = self.get_all_measures()
        
        if construct:
            measures = [m for m in measures if m.construct and construct.lower() in m.construct.lower()]
        if modality_category:
            measures = [m for m in measures if m.modality_category and modality_category.lower() in m.modality_category.lower()]
        if technique_category:
            measures = [m for m in measures if m.technique and technique_category.lower() in m.technique.lower()]
        if level:
            measures = [m for m in measures if m.level and level.lower() in m.level.lower()]
        
        return measures
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics about the ontology"""
        return {
            "total_measures": len(self.get_all_measures()),
            "total_constructs": len(self.get_all_constructs()),
            "total_modalities": len(self.get_all_modalities()),
            "modality_categories": len(self.get_modality_categories()),
            "total_techniques": len(self.get_all_techniques()),
            "technique_categories": len(self.get_technique_categories()),
            "levels_of_analysis": len(self.get_all_levels()),
            "total_triples": len(self.graph)
        }


def load_ontology(path: str | Path = "instances.ttl") -> OntologyAccess:
    """Load ontology from default or specified path"""
    return OntologyAccess(path)
