import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Optional, Union


@dataclass(frozen=True)
class HierarchyData:
    seed_edges: tuple[tuple[str, str], ...]
    seed_parents: Mapping[str, frozenset[str]]
    concepts: tuple[str, ...]


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError("concept name must be a string")
    normalized = " ".join(name.casefold().split())
    if not normalized:
        raise ValueError("concept name must not be empty")
    return normalized


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"hierarchy file must contain a JSON object: {path}")
    return data


def load_hierarchy_data(
    relation_path: Optional[Union[str, Path]] = None,
    concept_path: Optional[Union[str, Path]] = None,
) -> HierarchyData:
    hierarchy_dir = Path(__file__).with_name("hierarchy_set")
    relation_path = (
        Path(relation_path)
        if relation_path is not None
        else hierarchy_dir / "seed_hierarchy.json"
    )
    concept_path = (
        Path(concept_path)
        if concept_path is not None
        else hierarchy_dir / "umls_concept_hierarchy.json"
    )

    relation_data = _read_json(relation_path)
    relations = relation_data.get("relations")
    if not isinstance(relations, list):
        raise ValueError("hierarchy relations must be a list")

    edges = set()
    for index, relation in enumerate(relations):
        if not isinstance(relation, dict) or set(relation) != {"child", "parent"}:
            raise ValueError(
                f"relation {index} must contain exactly child and parent"
            )
        child = normalize_name(relation["child"])
        parent = normalize_name(relation["parent"])
        edge = (child, parent)
        if child == parent:
            raise ValueError(f"self-edge is not allowed: {child!r}")
        if edge in edges:
            raise ValueError(f"duplicate hierarchy edge: {edge!r}")
        if (parent, child) in edges:
            raise ValueError(f"reverse hierarchy conflict: {edge!r}")
        edges.add(edge)

    concept_data = _read_json(concept_path)
    concept_records = concept_data.get("concepts")
    if not isinstance(concept_records, list):
        raise ValueError("concept vocabulary must contain a concepts list")

    concepts = set()
    for index, record in enumerate(concept_records):
        if not isinstance(record, dict) or "concept" not in record:
            raise ValueError(f"concept record {index} must contain concept")
        concepts.add(normalize_name(record["concept"]))

    for child, parent in edges:
        concepts.add(child)
        concepts.add(parent)

    sorted_edges = tuple(sorted(edges))
    parents_by_child = {}
    for child, parent in sorted_edges:
        parents_by_child.setdefault(child, set()).add(parent)
    seed_parents = MappingProxyType(
        {
            child: frozenset(parents)
            for child, parents in sorted(parents_by_child.items())
        }
    )
    return HierarchyData(
        seed_edges=sorted_edges,
        seed_parents=seed_parents,
        concepts=tuple(sorted(concepts)),
    )
