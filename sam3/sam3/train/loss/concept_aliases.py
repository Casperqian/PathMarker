"""Explicit pretraining synonym identities, independent of model embeddings."""
import json
from pathlib import Path

from .hierarchy_data import normalize_name


class ConceptAliases:
    def __init__(self, path=None):
        path = Path(path) if path is not None else Path(__file__).with_name('hierarchy_set') / 'concept_aliases.json'
        data = json.loads(path.read_text(encoding='utf-8'))
        self.mapping = {}
        for group in data['groups']:
            canonical = normalize_name(group['canonical'])
            for value in [canonical, *group['aliases']]:
                name = normalize_name(value)
                previous = self.mapping.get(name)
                if previous is not None and previous != canonical:
                    raise ValueError(f'Alias {name!r} belongs to both {previous!r} and {canonical!r}')
                self.mapping[name] = canonical

    def canonicalize(self, name):
        name = normalize_name(name)
        return self.mapping.get(name, name)
