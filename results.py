import glob
import json
import os


class ResultsIndex:
    """Queryable index over all experiment results.

    Scans all .meta.json sidecars under base_dir and provides filtering
    by any metadata field, including nested fields (optimizer.weight_decay,
    model.dim, etc.) via flat-key lookup.

    Usage:
        index = ResultsIndex("data")
        index.query(experiment_type="speed", p=113)
        index.query(weight_decay=1.0, depth=2)
        index.query(param_count=lambda x: 10000 < x < 50000)
    """

    def __init__(self, base_dir: str = "data"):
        self.base_dir = base_dir
        self._entries: list = []
        self._scan()

    def _scan(self):
        pattern = os.path.join(self.base_dir, '**', '*.meta.json')
        self._entries = []
        for json_path in glob.glob(pattern, recursive=True):
            try:
                with open(json_path) as f:
                    entry = json.load(f)
                entry['_json_path'] = json_path
                # Strip '.meta.json' (not just '.json') to get the npz stem
                entry['_npz_path'] = json_path[:-len('.meta.json')] + '.npz'
                self._entries.append(entry)
            except (json.JSONDecodeError, OSError):
                pass

    def refresh(self):
        """Re-scan the directory (e.g. after new files are written)."""
        self._scan()

    def _get_nested(self, d: dict, key: str):
        """Search for key in d, checking top-level then recursing into nested dicts."""
        if key in d:
            return d[key], True
        for v in d.values():
            if isinstance(v, dict):
                val, found = self._get_nested(v, key)
                if found:
                    return val, True
        return None, False

    def query(self, **filters) -> list:
        """Return entries matching all filters.

        Filter values can be:
        - A scalar (equality check)
        - A callable (called with the field value; must return True to match)
        """
        results = []
        for entry in self._entries:
            match = True
            for key, expected in filters.items():
                val, _ = self._get_nested(entry, key)  # val is None when not found
                if callable(expected):
                    if not expected(val):
                        match = False
                        break
                elif val != expected:
                    match = False
                    break
            if match:
                results.append(entry)
        return results

    def exists(self, **filters) -> bool:
        """Return True if at least one entry matches all filters."""
        return len(self.query(**filters)) > 0

    def load_traces(self, entry: dict) -> dict:
        """Load the npz numeric traces for a given entry dict."""
        import numpy as np
        data = np.load(entry['_npz_path'], allow_pickle=True)
        return {key: data[key].item() if data[key].ndim == 0 else data[key]
                for key in data.files}

    def unique(self, field: str) -> list:
        """Return sorted unique values of a field across all entries."""
        vals = set()
        for entry in self._entries:
            v, found = self._get_nested(entry, field)
            if found and v is not None:
                try:
                    vals.add(v)
                except TypeError:
                    pass  # unhashable (e.g. list); skip
        try:
            return sorted(vals)
        except TypeError:
            return list(vals)

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"ResultsIndex(base_dir={self.base_dir!r}, n_entries={len(self._entries)})"
