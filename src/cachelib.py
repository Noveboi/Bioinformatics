import json
import logging
import pickle
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class Cache:
    """
    A simple file-based cache for storing/retrieving data.

    Supports both:
      - JSON for human-readable structured data
      - pickle for Python objects
    """

    def __init__(self, cache_path: str) -> None:
        self.dir = Path(cache_path)
        self._checked = False

    def _ensure_dir_exists(self) -> None:
        if not self._checked:
            self.dir.mkdir(parents=True, exist_ok=True)
            self._checked = True

    def _filepath(self, key: str, suffix: str) -> Path:
        return self.dir / f"{key}.{suffix}"

    def save_json(self, key: str, value: Any) -> Path:
        """
        Save a JSON-serializable object.
        """
        self._ensure_dir_exists()
        filepath = self._filepath(key, "json")

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, ensure_ascii=False)

        log.info("Saved JSON '%s' to '%s'", key, filepath)
        return filepath

    def load_json(self, key: str) -> Any:
        """
        Load a JSON object.
        """
        filepath = self._filepath(key, "json")

        with open(filepath, "r", encoding="utf-8") as f:
            obj = json.load(f)

        log.info("Loaded JSON '%s' from '%s'", key, filepath)
        return obj

    def save_pickle(self, key: str, value: Any) -> Path:
        """
        Save any pickle-serializable Python object.
        """
        self._ensure_dir_exists()
        filepath = self._filepath(key, "pkl")

        with open(filepath, "wb") as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)

        log.info("Saved pickle '%s' to '%s'", key, filepath)
        return filepath

    def load_pickle(self, key: str) -> Any:
        """
        Load a pickled Python object.
        """
        filepath = self._filepath(key, "pkl")

        with open(filepath, "rb") as f:
            obj = pickle.load(f)

        log.info("Loaded pickle '%s' from '%s'", key, filepath)
        return obj
