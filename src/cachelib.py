import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


class Cache:
    """
    A simple JSON file-based cache for storing/retrieving data.

    Used for asynchronous flow of the various 'steps' of the system.
    """

    def __init__(self, cache_path: str) -> None:
        self.dir = Path(cache_path)
        self._checked = False

    def _filepath(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def _ensure_dir_exists(self):
        if not self._checked:
            self.dir.mkdir(parents=True, exist_ok=True)

    def save(self, key: str, value: dict) -> Path:
        """
        Save a dictionary with the given key.

        Returns
        --------
        The path of the saved file.
        """
        self._ensure_dir_exists()

        filepath = self._filepath(key)

        with open(filepath, "w") as f:
            json.dump(value, f, indent=2)

        log.info("Saved '%s' to '%s'", key, filepath)

        return filepath

    def load(self, key: str) -> dict:
        """
        Load a dictionary with the given key from the file system.
        """
        filepath = self._filepath(key)

        with open(filepath, "r") as f:
            obj = json.load(f)

        log.info("Loaded '%s' from '%s'", key, filepath)

        return obj
