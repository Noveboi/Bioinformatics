import json
from pathlib import Path


class Cache:
    """
    A simple JSON file-based cache for storing/retrieving data.

    Used for asynchronous flow of the various 'steps' of the system.
    """

    def __init__(self, cache_path: str) -> None:
        self.dir = Path(cache_path)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _filepath(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def save(self, key: str, value: dict) -> Path:
        """
        Save a dictionary with the given key.

        Returns
        --------
        The path of the saved file.
        """
        filepath = self._filepath(key)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2)

        return filepath

    def load(self, key: str) -> dict:
        """
        Load a dictionary with the given key from the file system.
        """
        filepath = self._filepath(key)

        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
