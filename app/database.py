import json
import sqlite3
from pathlib import Path

from results import ExperimentResult


class Database:
    def __init__(self, path=None):
        self.path = Path(path) if path is not None else Path(__file__).resolve().parent.parent / "data/experiments.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, started_at TEXT NOT NULL,
                model TEXT NOT NULL, dataset TEXT NOT NULL, split TEXT NOT NULL,
                n_samples INTEGER NOT NULL, accuracy REAL NOT NULL, perplexity REAL NOT NULL,
                total_seconds REAL NOT NULL, seconds_per_sample REAL NOT NULL,
                model_revision TEXT NOT NULL, parameters TEXT NOT NULL,
                peak_mlx_gb REAL, UNIQUE(run_id, model))""")

    def save(self, result):
        values = dict(result.values)
        values["parameters"] = json.dumps(values["parameters"], ensure_ascii=False)
        columns = ", ".join(values)
        placeholders = ", ".join("?" for _ in values)
        with sqlite3.connect(self.path) as connection:
            connection.execute(f"INSERT INTO experiments ({columns}) VALUES ({placeholders})",
                               list(values.values()))

    def load(self):
        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute("SELECT * FROM experiments ORDER BY id DESC").fetchall()
        results = []
        for row in rows:
            values = dict(row)
            values["parameters"] = json.loads(values["parameters"])
            results.append(ExperimentResult(values))
        return results
