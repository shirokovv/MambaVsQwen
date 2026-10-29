import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from results import ExperimentResult


class BenchmarkDataset:
    def __init__(self, path=None):
        path = path or Path(__file__).resolve().parent.parent / "data/dataset.json"
        self.info = json.loads(Path(path).read_text())

    def __len__(self):
        return self.info["size"]


class Experiment:
    def __init__(self, clients, database):
        self.clients = clients
        self.database = database
        self.dataset = BenchmarkDataset()

    def run(self, limit):
        run_id = uuid4().hex
        started_at = datetime.now(timezone.utc).isoformat()
        for client in self.clients:
            metrics = client.evaluate(limit)
            expected = len(self.dataset) if limit is None else limit
            if (metrics["n_samples"] != expected or
                    metrics["parameters"]["dataset_fingerprint"] != self.dataset.info["fingerprint"] or
                    metrics["model"] != client.name):
                raise ValueError("Worker вернул другую модель или выборку датасета")
            result = ExperimentResult(dict(metrics, run_id=run_id, started_at=started_at))
            # Каждый завершённый результат сохраняется сразу, даже если следующий worker упадёт.
            self.database.save(result)
            yield result
