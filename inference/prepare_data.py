"""Подготавливает стандартную задачу LAMBADA из lm-eval."""
import json
from pathlib import Path

import lm_eval
import yaml
from lm_eval.tasks import TaskManager


DATASET_REVISION = "5953bd97664b64b95754f299b2309ecfbfbe81b9"


def load_task():
    path = Path(lm_eval.__file__).parent / "tasks/lambada/lambada_standard.yaml"
    spec = yaml.safe_load(path.read_text())
    spec["dataset_kwargs"] = {"revision": DATASET_REVISION}
    return TaskManager().load(spec)["tasks"]["lambada_standard"]


if __name__ == "__main__":
    task = load_task()
    docs = task.test_docs()
    info = {"name": "cimec/lambada", "task": "lambada_standard", "split": "test",
            "size": len(docs), "fingerprint": docs._fingerprint, "revision": DATASET_REVISION}
    data = Path(__file__).resolve().parent.parent / "data"
    data.mkdir(exist_ok=True)
    (data / "dataset.json").write_text(json.dumps(info, indent=2))
    print(info)
