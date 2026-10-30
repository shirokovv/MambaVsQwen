"""Один нативный MLX worker; запускать отдельно для каждой модели."""
import argparse
import fcntl
import json
import math
import os
import signal
import sys
import threading
import traceback
from functools import partial, wraps
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import mlx.core as mx
from lm_eval import simple_evaluate
from lm_eval.api.instance import Instance
from mlx_lm.evaluate import MLXLM

from prepare_data import DATASET_REVISION, load_task


def measure_time(function):
    @wraps(function)
    def inner(*args, **kwargs):
        mx.synchronize()
        start = perf_counter()
        result = function(*args, **kwargs)
        mx.synchronize()
        return result, perf_counter() - start
    return inner


@measure_time
def evaluate(model, task, limit):
    return simple_evaluate(
        model=model, tasks=[task], num_fewshot=0, limit=limit, batch_size=1,
        apply_chat_template=False, system_instruction=None,
        log_samples=False, bootstrap_iters=0, torch_random_seed=None,
    )


class ModelServer(ThreadingHTTPServer):
    def __init__(self, model_path, port):
        super().__init__(("127.0.0.1", port), Handler)
        if not mx.metal.is_available():
            raise RuntimeError("Metal GPU недоступен")
        mx.set_default_device(mx.gpu)
        mx.set_cache_limit(256 * 1024**2)
        self.provenance = json.loads(Path(model_path, "provenance.json").read_text())
        self.name = self.provenance["original"].split("/")[-1]
        self.lock_path = Path(__file__).resolve().parent / "logs/evaluation.lock"
        self.lock_path.parent.mkdir(exist_ok=True)
        self.busy = threading.Lock()
        self.model = MLXLM(model_path, batch_size=1, use_chat_template=False,
                           max_tokens=8192)
        # Только размер prefill-блока готового адаптера: важен для RAM Mamba.
        self.model._process_prompt = partial(self.model._process_prompt, step_size=256)
        self.task = load_task()
        self.docs = self.task.test_docs()
        warmup = Instance(request_type="loglikelihood", doc={},
                          arguments=("The capital of France is Paris. " * 16 +
                                     "The capital of France is", " Paris"), idx=0)
        score, _ = self.model.loglikelihood([warmup])[0]
        if not math.isfinite(score):
            raise RuntimeError("Прогрев модели вернул нечисловой log-likelihood")
        mx.synchronize()
        print(f"READY {self.name}, Metal GPU, warmup logp={score:.4f}", flush=True)

    def run_evaluation(self, limit):
        report, seconds = evaluate(self.model, self.task, limit)
        metrics = report["results"]["lambada_standard"]
        count = report["n-samples"]["lambada_standard"]["effective"]
        parameters = dict(self.provenance, limit=limit, task="lambada_standard",
                          num_fewshot=0, apply_chat_template=False, batch_size=1,
                          prefill_step_size=64, max_tokens=8192,
                          dataset_fingerprint=self.docs._fingerprint,
                          dataset_revision=DATASET_REVISION,
                          versions={name: version(name) for name in
                                    ["mlx", "mlx-lm", "lm-eval", "datasets", "transformers"]})
        return dict(model=self.name, dataset="cimec/lambada", split="test",
                    n_samples=count, accuracy=float(metrics["acc,none"]),
                    perplexity=float(metrics["perplexity,none"]),
                    total_seconds=seconds, seconds_per_sample=seconds / count,
                    model_revision=self.provenance["revision"], parameters=parameters)


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, data):
        body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        if self.path != "/health":
            self.reply(404, {"error": "Unknown endpoint"})
            return
        self.reply(200, {"status": "ready", "model": self.server.name,
                         "device": "Metal GPU", "pid": os.getpid(),
                         "busy": self.server.busy.locked()})

    def do_POST(self):
        if self.path != "/evaluate":
            self.reply(404, {"error": "Unknown endpoint"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 1024:
                raise ValueError("Нужен JSON размером до 1024 байт")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("Нужен JSON-объект")
            limit = payload.get("limit", 10)
            if limit is not None and (type(limit) is not int or limit not in [10, 100, 1000]):
                raise ValueError("limit: 10, 100, 1000 или null (полный test split)")
        except (ValueError, UnicodeError) as error:
            self.reply(400, {"error": str(error)})
            return
        if not self.server.busy.acquire(blocking=False):
            self.reply(409, {"error": "Эта модель уже выполняет эксперимент"})
            return
        try:
            # Общий файл не позволяет двум workers одновременно измерять GPU.
            with self.server.lock_path.open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    self.reply(409, {"error": "Другой worker уже использует GPU"})
                    return
                result = self.server.run_evaluation(limit)
            self.reply(200, result)
        except Exception as error:
            traceback.print_exc()
            self.reply(500, {"error": str(error)})
        finally:
            self.server.busy.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))
    with ModelServer(args.model, args.port) as server:
        server.serve_forever()
