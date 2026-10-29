import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ModelClient:
    def __init__(self, name, port):
        self.name = name
        host = os.getenv("MODEL_HOST", "127.0.0.1")
        self.url = f"http://{host}:{port}"

    def health(self):
        try:
            with urlopen(self.url + "/health", timeout=3) as response:
                return json.load(response)
        except (URLError, TimeoutError, OSError) as error:
            return {"status": "unavailable", "error": str(error)}

    def evaluate(self, limit):
        request = Request(self.url + "/evaluate", json.dumps({"limit": limit}).encode(),
                          {"Content-Type": "application/json"})
        # Полный test split может выполняться часами. Быстрого read timeout нет.
        try:
            with urlopen(request, timeout=None) as response:
                return json.load(response)
        except HTTPError as error:
            raise RuntimeError(json.load(error).get("error", str(error))) from error


class MambaClient(ModelClient):
    def __init__(self):
        super().__init__("Falcon3-Mamba-7B-Base", 8001)


class QwenClient(ModelClient):
    def __init__(self):
        super().__init__("Qwen3-8B-Base", 8002)
