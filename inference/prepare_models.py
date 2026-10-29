"""Однократная подготовка точных base-моделей, без обучения."""
import json
import shutil
from importlib.metadata import version
from pathlib import Path

from huggingface_hub import snapshot_download
from mlx_lm.convert import convert


MODELS = [
    ("mamba", "tiiuae/Falcon3-Mamba-7B-Base",
     "89b36f48111b7fa85858189eb5fdd8e9c9d1ed79", True),
    ("qwen", "jesusoctavioas/Qwen3-8B-Base-mlx-4Bit",
     "47424c719f76443bf0abbf3bc441d32ff2d3a48b", False),
]


def prepare(name, repo, revision, needs_conversion):
    target = Path(__file__).resolve().parent / f"models/{name}-base-4bit"
    target.parent.mkdir(exist_ok=True)
    if (target / "provenance.json").exists():
        print(f"Уже подготовлена: {target}")
        return
    free_gb = shutil.disk_usage(target.parent).free / 1024**3
    if free_gb < (25 if needs_conversion else 7):
        raise RuntimeError(f"Недостаточно места на диске: {free_gb:.1f} ГБ")
    source = snapshot_download(
        repo, revision=revision,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
        max_workers=3,
    )
    staging = target.with_name(target.name + "-preparing")
    if staging.exists():
        shutil.rmtree(staging)
    if needs_conversion:
        convert(source, str(staging), quantize=True, q_bits=4, q_group_size=64)
    else:
        staging.mkdir()
        for file in Path(source).iterdir():
            if file.is_file():
                # Жёсткие ссылки не занимают второй размер весов на диске.
                staging.joinpath(file.name).hardlink_to(file.resolve())
    config = json.loads((staging / "config.json").read_text())
    quant = config["quantization"]
    if quant["bits"] != 4 or quant["group_size"] != 64:
        raise ValueError(f"Неожиданное квантование: {quant}")
    original = "tiiuae/Falcon3-Mamba-7B-Base" if name == "mamba" else "Qwen/Qwen3-8B-Base"
    provenance = {"original": original, "repository": repo, "revision": revision,
                  "original_revision": revision if name == "mamba" else "49e3418fbbbca6ecbdf9608b4d22e5a407081db4",
                  "conversion_mlx_lm": version("mlx-lm") if needs_conversion else "0.28.3",
                  "bits": 4, "group_size": 64, "mode": quant.get("mode", "affine")}
    (staging / "provenance.json").write_text(json.dumps(provenance, indent=2))
    for document in staging.glob("*.md"):
        document.unlink()
    staging.rename(target)
    print(f"Подготовлена: {target}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["mamba", "qwen"])
    args = parser.parse_args()
    for model in MODELS:
        if args.only is None or args.only == model[0]:
            prepare(*model)
