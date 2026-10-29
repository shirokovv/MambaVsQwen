#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export DOCKER_CONTEXT=colima-mas-local

if [[ "${1:-}" == --help ]]; then
    echo "./start.sh — подготовить зависимости и запустить модели и приложение."
    echo "./stop.sh  — остановить модели и приложение."
    exit 0
fi
[[ $# == 0 ]] || { echo "Использование: ./start.sh" >&2; exit 1; }
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || {
    echo "Для MLX требуется macOS на Apple Silicon." >&2; exit 1;
}
command -v python3 >/dev/null || { echo "Установите Python 3.13." >&2; exit 1; }
command -v docker >/dev/null || { echo "Установите Docker CLI и Colima с Docker Compose." >&2; exit 1; }
if ! docker info >/dev/null 2>&1; then
    echo "Сначала запустите Colima: colima start --profile mas-local" >&2; exit 1
fi

mkdir -p inference/logs data
mkdir inference/logs/start.lock 2>/dev/null || { echo "Запуск уже выполняется." >&2; exit 1; }
started_ok=0
finish() {
    status=$?
    rmdir "$ROOT/inference/logs/start.lock"
    if [[ "$status" != 0 || "$started_ok" != 1 ]]; then
        [[ "$status" != 0 ]] || status=1
        echo "Запуск прерван. Останавливаю сервисы." >&2
        "$ROOT/stop.sh" || true
    fi
    exit "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

export HF_HOME="$ROOT/data/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export XDG_CACHE_HOME="$ROOT/data/cache"
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1

if [[ ! -x inference/.venv/bin/python ]]; then
    echo "Создаю окружение для инференса…"
    python3 -m venv inference/.venv
fi
PYTHON="$ROOT/inference/.venv/bin/python"
if ! cmp -s inference/requirements.txt inference/.venv/requirements.txt; then
    echo "Устанавливаю зависимости инференса…"
    "$PYTHON" -m pip install --disable-pip-version-check -r inference/requirements.txt
    cp inference/requirements.txt inference/.venv/requirements.txt
fi
if [[ ! -f data/dataset.json ]]; then
    echo "Подготавливаю LAMBADA…"
    "$PYTHON" inference/prepare_data.py
fi
for name in mamba qwen; do
    if [[ ! -f "inference/models/$name-base-4bit/provenance.json" ]]; then
        echo "Подготавливаю $name: первая загрузка может занять длительное время…"
        "$PYTHON" inference/prepare_models.py --only "$name"
    fi
done

export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
for name in mamba qwen; do
    port=8001
    [[ "$name" != qwen ]] || port=8002
    pid_file="inference/logs/$name.pid"
    if [[ -f "$pid_file" ]]; then
        pid=$(cat "$pid_file")
        if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null &&
            ps -p "$pid" -o command= | grep -Fq "$ROOT/inference/server.py"; then
            echo "$name уже запущен (PID $pid)."
            continue
        fi
        rm -f "$pid_file"
    fi
    if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null; then
        echo "Порт $port занят другим процессом." >&2; exit 1
    fi
    echo "Загружаю ${name}…"
    nohup "$PYTHON" -c 'import os, sys; os.setsid(); os.execv(sys.executable, [sys.executable] + sys.argv[1:])' "$ROOT/inference/server.py" --model "$ROOT/inference/models/$name-base-4bit" --port "$port" > "inference/logs/$name.log" 2>&1 < /dev/null &
    pid=$!
    echo "$pid" > "$pid_file"
    ready=0
    for ((attempt=0; attempt<300; attempt++)); do
        if grep -q '^READY ' "inference/logs/$name.log" && curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
            ready=1; break
        fi
        kill -0 "$pid" 2>/dev/null || { tail -20 "inference/logs/$name.log"; exit 1; }
        sleep 2
    done
    [[ "$ready" == 1 ]] || { echo "$name не загрузилась; см. inference/logs/$name.log" >&2; exit 1; }
    echo "$name готова: http://127.0.0.1:$port"
done

docker compose -f app/compose.yaml up -d --build --wait --wait-timeout 120
docker compose -f app/compose.yaml exec -T app python -c 'from pathlib import Path; assert Path("/data/dataset.json").is_file(), "Настройте общий каталог проекта в Colima: контейнер не видит data."'
echo "Готово: http://localhost:8501 — остановка: ./stop.sh"
started_ok=1
