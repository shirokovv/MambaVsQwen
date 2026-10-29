#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export DOCKER_CONTEXT=colima-mas-local
status=0

if command -v docker >/dev/null && docker info >/dev/null 2>&1; then
    docker compose -f app/compose.yaml down || status=1
else
    echo "Colima недоступна: состояние контейнера проверить не удалось." >&2
    status=1
fi

for name in mamba qwen; do
    file="inference/logs/$name.pid"
    [[ -f "$file" ]] || continue
    pid=$(cat "$file")
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
        if ! ps -p "$pid" -o command= | grep -Fq "$ROOT/inference/server.py"; then
            echo "PID $pid принадлежит другому процессу; остановка пропущена." >&2
            status=1
            continue
        fi
        kill -TERM "$pid" 2>/dev/null || true
        for ((attempt=0; attempt<30; attempt++)); do
            kill -0 "$pid" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$pid" 2>/dev/null; then
            echo "$name ещё завершается; PID сохранён." >&2
            status=1
            continue
        fi
    fi
    rm -f "$file"
    echo "$name остановлена."
done
if [[ "$status" == 0 ]]; then echo "Приложение и модели остановлены. Данные сохранены."; fi
exit "$status"
