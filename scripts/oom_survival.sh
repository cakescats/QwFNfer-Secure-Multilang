#!/usr/bin/env bash
# Выживает ли сервер при нехватке VRAM посреди запроса.
#   обычный запрос -> занять VRAM (qwfn-vram-hog) -> длинный запрос -> жив ли процесс -> два запроса после
#
#   MODEL=/путь/к/...-00001-of-0000N.gguf scripts/oom_survival.sh <qwfn-server> [метка]
#
# Требует build/qwfn-vram-hog (см. tools/qwfn_vram_hog.cu). Порт 18099, сервер запускается
# и останавливается самим скриптом; другой сервер на GPU в это время работать не должен.
# Переменная GGML_CUDA_DISABLE_GRAPHS=1 передаётся серверу, если задана.
set -u
cd "$(dirname "$0")/.."
BIN=${1:?qwfn-server}; LABEL=${2:-run}
MODEL=${MODEL:?укажите MODEL=первый шард .gguf}
HOG=${HOG:-build/qwfn-vram-hog}; PORT=${PORT:-18099}; U=http://127.0.0.1:$PORT
[ -x "$HOG" ] || { echo "нет $HOG: nvcc -O2 -o $HOG tools/qwfn_vram_hog.cu"; exit 1; }
LOG=oom-$LABEL.log
"$BIN" "$MODEL" --port "$PORT" --ram 12 --vram 9 --ctx 32768 > "$LOG" 2>&1 & SP=$!
for _ in $(seq 1 180); do curl -sf "$U/health" >/dev/null && break; kill -0 $SP 2>/dev/null || break; sleep 1; done
curl -sf "$U/health" >/dev/null || { echo "[$LABEL] сервер не поднялся"; tail -5 "$LOG"; kill $SP 2>/dev/null; exit 1; }

ask() {
    python3 - "$U" "$1" <<'PY'
import json, sys, urllib.request, urllib.error
u, txt = sys.argv[1], sys.argv[2]
body = json.dumps({"model": "x", "max_tokens": 24, "messages": [{"role": "user", "content": txt + " /no_think"}]}).encode()
try:
    r = urllib.request.urlopen(urllib.request.Request(u + "/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=600)
    print(r.status, repr(json.load(r)["choices"][0]["message"]["content"][:60]))
except urllib.error.HTTPError as e: print(e.code, e.read()[:160])
except Exception as e: print("EXC", type(e).__name__, str(e)[:100])
PY
}
echo "[$LABEL] 1 обычный:        $(ask 'What is the capital of France? One word.')"
"$HOG" 40 & HP=$!; sleep 4
LONG=$(python3 -c "print(' '.join(open('src/qwfn_engine.cpp').read().split()[:9000]))")
echo "[$LABEL] 2 длинный + OOM:  $(ask "$LONG Summarise this code in one sentence.")"
echo "[$LABEL] процесс жив:      $(kill -0 $SP 2>/dev/null && echo да || echo НЕТ)"
kill $HP 2>/dev/null; wait $HP 2>/dev/null; sleep 2
echo "[$LABEL] 3 после:          $(ask 'What is the capital of France? One word.')"
echo "[$LABEL] 4 ещё раз:        $(ask 'What is 2+2? Answer with a digit.')"
grep -m3 -E 'CUDA error|in function|SIGABRT' "$LOG" | sed "s/^/[$LABEL] лог: /"
kill $SP 2>/dev/null; wait $SP 2>/dev/null; echo "[$LABEL] код выхода: $? (143 = остановлен скриптом, 134 = abort)"
