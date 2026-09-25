#!/usr/bin/env bash
# A/B-замер декода: две сборки qwfn-gen на одном промпте, жадный декод.
# Сравнивает скорость и побитовое совпадение сгенерированных токенов.
#
#   MODEL=/путь/к/...-00001-of-0000N.gguf scripts/ab_decode.sh <gen_A> <gen_B> [флаги qwfn-gen...]
#
#   RUNS=3      сколько чередующихся пар прогонов
#   GEN=128     сколько токенов генерировать
#   OUT=dir     куда класть логи (по умолчанию ./ab-out)
#
# Пример (оригинал против этой ветки, бэкенд io_uring):
#   MODEL=... scripts/ab_decode.sh ../clean/build/qwfn-gen build/qwfn-gen --io-uring
set -u
cd "$(dirname "$0")/.."
A=${1:?gen_A}; B=${2:?gen_B}; shift 2
MODEL=${MODEL:?укажите MODEL=первый шард .gguf}
RUNS=${RUNS:-3}; GEN=${GEN:-128}; OUT=${OUT:-ab-out}
mkdir -p "$OUT"
PROMPT="$OUT/prompt.txt"
if [ ! -s "$PROMPT" ]; then
    ./build/qwfn-tok "$MODEL" --chat "Explain in three sentences why mixture-of-experts models are cheaper to run than dense models with the same parameter count." --think off > "$PROMPT" 2>/dev/null \
        || { echo "не удалось токенизировать промпт (соберите build/qwfn-tok)"; exit 1; }
fi

run() {  # label bin [flags...]
    local label=$1 bin=$2; shift 2
    local log="$OUT/run-$label.log"
    QWFN_IO_PROFILE=1 "$bin" "$MODEL" --prompt-file "$PROMPT" --gen "$GEN" --ram 12 "$@" > "$log" 2>&1
    local rc=$?
    local tps; tps=$(grep -oE '^decode: .*\(([0-9.]+) tok/s' "$log" | grep -oE '[0-9.]+ tok/s' | cut -d' ' -f1)
    local wait; wait=$(grep -oE 'total begin [0-9.]+ ms, end [0-9.]+ ms' "$log" | grep -oE 'end [0-9.]+' | cut -d' ' -f2)
    sed -n '/^generated:/,/^$/p' "$log" | tr -s ' \n' ' ' | md5sum | cut -c1-12 > "$OUT/ids-$label"
    printf '%-6s rc=%s  %6s tok/s  ожидание чтений %6s мс/ток  токены %s\n' "$label" "$rc" "${tps:-?}" "${wait:-?}" "$(cat "$OUT/ids-$label")"
    echo "${tps:-0}" >> "$OUT/tps-${label%%-*}"
}
rm -f "$OUT"/tps-* "$OUT"/ids-*
for i in $(seq 1 "$RUNS"); do
    run "A-$i" "$A" "$@"
    run "B-$i" "$B" "$@"
done
avg() { awk '{s+=$1} END {if (NR) printf "%.2f", s/NR}' "$1"; }
echo "среднее: A $(avg "$OUT/tps-A") tok/s, B $(avg "$OUT/tps-B") tok/s"
if [ "$(cat "$OUT"/ids-* | sort -u | wc -l)" = 1 ]; then echo "токены: во всех прогонах побитово одинаковые"
else echo "токены: различаются (ожидаемо с --vram: эксперты из VRAM и RAM считаются в разном порядке)"; fi
