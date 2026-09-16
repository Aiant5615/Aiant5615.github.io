#!/usr/bin/env bash
# Runs every implementation and reports pass/fail and wall time. Usage: bash code/run_all.sh [python]
PY=${1:-python3}; cd "$(dirname "$0")"; fail=0
for f in */*.py; do
  s=$(date +%s); if out=$($PY "$f" 2>&1); then st=ok; else st=FAIL; fail=$((fail+1)); fi
  printf '%-45s %-4s %3ss\n' "$f" "$st" "$(( $(date +%s) - s ))"; [ "$st" = FAIL ] && printf '%s\n' "$out" | tail -5
done; echo "failures: $fail"; exit $fail
