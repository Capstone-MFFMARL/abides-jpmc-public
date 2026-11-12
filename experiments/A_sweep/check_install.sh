#!/usr/bin/env bash
# "All systems go" check for Experiment A.
# Run from anywhere inside the repo, using the active Python environment:
#     bash experiments/A_sweep/check_install.sh

set -e
ROOT="$(git rev-parse --show-toplevel)"
echo "branch: $(git -C "$ROOT" rev-parse --abbrev-ref HEAD)"
echo "commit: $(git -C "$ROOT" rev-parse --short HEAD)"
echo "python: $(which python || which python3)"
python --version

echo ""
echo "=== 1/3 rmsc_density imports from THIS checkout ==="
python - <<PY
import abides_markets.configs.rmsc_density as m, os
print("loaded from:", m.__file__)
assert os.path.realpath(m.__file__).lower().startswith(os.path.realpath("$ROOT").lower()), \
    f"rmsc_density is loaded from somewhere else (not this repo) - re-run: pip install -e abides-core -e abides-markets"
PY

echo ""
echo "=== 2/3 short simulation (num_mm=2, 5 simulated minutes) ==="
TMP="$ROOT/experiments/A_sweep/.check_temp.csv"
python "$ROOT/experiments/A_sweep/run_sweep.py" --num-mm 2 --seeds 1 --end-time 09:35:00 --no-snapshots --out "$TMP"
rm -f "$TMP"

echo ""
echo "=== 3/3 Kyle tests (expect: 5 passed) ==="
(cd "$ROOT/experiments/B_kyle" && pytest -q)

echo ""
echo "ALL CHECKS PASSED"
