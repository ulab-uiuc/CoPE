#!/usr/bin/env bash
#
# Apply the one upstream fix a tau2-bench checkout needs before anything here runs.
# Idempotent: safe to call on every launch.
#
#   bash scripts/patch_tau2_bench.sh [path/to/tau2-bench]
#
# tau2's `to_litellm_messages` emits a top-level "name" on each tool call alongside
# function.name. Current vLLM validates tool_calls strictly and rejects the duplicate
# key as `extra_forbidden`, so every assistant turn after a tool call fails, each task
# ends as `infrastructure_error` with zero messages, and the run scores 0% -- which
# reads as a terrible policy rather than a schema mismatch. This only matters for
# evaluation through tau2's own CLI (scripts/sbatch_tau2_align.sh); the training
# harness never routes the policy through litellm.

set -euo pipefail

TAU2_BENCH_DIR="${1:-${TAU2_BENCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/tau2-bench}}"
F="${TAU2_BENCH_DIR}/src/tau2/utils/llm_utils.py"

[[ -f "${F}" ]] || { echo "patch_tau2_bench: ${F} not found (is tau2-bench cloned at ${TAU2_BENCH_DIR}?)" >&2; exit 1; }

WANT=c5b2d22
HAVE="$(git -C "${TAU2_BENCH_DIR}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
if [[ "${HAVE}" != "${WANT}"* ]]; then
  echo "patch_tau2_bench: WARNING tau2-bench is at ${HAVE}, not ${WANT} -- numbers will not be" \
       "comparable to docs/TAU2_GRPO.md (see 'Comparing against published numbers')." >&2
fi

# The duplicate is the "name" that directly follows "id": tc.id inside the tool_calls
# dict; the one under "function": {...} is the correct field and must stay, so a plain
# grep for '"name": tc.name' would misreport an already-patched tree as unpatched.
if grep -Pzoq '"id": tc\.id,\n\s*"name": tc\.name,' "${F}"; then
  python3 - "${F}" <<'PY'
import io, sys, re
p = sys.argv[1]
s = io.open(p, encoding="utf-8").read()
# Drop only the top-level duplicate inside the tool_calls dict; function.name stays.
new, n = re.subn(r'(\{\s*\n\s*"id": tc\.id,\n)\s*"name": tc\.name,\n', r"\1", s, count=1)
assert n == 1, "expected exactly one top-level 'name': tc.name to remove"
io.open(p, "w", encoding="utf-8").write(new)
PY
  echo "patch_tau2_bench: removed duplicate top-level tool_call 'name' in ${F}"
else
  echo "patch_tau2_bench: already applied"
fi
