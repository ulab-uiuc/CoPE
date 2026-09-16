#!/usr/bin/env bash
# Check whether the hosted user simulator (gpt-4o-mini) is actually reachable and
# usable from this host, layer by layer, so a failure points at one thing.
#
#   bash scripts/check_openai.sh                 # run here (login node)
#   srun --nodes=1 -w <compute-node> --overlap bash scripts/check_openai.sh
#
# Egress and the API key are separate failures and the layers below separate them.
# Nothing here prints the key.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KEY_FILE="${KEY_FILE:-${ROOT}/.secrets/openai_api_key}"
MODEL="${MODEL:-gpt-4o-mini}"
PY=${TAU2_ENV_DEFAULT}/bin/python

pass() { printf '  \033[32mPASS\033[0m  %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; }
info() { printf '  ....  %s\n' "$1"; }

echo "host    : $(hostname)"
echo "proxy   : ${HTTPS_PROXY:-<unset>}"
echo "key file: ${KEY_FILE}"
echo

# ---- 0. key present -----------------------------------------------------------------
echo "[0] API key file"
if [[ -r "${KEY_FILE}" ]]; then
  K=$(tr -d '\r\n' < "${KEY_FILE}")
  pass "readable, ${#K} chars, prefix ${K:0:7}..., perms $(stat -c %a "${KEY_FILE}")"
else
  fail "not readable at ${KEY_FILE}"; exit 1
fi
echo

# ---- 1. egress ----------------------------------------------------------------------
echo "[1] Reaching api.openai.com (TLS + proxy)"
CODE=$(curl -sS --max-time 25 -o /dev/null -w '%{http_code}' https://api.openai.com/v1/models 2>/tmp/_curl_err)
if [[ "${CODE}" == "401" || "${CODE}" == "200" ]]; then
  pass "reachable (HTTP ${CODE} without auth is expected)"
elif [[ "${CODE}" == "000" ]]; then
  fail "no route: $(tr -d '\n' < /tmp/_curl_err | tail -c 120)"
  info "the proxy is refusing CONNECT, or this node has no egress at all"
else
  fail "HTTP ${CODE}"
fi
echo

# ---- 2. key is valid ----------------------------------------------------------------
echo "[2] Authenticating"
RESP=$(curl -sS --max-time 30 https://api.openai.com/v1/models \
        -H "Authorization: Bearer ${K}" 2>&1)
if grep -q '"object": *"list"' <<<"${RESP}"; then
  pass "key accepted"
  grep -q "\"${MODEL}\"" <<<"${RESP}" && pass "${MODEL} visible to this key" \
                                      || info "${MODEL} not in /models (often still callable)"
else
  fail "$(head -c 220 <<<"${RESP}" | tr -d '\n')"
fi
echo

# ---- 3. a real completion -----------------------------------------------------------
echo "[3] chat.completions on ${MODEL}"
RESP=$(curl -sS --max-time 45 https://api.openai.com/v1/chat/completions \
        -H "Authorization: Bearer ${K}" -H 'Content-Type: application/json' \
        -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: OK\"}],\"max_tokens\":5}" 2>&1)
if grep -q '"content"' <<<"${RESP}"; then
  pass "reply: $(sed -n 's/.*"content": *"\([^"]*\)".*/\1/p' <<<"${RESP}" | head -1)"
else
  fail "$(head -c 220 <<<"${RESP}" | tr -d '\n')"
fi
echo

# ---- 4. tool calling ----------------------------------------------------------------
# telecom's user simulator is handed tools and called with tool_choice="auto"; this is
# the capability whose absence silently killed every telecom episode at turn 2.
echo "[4] tool calling (required by the telecom user simulator)"
RESP=$(curl -sS --max-time 45 https://api.openai.com/v1/chat/completions \
        -H "Authorization: Bearer ${K}" -H 'Content-Type: application/json' \
        -d "{\"model\":\"${MODEL}\",\"tool_choice\":\"auto\",
             \"messages\":[{\"role\":\"user\",\"content\":\"Check my status bar.\"}],
             \"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"check_status_bar\",
               \"description\":\"Read the phone status bar\",
               \"parameters\":{\"type\":\"object\",\"properties\":{}}}}]}" 2>&1)
grep -q 'tool_calls' <<<"${RESP}" && pass "model emitted a tool_call" \
                                  || fail "no tool_call: $(head -c 200 <<<"${RESP}" | tr -d '\n')"
echo

# ---- 5. the path tau2 actually uses -------------------------------------------------
# tau2 does not call OpenAI directly; it goes through litellm, which has its own proxy
# and retry handling. Layers 1-4 can pass while this one fails.
echo "[5] litellm (what tau2 actually calls)"
TAU2_USER_API_KEY_FILE="${KEY_FILE}" MODEL="${MODEL}" "${PY}" - <<'PYEOF' 2>&1 | sed 's/^/  /'
import os, sys
key = open(os.environ["TAU2_USER_API_KEY_FILE"]).read().strip()
try:
    import litellm
    litellm.suppress_debug_info = True
    r = litellm.completion(
        model=f"openai/{os.environ['MODEL']}",
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        api_key=key, max_tokens=5, timeout=45,
    )
    print(f"\033[32mPASS\033[0m  litellm reply: {r.choices[0].message.content!r}")
except Exception as e:
    print(f"\033[31mFAIL\033[0m  {type(e).__name__}: {str(e)[:240]}")
    sys.exit(1)
PYEOF
echo
echo "All five must pass before a hosted-user-simulator run is worth launching."
