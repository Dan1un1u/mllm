#!/usr/bin/env bash

# Source this file from WSL to route command-line downloads through Clash Verge
# running on Windows. Override either value before sourcing when needed.
if [[ -z "${CLASH_PROXY_HOST:-}" ]]; then
    CLASH_PROXY_HOST="$(ip route show default | awk 'NR == 1 { print $3 }')"
fi
CLASH_PROXY_PORT="${CLASH_PROXY_PORT:-7897}"

if [[ -z "${CLASH_PROXY_HOST}" ]]; then
    echo "Unable to discover the Windows host from the WSL default route." >&2
    return 1 2>/dev/null || exit 1
fi

_mllm_proxy_url="http://${CLASH_PROXY_HOST}:${CLASH_PROXY_PORT}"
export HTTP_PROXY="${_mllm_proxy_url}"
export HTTPS_PROXY="${_mllm_proxy_url}"
export ALL_PROXY="${_mllm_proxy_url}"
export http_proxy="${_mllm_proxy_url}"
export https_proxy="${_mllm_proxy_url}"
export all_proxy="${_mllm_proxy_url}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,::1}"
export no_proxy="${no_proxy:-${NO_PROXY}}"

echo "WSL proxy: ${_mllm_proxy_url}"
unset _mllm_proxy_url
