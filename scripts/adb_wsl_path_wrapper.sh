#!/usr/bin/env bash

# Invoke a Windows ADB binary from WSL while translating only local WSL
# mount paths.  The profiling script deliberately keeps its host-side paths
# in WSL form for sha256sum, QAIRT tools, and result files; Windows adb needs
# those same paths in drive-letter form for `push`.

set -Eeuo pipefail

ADB_EXE="${ADB_EXE:-/mnt/c/adb/adb.exe}"
[[ -x "${ADB_EXE}" ]] || {
    echo "ERROR: Windows ADB executable not found: ${ADB_EXE}" >&2
    exit 1
}

converted=()
for arg in "$@"; do
    if [[ "${arg}" == /mnt/* ]]; then
        arg="$(wslpath -w -- "${arg}")"
    fi
    converted+=("${arg}")
done

exec "${ADB_EXE}" "${converted[@]}"
