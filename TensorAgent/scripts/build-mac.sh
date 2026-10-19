#!/usr/bin/env bash
# Builds the TensorAgent Mac app: TensorAgent.Maui for Mac Catalyst.
#
# The Mac app is the phone app's code under Mac Catalyst with the desktop engine: the
# same libGgmlOps.dylib the desktop hosts load, which the build brings up to date before
# the app (TensorAgentBuildDesktopEngine in the csproj), and real python3, node and npm
# processes confined by Seatbelt instead of the phone's embedded interpreters.
#
# Needs the user-local ~/.dotnet SDK with the maui-maccatalyst workload (see README.md),
# and CMake with the Xcode command-line tools for the engine library.
#
# Env:
#   CONFIGURATION   Debug (default) | Release
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIGURATION="${CONFIGURATION:-Debug}"

export DOTNET_ROOT="${DOTNET_ROOT:-$HOME/.dotnet}"
export PATH="${DOTNET_ROOT}:${PATH}"
# The app's LTX executor uses the packaged MLX runtime. Skip duplicate referenced-project
# builds here; TensorAgentBuildMlxDesktopEngine handles the app's runtime separately.
# The GGML native build stays enabled so the desktop engine matches its exports.
export TENSORSHARP_MLX_NATIVE_SKIP=true

echo "==> dotnet $(dotnet --version): building TensorAgent.Maui (${CONFIGURATION}, Mac Catalyst)"
# TensorSharpAppleTargets=true must be on the command line for the same reason as in
# build-sim.sh: restore resolves TensorSharp.Models' target frameworks before a
# ProjectReference's AdditionalProperties apply. Single-node for the same reason too.
dotnet build "${REPO_ROOT}/TensorAgent/src/TensorAgent.Maui/TensorAgent.Maui.csproj" \
    -f net10.0-maccatalyst \
    -c "${CONFIGURATION}" \
    -p:TensorSharpAppleTargets=true \
    -m:1 \
    -nologo

APP="$(ls -d "${REPO_ROOT}/TensorAgent/src/TensorAgent.Maui/bin/${CONFIGURATION}/net10.0-maccatalyst/"*/TensorAgent.app 2>/dev/null | head -1)"
if [[ -z "${APP}" || ! -d "${APP}" ]]; then
    echo "Build finished but no TensorAgent.app is under bin/${CONFIGURATION}/net10.0-maccatalyst/" >&2
    exit 1
fi
echo "App: ${APP}"

# The other configuration's app shares this one's data but not its code, and nothing else
# says when it falls behind: a Release app built before five catalog entries were added
# was opened a day later and listed none of them. It is behind when one of its inputs
# changed after it was built: a file of TensorAgent.Maui's project closure, a bundled
# skill, a plug-in config or the native engine's sources. Its build time is its signature,
# which every build rewrites; an assembly keeps its compile time when nothing in it
# changed, so comparing assemblies flagged an app built a minute earlier from the same
# sources, and changes that compile into no assembly went unnoticed.
app_projects() {
    # TensorAgent.Maui and every project it references, transitively, one directory a line.
    local todo="${REPO_ROOT}/TensorAgent/src/TensorAgent.Maui/TensorAgent.Maui.csproj" visited="" proj dir ref
    while [[ -n "${todo}" ]]; do
        proj="${todo%%$'\n'*}"
        if [[ "${todo}" == *$'\n'* ]]; then todo="${todo#*$'\n'}"; else todo=""; fi
        case $'\n'"${visited}" in *$'\n'"${proj}"$'\n'*) continue ;; esac
        visited+="${proj}"$'\n'
        dir="$(dirname "${proj}")"
        echo "${dir}"
        for ref in $(grep -o '<ProjectReference Include="[^"]*"' "${proj}" | sed -E 's/.*Include="([^"]*)"/\1/; s#\\#/#g'); do
            [[ -f "${dir}/${ref}" ]] || continue
            todo+="${todo:+$'\n'}$(cd "$(dirname "${dir}/${ref}")" && pwd)/$(basename "${ref}")"
        done
    done
}
changed_input() {
    # The first input newer than $1, or nothing.
    local built="$1" dir
    {
        app_projects | while IFS= read -r dir; do
            find "${dir}" \( -name bin -o -name obj -o -name 'build*' -o -name dist -o -name tests \
                -o -name node_modules -o -name __pycache__ -o -name '.*' \) -prune \
                -o -type f -newer "${built}" ! -name '*.md' -print 2>/dev/null
        done
        find "${REPO_ROOT}/TensorAgent/skills" \( -name node_modules -o -name __pycache__ -o -name '.*' \) -prune \
            -o -type f -newer "${built}" ! -name verdicts.json -print 2>/dev/null
        find "${REPO_ROOT}/config/lora" -type f -newer "${built}" -print 2>/dev/null
        find "${REPO_ROOT}/TensorSharp.GGML.Native" \( -name 'build*' -o -name tests -o -name '.*' \) -prune \
            -o -type f -newer "${built}" \( -name '*.c' -o -name '*.cpp' -o -name '*.h' -o -name '*.m' -o -name '*.mm' \
            -o -name '*.metal' -o -name '*.inc' -o -name '*.cmake' -o -name CMakeLists.txt \) -print 2>/dev/null
        find "${REPO_ROOT}" -maxdepth 1 -type f -name 'Directory.*' -newer "${built}" -print 2>/dev/null
    } | head -1
}
for OTHER_CONFIGURATION in Debug Release; do
    [[ "${OTHER_CONFIGURATION}" == "${CONFIGURATION}" ]] && continue
    # Never built: ls fails, which must not fail a build that succeeded.
    OTHER="$(ls -d "${REPO_ROOT}/TensorAgent/src/TensorAgent.Maui/bin/${OTHER_CONFIGURATION}/net10.0-maccatalyst/"*/TensorAgent.app 2>/dev/null | head -1)" || true
    [[ -n "${OTHER}" ]] || continue
    BUILT="${OTHER}/Contents/_CodeSignature/CodeResources"
    [[ -f "${BUILT}" ]] || BUILT="$(ls -t "${OTHER}"/Contents/MonoBundle/*.dll 2>/dev/null | head -1)" || true
    [[ -n "${BUILT}" ]] || continue
    CHANGED="$(changed_input "${BUILT}")" || true
    if [[ -n "${CHANGED}" ]]; then
        echo "Note: the ${OTHER_CONFIGURATION} app (${OTHER}) was built before ${CHANGED#"${REPO_ROOT}"/} changed;" \
             "rebuild it with CONFIGURATION=${OTHER_CONFIGURATION} before opening it." >&2
    fi
done
