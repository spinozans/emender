#!/usr/bin/env bash
set -euo pipefail
umask 077
if [[ $# != 2 ]]; then echo 'usage: capture_e97_pi_tool_surface.sh PI_BIN OUTPUT_DIR' >&2; exit 2; fi
PI_BIN=$(realpath "$1"); OUT=$2
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CAPTURE="$ROOT/configs/pi/e97-capture-pi-tool-surface.ts"
[[ -x "$PI_BIN" && -f "$CAPTURE" && ! -e "$OUT" ]]
mkdir -m 700 "$OUT"
FFF=/home/erikg/.pi/agent/npm/node_modules/@ff-labs/pi-fff/src/index.ts
PROCESS=/home/erikg/.pi/agent/npm/node_modules/@aliou/pi-processes/extensions/processes/index.ts
WEB=/home/erikg/.pi/agent/npm/node_modules/pi-web-access/index.ts
for file in "$FFF" "$PROCESS" "$WEB"; do [[ -f "$file" ]]; done

capture() {
 local name=$1 tools=$2; shift 2
 local dir="$OUT/$name"; mkdir -p -m 700 "$dir/home/.pi/agent" "$dir/cwd" "$dir/sessions"
 printf '%s\n' '{"compaction":{"enabled":false},"retry":{"enabled":false},"defaultTools":[],"packages":[],"extensions":[],"enableInstallTelemetry":false,"enableAnalytics":false,"defaultProjectTrust":"never","quietStartup":true}' > "$dir/home/.pi/agent/settings.json"
 (
  cd "$dir/cwd"
  export E97_PI_TOOL_SURFACE_OUTPUT="$dir/tools.json" E97_PI_TOOL_SURFACE_TRACE="$dir/trace.log"
  export HOME="$dir/home" PI_CODING_AGENT_DIR="$dir/home/.pi/agent" PI_OFFLINE=1 PI_SKIP_VERSION_CHECK=1 NO_COLOR=1 TERM=dumb
  export PI_FFF_MODE=tools-only FFF_ENABLE_HOME_SCAN=0 FFF_ENABLE_ROOT_SCAN=0
  local args=(--offline --no-approve --no-extensions --no-skills --no-prompt-templates --no-themes --no-context-files)
  if [[ $name != core ]]; then args+=(--no-builtin-tools); fi
  args+=(-e "$CAPTURE")
  while (($#)); do args+=(-e "$1"); shift; done
  args+=(--provider e97-tool-surface-capture --model capture-only --tools "$tools" --system-prompt capture --session-dir "$dir/sessions" --mode json -p -- E97_CAPTURE_ACTIVE_PI_TOOL_SURFACE_V1)
  printf '%q ' "$PI_BIN" "${args[@]}" > "$dir/command.txt"; printf '\n' >> "$dir/command.txt"
  timeout --kill-after=5 30 "$PI_BIN" "${args[@]}" < /dev/null > "$dir/pi-events.jsonl" 2> "$dir/pi-stderr.log"
 )
 [[ -s "$dir/tools.json" ]] && grep -q 'stream-invoked' "$dir/trace.log" && grep -q '"type":"message_end"' "$dir/pi-events.jsonl"
}

capture core 'read,bash,edit,write'
capture process 'process' "$PROCESS"
capture fff 'fffind,ffgrep' "$FFF"
capture web 'web_search,source_check,fetch_content,get_search_content' "$WEB"
printf '%s\n' 'PI_TOOL_SURFACE_GROUP_CAPTURE_PASSED core=4 process=1 fff=2 web=4'
