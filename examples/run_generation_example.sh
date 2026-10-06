#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <audio-file> [output.osu]" >&2
  exit 2
fi

audio=$1
output=${2:-output/generated_oni.osu}
mkdir -p "$(dirname "$output")"

uv run taikonation-generate "$audio" "$output" --difficulty oni
echo "Wrote $output"
