#!/bin/sh
set -eu
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$project_dir"
config=${1:-config.yaml}
if [ "$#" -gt 0 ]; then
    shift
fi
exec "$project_dir/.venv-pi/bin/python" "$project_dir/video_monitor.py" --config "$config" "$@"
