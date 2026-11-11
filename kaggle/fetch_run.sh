#!/usr/bin/env bash
# Usage: kaggle/fetch_run.sh <kernel-slug> [kaggle-username]
# Downloads results/, figures/ and the log of a finished kernel into runs/<slug>_<timestamp>/
set -euo pipefail
slug=$1; user=${2:-onethybeing}
dest="$(dirname "$0")/../runs/${slug}_$(date +%Y%m%d_%H%M)"
mkdir -p "$dest"
kaggle kernels output "$user/$slug" -p "$dest" --file-pattern '.*(results|figures)/.*|.*\.log$|.*\.ipynb$'
python - "$dest/$slug.log" <<'PY'
import json, sys
p = sys.argv[1]
try:
    txt = "".join(e.get("data", "") for e in json.load(open(p, encoding="utf-8")))
    open(p.replace(".log", "_stdout.txt"), "w", encoding="utf-8").write(txt)
except Exception as ex:
    print("log parse failed:", ex)
PY
echo "$dest"
