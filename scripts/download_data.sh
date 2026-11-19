#!/usr/bin/env bash
# NASA C-MAPSS (public domain). Official source: NASA Prognostics Data Repository
# (https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).
# Mirror used here for scriptable download:
set -euo pipefail
mkdir -p data/raw && cd data/raw
BASE=https://raw.githubusercontent.com/hankroark/Turbofan-Engine-Degradation/master/CMAPSSData
for s in FD001 FD002 FD003 FD004; do for f in train test RUL; do curl -sSfO "$BASE/${f}_${s}.txt"; done; done
ls -la
