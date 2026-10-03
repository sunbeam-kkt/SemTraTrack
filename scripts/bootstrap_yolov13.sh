#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD="${ROOT}/third_party/yolov13"
UPSTREAM_COMMIT="70f23ede45ee00a30cf6139c3d1ea7abe3df4eec"
mkdir -p "${ROOT}/third_party"
if [[ ! -d "${THIRD}/.git" ]]; then
  git clone --filter=blob:none --no-checkout https://github.com/iMoonLab/yolov13.git "${THIRD}"
fi
git -C "${THIRD}" fetch --depth 1 origin "${UPSTREAM_COMMIT}"
# Deliberately omit --force: local changes in third_party/yolov13 must not be
# discarded silently.
git -C "${THIRD}" checkout --detach "${UPSTREAM_COMMIT}"
python -m pip install -e "${THIRD}"
python "${ROOT}/scripts/verify_yolov13_upstream.py"
