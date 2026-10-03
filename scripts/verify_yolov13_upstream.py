#!/usr/bin/env python3
"""Fail fast when a generic Ultralytics build shadows the official YOLOv13 fork."""
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ultralytics
from ultralytics.nn import modules

EXPECTED_COMMIT = "70f23ede45ee00a30cf6139c3d1ea7abe3df4eec"
required = ["DSC3k2", "DSConv", "A2C2f", "HyperACE", "FullPAD_Tunnel", "DownsampleConv"]
missing = [x for x in required if not hasattr(modules, x)]
if missing:
    # Some fork versions expose modules through submodules rather than __init__.
    try:
        from ultralytics.nn.modules import block, conv
        missing = [x for x in required if not hasattr(block, x) and not hasattr(conv, x) and not hasattr(modules, x)]
    except Exception:
        pass
if missing:
    raise SystemExit(
        "Wrong Ultralytics package: missing YOLOv13 modules " + ", ".join(missing) +
        ". Remove the generic package and run scripts/bootstrap_yolov13.sh."
    )
package_path = Path(ultralytics.__file__).resolve()
repository = package_path.parent.parent
try:
    actual_commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
except (OSError, subprocess.CalledProcessError) as exc:
    raise SystemExit(
        "YOLOv13 source is not an auditable Git checkout. Run scripts/bootstrap_yolov13.sh."
    ) from exc
if actual_commit != EXPECTED_COMMIT:
    raise SystemExit(
        f"Wrong YOLOv13 revision: found {actual_commit}, expected {EXPECTED_COMMIT}. "
        "Run scripts/bootstrap_yolov13.sh."
    )
print(f"YOLOv13 upstream OK: {package_path} @ {actual_commit}")
