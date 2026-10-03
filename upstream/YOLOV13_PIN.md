# Pinned detector dependency

SemTraTrack uses the **official iMoonLab YOLOv13 repository**:

- Repository: `https://github.com/iMoonLab/yolov13`
- Pinned commit: `70f23ede45ee00a30cf6139c3d1ea7abe3df4eec`
- Upstream license: AGPL-3.0
- Paper architecture used here: official YOLOv13-S graph (`scale: s`) with `nc: 1`

This commit is the object referenced by the upstream `yolov13` tag when the
release package was finalized. The dependency is installed by
`scripts/bootstrap_yolov13.sh`. It is kept as an
external upstream dependency rather than copied into this archive, so its
original history/license remain unmodified and a generic PyPI `ultralytics`
package cannot silently replace the YOLOv13 implementation.

After installation, `scripts/verify_yolov13_upstream.py` checks the presence of
the YOLOv13-specific DSC3k2/HyperACE/FullPAD modules before training.
