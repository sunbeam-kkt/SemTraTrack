# Third-party notice

SemTraTrack depends on the official **iMoonLab/YOLOv13** implementation, which is
released under AGPL-3.0. The pinned upstream repository is not re-licensed by
this package. `scripts/bootstrap_yolov13.sh` obtains the original source and
checks out commit `70f23ede45ee00a30cf6139c3d1ea7abe3df4eec`, then installs it
in editable mode so its upstream LICENSE remains intact.
