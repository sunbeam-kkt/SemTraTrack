# LINR and generic-SOT reproduction note

Checked on 2026-10-03.

## What LINR is

LINR (Chen et al., IEEE TCSVT 2025, DOI
`10.1109/TCSVT.2025.3578667`) is a plug-and-play local implicit neural
representation module inserted into existing one-stream SOT trackers. The
paper evaluates OSTrack-LINR, AQATrack-LINR, and FERMT-LINR. It is not a
standalone detector or a detachable "LINR localization branch."

## Public material currently available

The official repository is <https://github.com/Xiaochen918/LINR>. At commit
`2bdbea2828c47ea98e904db84afbc4c279fc7177`, it contains only `README.md`,
`LICENSE`, and a link to raw results. The README states that the source code is
still being organized for release. Consequently, an exact LINR module
reproduction is not currently possible from that repository alone.

## Why the search factor is 4.0

The manuscript's LINR row is AQATrack-LINR. The official AQATrack repository
<https://github.com/GXNU-ZhongLab/AQATrack> defines the 256-pixel configuration
with `TEST.SEARCH_FACTOR: 4.0` and implements a square crop of side

```text
ceil(search_factor * sqrt(target_width * target_height)).
```

SemTraTrack now adopts that published convention. This is materially different
from independently multiplying the target width and height by four.

## Materials still needed for the SemTraTrack SOT table

These are SemTraTrack reproduction inputs, not a "LINR localization branch":

1. The exact class-agnostic candidate-localizer architecture and configuration.
2. Pretraining and training datasets, splits, frame sampling, augmentations,
   optimizer schedule, and random seeds.
3. Dataset-specific checkpoints, especially the GOT-10K-only checkpoint needed
   by its one-shot protocol.
4. Export/runtime versions and the precise candidate decoding, confidence/NMS,
   lost-target, and clipping settings.
5. Per-sequence raw predictions and benchmark commands for LaSOT,
   LaSOT_ext, GOT-10K, and TrackingNet.

The current runner reproduces the documented first-frame initialization,
factor-4 square search crop, candidate-to-image mapping, LGATracker scoring,
and result-file formats once those localizer assets are supplied.
