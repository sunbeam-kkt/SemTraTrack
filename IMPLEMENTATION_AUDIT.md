# Implementation audit against the current manuscript

Audit target: **SemTraTrack: Multilevel Fusion of Semantic Foreground Cues and
Causal Trajectory Context for Robust UAV Tracking**.

## Corrected mismatches

| Area | Earlier implementation | Correction in this revision |
|---|---|---|
| Project identity | Legacy SPCATrack package/config/artwork remained | Renamed the executable package/configuration to `semtratrack` and replaced the architecture/result artwork with current SemTraTrack assets |
| Stage-I pair selection | Consecutive frames without a shared identity were dropped before training | Retains every consecutive annotated pair by default and lets TCR select only identities valid in both frames |
| MSA-NWD attention | One scale-softmax was derived after averaging logits across the whole batch | Computes `alpha_s` independently for each image, masks scales without positive anchors, then averages image losses |
| Stage-II corpus | One MOT file and one shared frame size per run | Supports a multi-sequence JSONL manifest, isolates identity/history state by sequence, validates names and dimensions, and writes protocol metadata |
| Protocol drift | A corpus/checkpoint could be combined with different temporal or cue settings | Persists and validates `W_s`, `K`, `L_max`, dimensions, beta, cue weights, state thresholds, and motion threshold |
| Semantic-control runs | Documentation launched only seed 0 | Added a launcher for the same three default seeds (`0,1,2`) across every reported context variant |
| Generic SOT result file | First row used `xyxy`, later rows used `xywh`, and an extra frame column was written | Emits exactly four `x,y,w,h` values for every frame, with zero rows after termination |
| Generic SOT search crop | Factor and geometry were unspecified; code multiplied width and height independently | Uses the AQATrack-256 factor `4.0` with square side `ceil(4*sqrt(w*h))`, constant-zero border padding, and image-bound clipping after coordinate restoration |
| Anti-UAV410 | No benchmark-format entry point | Added first-frame-only evaluation and official `{"res": ...}` output with `[0]` for unmatched/lost frames |
| Checkpoint loading | Variant checks were partial | Enforces variant, prompt-control seed, cue/attribute switches, dimensions, and association protocol |
| Documentation | Old title, table-number coupling, stale manifest, and outdated images | Updated to the current title and implementation paths and documented unresolved reproduction inputs |

## Verified unchanged mappings

- SPS produces a shared single-channel response and foreground/background
  gated feature tensors.
- TCR uses valid adjacent-frame observations with the same identity; MIFD uses
  off-diagonal same-frame prototype similarities.
- BNS-HM selects the top 2% foreground-like background locations and applies
  margin `0.30`.
- Stage-I loss weights and optimizer schedule match the current manuscript.
- LGATracker uses the four stated pairwise cues, original-frame scale
  thresholds, diagonal-normalized motion, causal completed windows, and
  Hungarian cost `1-S`.
- The structured mapper dimensions are `64 -> 128 -> 128`; `W_g` and `W_e`
  project into a shared 128-D space.

## Resolved manuscript items and remaining evidence gap

The overall loss now contains exactly one `lambda_nwd * L_nwd` term, matching
the coefficient paragraph and the code's single MSA-NWD component with weight
`0.50`.

The generic-SOT section now states factor `4.0` and its exact square-crop
geometry, and both SOT runners use the same tested default. LINR itself is a
plug-in representation module rather than a detachable localization branch.
As of 2026-10-03, its official public repository contains only a README,
license, and raw-results link; it says the source is still being organized.
The unresolved input for this project is therefore the SemTraTrack
class-agnostic candidate-localizer package: model/configuration, training data
and sampling protocol, dataset-specific checkpoints (including the GOT-10K
one-shot checkpoint), export command, decoding thresholds, and raw benchmark
outputs. The included runner covers the causal association protocol around an
experimenter-supplied Ultralytics-compatible localizer.

## Validation scope

The source tree is syntax-checked and includes regression tests for the
corrected logic. End-to-end numerical reproduction still requires the pinned
YOLOv13 dependency, PyTorch/CUDA environment, datasets, detector exports, and
trained checkpoints listed in the README.
