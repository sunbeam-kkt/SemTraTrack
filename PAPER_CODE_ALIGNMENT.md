# SemTraTrack paper-code alignment

This source tree is aligned to **SemTraTrack: Multilevel Fusion of Semantic
Foreground Cues and Causal Trajectory Context for Robust UAV Tracking**.

## Stage I: ISPS-SOD

| Manuscript component | Implementation |
|---|---|
| YOLOv13-S, one UAV class | `configs/yolov13s-UAV.yaml` |
| Adjacent-frame identity pairs | `semtratrack/training/data.py` |
| SPS: `F -> A -> M_F, M_B` | `semtratrack/sps.py::SPSBranch` |
| Box-conditioned `q_sep` diagnostic | `semtratrack/sps.py::q_sep` |
| Foreground/background contrastive loss | `foreground_background_contrastive_loss` |
| Per-image MSA-NWD scale attention | `semtratrack/training/model.py::ScaleAttentionNWD` |
| TCR | `PaperAlignedDetectionModel._tcr` |
| MIFD | `mifd_loss` |
| BNS-HM | `bns_hard_negative_loss` |
| Complete weighted detector objective | `PaperAlignedDetectionModel.loss` |
| 50 epochs, batch 16, 640, AdamW, AMP | `train.py` |
| Component and 1280-pixel controls | `train.py --detector-ablation/--imgsz` |

SPS is applied to the high-resolution P3 feature stream. Detector-positive
locations used by the contrastive objective are recomputed from the native
YOLOv13 task-aligned assignment used by the detection loss. TCR includes only
identities valid in both adjacent frames. TCR and MIFD construct instance
prototypes by restricting the shared SPS response to each ground-truth box.

## Stage II: LGATracker

| Manuscript component | Implementation |
|---|---|
| Five window attributes (QS, SS, AP, MD, AC) | `summarize_window`, `WindowSummary.prompt` |
| Task-specific fixed tokenizer | `StructuredPromptTokenizer` |
| Trainable `E_tok`, `d_t=64` | `StructuredPromptEncoder.token_embedding` |
| Mean pooling + GELU MLP `64->128->128` | `StructuredPromptEncoder` |
| `W_g: 4->128`, `W_e: 128->128` | `PairContextAdapter` |
| Position, size, state, direction cues | `LGATracker.pairwise_cues` |
| Cue weights `.50/.20/.10/.20` | `SemTraTrackConfig.cue_weights` |
| `W_s=30`, `K=10`, `L_max=5` | `SemTraTrackConfig` |
| Causal completed-window history | `LGATracker._complete_window` |
| Context compatibility and `beta=.20` fusion | `PairContextAdapter.compatibility`, `_score_matrix` |
| Hungarian cost `1-S` | `LGATracker.update` |
| Joint mapper/projection BCE training | `scripts/train_association.py` |
| Multi-sequence causal corpus | `scripts/build_assoc_pairs_from_mot.py` |
| Geometry-only association control | `context_variant=geometry-only` |
| Cue/attribute removals | `--disable-cue`, `--disable-attribute` |
| Three-seed semantic controls | `scripts/train_semantic_controls.py` |
| Corpus/checkpoint protocol guard | `association_signature()` |

The detector is frozen during Stage-II training. `E_tok`, the two-layer
language MLP, `W_g`, and `W_e` are optimized jointly with the clipped
context-compatibility BCE. Hungarian assignment remains inference-only.

## Evaluation paths

| Manuscript protocol | Implementation |
|---|---|
| Multi-UAV MOT output | `track_mot.py` |
| First-frame-only generic SOT | `track_sot.py` |
| Anti-UAV410 official JSON convention | `track_antiuav410.py` |
| Batch-1 end-to-end timing boundary | `scripts/benchmark_e2e.py` |
| Variant peak GPU memory | `scripts/benchmark_peak_memory.py` |

The generic SOT result writer emits one `x,y,w,h` row per frame. The
Anti-UAV410 writer emits `{"res": [...]}` with `[0]` for an unmatched/lost
frame. Neither path reads later-frame ground truth for update or
re-initialization.

## Controlled semantic-content study

`controlled_prompt` and `scripts/train_association.py --variant` implement
`constant`, `shuffled-window`, `attribute-permuted`, `numeric`, and `full`.
All controls reuse `W_s=30`, `K=10`, the 128-D context space, the same pair
features, and the same association loss. Shuffled/permuted values are sampled
only from earlier completed windows in the same sequence.

## Resolved manuscript details

The overall-objective equation now contains one `lambda_nwd * L_nwd` term,
exactly matching the implementation's single MSA-NWD contribution with weight
`0.50`.

The generic-SOT search region now uses factor `4.0` by default under the
AQATrack area convention. `semtratrack/sot.py::search_crop_from_box` extracts a
zero-padded square with side `ceil(4*sqrt(w*h))`; both evaluation entry points
share this implementation and may override the value only for an explicitly
reported sensitivity experiment.

## Reproducibility boundary

Source-level alignment cannot reproduce reported tables by itself. Exact
reproduction also requires the authors' split, detector-to-GT matching
exports, checkpoints, three-seed logs, dataset-specific localization weights,
the generic-SOT localization training/export recipe, and the A100 environment;
none is synthesized here. LINR is a representation plug-in, not a standalone
localization branch, and its public repository does not yet expose executable
source/configuration.
