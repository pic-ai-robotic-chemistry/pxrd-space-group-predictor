# Changelog

## 0.3.0.dev1 — current development version

- Replaced DTW candidate ranking with PXRD peak matching ported from Zikai Xie's
  `pxrd_csp_stop` 1.0.0: native-grid experimental pattern, SNIP correction,
  lattice refinement, bidirectional strong-peak guard and score B ≥ 0.80.
- `pxrd-sg run` and `pxrd-sg rank` now search space groups sequentially and stop
  at the first group whose top-B candidate matches; `run.json` records `searched`
  and `result`. `rank` requires `--space-groups` and accepts `--workers`.
- Processing now keeps a 3.0 kJ/mol per-group energy window and deduplicates it
  with a 0.2 g/cm³ density gate; PXRD arrays are no longer exported.
- `score_pxrd_candidates.py` takes `--experimental`, `--manifest`, `--output-dir`
  and `--workers`, and writes `scores.csv`, `decision.json` and, on a match,
  `selected_refined.cif`.
- Removed DTW, the 200-by-energy/top-ten selection, `dtaidistance` and the
  `verification` extra.
- Workflow stages now run the `src`/`tools` code of the invoking package. An
  older `pxrd-sg` left installed in the same environment previously took
  precedence and ran its own stage tools.

## 0.2.0.dev1

- Added the unified predict, run, rank and doctor CLI, with checked stage resume.
- Published the designated classifier separately in weights-v1.
- Added validated separate Conda prediction and GPU-engine profiles.
- Simplified the agent skill and made CLI installation the primary documented path.
- Moved training and direct-adapter guidance to advanced usage.
- Removed superseded internal audit records, the unused container recipe and
  the unpinned alternative dependency list from the current source tree.


## 0.1.0.dev1 — manuscript/SI synchronization (2026-10-08)

- Aligned DTW with linear intensities, Euclidean local cost and window 31 semantics.
- Added experimental classification preprocessing and CSD-only augmentation.
- Added explicit pretraining/adaptation stage defaults and corrected the graph
  projection learning-rate group.
- Added Others mapping, stable logit ranking, macro-F1, per-class/confusion and
  rank-coverage outputs, and a five-seed paired comparison helper.
- Added deterministic 200-to-ten screening for pre-deduplicated inputs.
- Documented correspondence to the current manuscript and full SI, preserving
  the existing public-core/external-CSP boundary and network topology.

## 0.1.0.dev0

- Rejected non-finite features, logits and losses, invalid graph indices and masks, and degenerate DTW profiles; training now rejects non-finite gradients.
- Preserved explicit or inherited class counts during subset fine-tuning and added optional checkpointed class-to-space-group mappings to prediction outputs.
- Switched checkpoint loading to the restricted weights-only loader while retaining schema-1 compatibility.
- Added input-contract regressions and a synthetic training, fine-tuning, evaluation, inference and DTW workflow test.
- Created a clean staging package for the reader-facing XRayPro space-group screening code.
- Added the formal SERes PXRD encoder, molecular-embedding fusion and training-only graph teacher.
- Added graph-free inference, labeled evaluation and constrained PXRD-DTW candidate scoring.
- Added a pinned ChemBERTa embedding entry point and synthetic smoke workflow.
- Excluded paper evidence production, restricted data, model weights and case-specific CSP implementation.

This entry records staging work only. It is not a public software release.
