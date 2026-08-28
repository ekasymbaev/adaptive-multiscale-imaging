# Adaptive Multiscale Imaging

Research code for an uncertainty-guided selective high-resolution inspection
prototype using scientific imaging data.

## Prototype 1 status

Step 1 (dataset acquisition/audit), Step 2 (coarse segmentation), Step 3
(held-out MC-dropout uncertainty evaluation), Step 4 (tile-level ranking
evaluation), Step 5 (native-tile fine segmentation), Step 6 (adaptive fusion
evaluation), and Step 7 (online conditional inference and synchronized runtime
evaluation) are implemented.

The first dataset is the M-A island segmentation dataset for bainitic steel SEM
images (Figshare DOI: `10.6084/m9.figshare.19232523.v2`). The audit uses the 40
native 2048 x 1536 TIFF images and COCO polygon annotations from the archive's
combined appendix. Raster masks are derived from the polygons because the source
archive does not include full-resolution PNG masks.

Run the reproducible workflow from the repository root:

```bash
conda run -n multiscale-imaging python scripts/download_ma_islands.py
PYTHONPATH=src conda run -n multiscale-imaging python scripts/audit_ma_islands.py
PYTHONPATH=src conda run -n multiscale-imaging python scripts/prepare_coarse_data.py
PYTHONPATH=src conda run -n multiscale-imaging python scripts/train_coarse_cv.py
PYTHONPATH=src conda run -n multiscale-imaging python scripts/evaluate_coarse_uncertainty.py
PYTHONPATH=src conda run -n multiscale-imaging python scripts/evaluate_tile_selection.py
PYTHONPATH=src conda run -n multiscale-imaging python scripts/train_fine_cv.py
conda run -n multiscale-imaging python scripts/audit_fine_reconstruction.py
PYTHONPATH=src conda run -n multiscale-imaging python scripts/evaluate_adaptive_fusion.py
PYTHONPATH=src conda run -n multiscale-imaging python scripts/audit_adaptive_fusion.py
PYTHONPATH=src conda run -n multiscale-imaging python scripts/evaluate_online_adaptive.py
PYTHONPATH=src conda run -n multiscale-imaging python scripts/audit_online_adaptive.py
PYTHONPATH=src conda run -n multiscale-imaging python -m unittest discover -s tests -v
node scripts/build_coarse_results_workbook.mjs
node scripts/build_uncertainty_results_workbook.mjs
node scripts/build_tile_selection_results_workbook.mjs
node scripts/build_fine_results_workbook.mjs
node scripts/build_adaptive_fusion_results_workbook.mjs
```

Generated raw and processed data live under `data/` and audit outputs under
`results/dataset_audit/`; both are intentionally ignored by Git. The deterministic
image-level split manifests under `data/splits/` are versioned.

See [`data/README.md`](data/README.md) for provenance and directory details.

The coarse experiment uses the existing five-fold, image-level split manifest.
Each fold contains 24 training, 8 validation, and 8 test images. The compact
U-Net is selected by validation Dice and evaluated exactly once on each fold's
held-out test images. Outputs are written to `results/coarse_model/`.
The optional workbook builder requires the `@oai/artifact-tool` Node package.

Step 3 loads the five frozen best checkpoints and runs eight stochastic passes
per held-out image. Only dropout is enabled; batch normalization remains frozen.
Uncertainty maps are generated before labels are loaded, and ground truth is
used only for the subsequent error-association analysis. Outputs are written to
`results/coarse_uncertainty/`; the verified Excel summary is written to
`outputs/step3_coarse_uncertainty/uncertainty_results.xlsx`.

Step 4 maps each native 256 x 256 tile exactly to one 64 x 64 coarse block,
ranks the 48 tiles within each image by 90th-percentile predictive entropy,
and compares fixed budgets with 100 reproducible random trials, predictive
variance, and an evaluation-only error oracle. Ground truth is loaded only
after uncertainty and random rankings have been generated. Outputs are written
to `results/tile_selection/`; the verified workbook is written to
`outputs/step4_tile_selection/tile_selection_results.xlsx`.

Step 5 trains the same compact 482,449-parameter U-Net independently in each
fold using all native 256 x 256 tiles from training images only. Uniform tile
sampling is retained because only 17 of 1,920 dataset tiles are empty; a
fold-specific positive-class weight handles pixel imbalance. All 48 tiles of
each held-out image are predicted and reassembled into exact 2048 x 1536 maps.
The primary comparison upsamples each frozen coarse prediction with nearest
neighbor interpolation and evaluates both models against the same native mask.
Outputs are written to `results/fine_model/`; the verified workbook is written
to `outputs/step5_fine_model/fine_model_results.xlsx`.

Across 40 out-of-fold images, the fine model raises macro Dice from 0.7799 to
0.8168 and reduces mean absolute M-A area-fraction error from 0.0427 to 0.0263.
The reconstruction audit also detects a 1.27x error-rate ratio in a four-pixel
band around internal fine-tile seams, versus 0.98x for the coarse control. This
boundary effect should be addressed by contextual padding, overlap, or a
fusion-specific blending rule in the next experiment.

Step 6 freezes all five coarse and fine checkpoints; it performs no additional
training. For each held-out image, it ranks tiles using the Step 4 entropy or
variance scores and substitutes the corresponding frozen fine-model
probabilities into the frozen coarse probability map. The primary fusion rule
uses an eight-pixel, cosine-squared feather around internal tile boundaries;
hard replacement is retained as an ablation. Random selection uses 100 fixed
trials per image and budget, and a ground-truth-dependent oracle is used for
evaluation only. Outputs are written to `results/adaptive_fusion/`; the verified
workbook is written to
`outputs/step6_adaptive_fusion/adaptive_fusion_results.xlsx`.

Across all 40 out-of-fold images, entropy-guided fusion at K=12 (25% native
coverage) reaches 0.7968 macro Dice, compared with 0.7799 for coarse-only,
0.7888 for random selection, and 0.8168 for full fine inference. At K=24 (50%
coverage), it reaches 0.8061 Dice and recovers 71.0% of the full-fine Dice gain.
Entropy selection beats random selection at every partial budget and in all
four source regimes at K=12. Feathering reduces the internal-boundary error
ratio from 1.27 for tiled full-fine inference to 1.09 for K=12 adaptive fusion.

This is an offline counterfactual evaluation: fine probabilities are computed
for all tiles once so every policy can be compared reproducibly. Reported cost
is therefore selected native-resolution tile coverage, not measured acquisition
time, wall-clock runtime, or hardware energy. A later online implementation
must execute the fine model only for selected tiles before claiming realized
computational savings.

Step 7 implements that online execution test. It regenerates each held-out
image's eight-pass MC-dropout uncertainty, ranks tiles online, and passes only
the selected K=12 or K=24 native tiles to the frozen fine model. Coarse,
uncertainty, selection, fine-inference, and fusion stages are timed separately
with accelerator synchronization. Common tensor shapes are warmed up and
excluded, each stage is measured three times, and the per-image median is used.
All online rankings, masks, and metrics reproduce the frozen Step 4-6 references
exactly. Outputs are written to `results/online_adaptive/`.

On Apple Silicon MPS, entropy-guided K=12 reaches 0.7968 macro Dice in 172.6 ms
per image, versus 0.8168 Dice in 184.4 ms for full-fine inference. The selected
fine stage itself takes 45.2 ms, or 25.7% of full-fine-model time, but eight-pass
MC dropout adds 101.1 ms. The resulting end-to-end saving is therefore only
11.75 ms per image (paired 95% CI 9.00-14.25 ms), with K=12 faster on 92.5% of
images. K=24 reaches 0.8061 Dice but takes 220.2 ms, making it 35.85 ms slower
than full fine (paired 95% CI 33.28-38.92 ms). The next systems priority is to
reduce uncertainty-estimation cost rather than increase the inspection budget.

Step 7 measures synchronized model and array-processing time, excluding dataset
file I/O. The fine model genuinely processes only selected tiles, but the stored
native TIFF must still be decoded as a full image. High-resolution acquisition
cost therefore remains represented by tile coverage rather than measured
microscope or region-addressable storage time.
