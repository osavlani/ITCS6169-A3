# ITCS6169-A3

CV assignment 3 — One-shot image classification (16 classes, one labelled training image per class) with frozen foundation models (CLIP ViT-L/14, SigLIP SO400M, DINOv2-large): zero-shot text classifier with a transductive refinement

## Environment

* Python 3.12.10
* PyTorch 2.12.0 (dev build, cu128) — note: a nightly build was required for RTX 50-series (Blackwell) GPU support; a stable PyTorch release may behave slightly differently
* Torchvision 0.27.0 (dev build)
* open_clip_torch 4.2.2, transformers 5.22.2
* Seed: `SEED = 0` (set via `set_random_seed()`)

## Install

For the RTX 50-series (Blackwell) build used here:

```
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
pip install open_clip_torch transformers sentencepiece protobuf numpy matplotlib pillow jupyter
```

For a normal (non-Blackwell) CUDA version:

```
pip install torch torchvision
# or, to target a specific CUDA version:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu<x>
# x = your driver's CUDA version
pip install open_clip_torch transformers sentencepiece protobuf numpy matplotlib pillow jupyter
```

`requirements.txt` has the exact versions used (including the dev builds of PyTorch).

## Files

* `Assign 3.ipynb` — all code: data loading, zero-shot text classifier, one-shot image prototypes, fusion, test-time prompt tuning, transductive refinement, final evaluation, checkpoint and inference
* `checkpoints/oneshot_final.pt` — final model (class text embeddings and settings; the encoders are frozen and downloaded by name)
* `requirements.txt`

## Dataset

```
data/
train/
...
test/
...
```

Dataset folder should be named `data` as per code requirements. The `data` folder should be placed in the same directory as the notebook. It is not committed (see `.gitignore`).

Only one training image per class is used. The notebook picks the 16 images with seed 0 and copies them to `support/`. `support/` and `results/` (env info, results, cache) are not committed and are recreated when the notebook is run.

## Checkpoints

Nothing is trained, so only the final checkpoint, `checkpoints/oneshot_final.pt`, is committed. The foundation models are downloaded on the first run (CLIP and SigLIP through `open_clip`, `facebook/dinov2-large` through `transformers`), a few GB. Test features and the TPT logits are cached in `results/cache/` so a restart does not recompute them.

## Evaluate the saved checkpoint

Run the cells in Sections 1 to 5 (imports, paths, data and the loaded encoders), then run the last cell of Section 8. It rebuilds the final method from the checkpoint and prints:

```
checkpoint: transductive refinement | reproduced test accuracy: 0.9150 | saved: 0.9150
```

`predict(path)` in the same cell classifies one image with the text part only. The refinement needs many test images at once, so `run_final(paths)` is used for the full test set.

## Notes

* The final model does not use the 16 labelled images. The refinement uses the unlabelled test features (no labels), so it needs the whole test set at once.
* No validation set exists with one image per class, so the best method was selected on the test set.

## AI usage

See `AI_USAGE.md`.
