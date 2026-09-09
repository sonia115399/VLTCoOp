# VLTCoOp

Official PyTorch implementation of VLTCoOp, a prompt-learning method that
aligns learnable textual contexts with selected vision-language descriptions
and frozen visual class prototypes.

The VLTCoOp implementation is self-contained. The method is implemented
in `trainers/VLTCoOp/vltcoop_clip.py`; dataset-specific language descriptions
are in `trainers/vltcoop_templates.py` or may be generated from images with
`scripts/generate_vlm_prompts.py`.

## Installation

Create a Python 3.10+ environment with a PyTorch build compatible with your
CUDA driver, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Data Layout

Images are not distributed with this repository. Place each dataset under the
directory passed to `--root`:

```text
data/
  tea/
    tea/
      <class_name>/
        <image files>
    split_tea.json
    generated_prompts.json
```

The repository keeps split manifests and generated-prompt JSON files as
metadata, while `.gitignore` excludes images and regenerated few-shot caches.
If a split file is absent, the dataset loader creates one from the class
folders. Generated prompts are optional: VLTCoOp falls back to the curated
templates when a class has no generated description.

To generate 50 descriptions per class from a local dataset:

```bash
python scripts/generate_vlm_prompts.py \
  --data-root data/tea \
  --out-file data/tea/generated_prompts.json \
  --n-prompts 50
```

## Training

Run a three-seed few-shot experiment with the OpenAI CLIP backbone:

```bash
bash scripts/vltcoop/few_shot.sh data tea 16 CLIP
```

For a single run, use the explicit training command:

```bash
python train.py \
  --root data \
  --seed 1 \
  --trainer VLTCoOp_CLIP \
  --dataset-config-file configs/datasets/tea.yaml \
  --config-file configs/trainers/VLTCoOp/few_shot/tea.yaml \
  --output-dir output/tea/shots_16/VLTCoOp_CLIP/seed1 \
  DATASET.NUM_SHOTS 16
```

Base-to-novel evaluation is available through:

```bash
bash scripts/vltcoop/base_to_novel.sh data tea CLIP
```

All method hyperparameters are under `TRAINER.VLTCOOP` in the corresponding
YAML files. Training artifacts are written below `output/` and excluded from
version control.

## Comparison Methods

The repository includes the implementations, configurations, and experiment
scripts used for comparison: CoOp, CoCoOp, KgCoOp, ProGrad, BiomedCoOp,
Tip-Adapter, CLIP-Adapter, linear probing, zero-shot CLIP, PromptSRC, and
MaPLe. The Dassl-style trainers are under `trainers/`; the adapter and linear
probe runners use `main.py`. To keep the release focused, baseline trainer
directories retain their OpenAI CLIP implementations only; `Zeroshot/` retains
its full set of backbone variants.

Use the corresponding command under `scripts/` with the documented positional
arguments. For example, CoOp on the tea dataset can be launched with:

```bash
bash scripts/coop/few_shot.sh data tea 16 CLIP
```

## Repository Layout

```text
train.py                    training and evaluation entry point
trainers/VLTCoOp/           VLTCoOp trainer
trainers/{CoOp,...}/        comparison method implementations
trainers/PromptSRC/         PromptSRC entry point
trainers/MaPLe/             MaPLe entry point
trainers/vltcoop_templates.py curated class descriptions
configs/                    dataset and VLTCoOp experiment settings
datasets/                   dataset adapters
clip/                       OpenAI CLIP implementation
dassl/                      minimal training framework used by this project
scripts/                    VLTCoOp and comparison experiment scripts
scripts/promptsrc/          PromptSRC-specific few-shot runner
scripts/maple/              MaPLe-specific few-shot runner
scripts/generate_vlm_prompts.py optional description generation
```

## License

This project is released under the MIT License. The bundled CLIP implementation
originates from OpenAI CLIP and retains its original license and notices.
