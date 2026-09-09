"""
Generate per-class lesion-focused captions for a dataset.

Policy:
- Looks for class subdirectories under --data-root (each subdir = class name).
- For each class, samples images (cycling if necessary) to generate N captions (default 50).
- Tries a multimodal LLaVA model if --model-dir is provided and can be loaded.
  If that fails or not provided, falls back to BLIP (image captioning) + Flan-T5 rewriter.
- Enforces 10-20 word constraint via regeneration attempts and light postprocessing.
- Saves JSON mapping: { class_name: [caption1, caption2, ...], ... }

Usage example:
  python scripts/generate_vlm_prompts.py \
      --data-root data/tea \
      --out-file data/tea/generated_prompts.json \
      --n-prompts 50

"""

import argparse
import os
import random
import json
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm

from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoConfig,
)

# Try imports for BLIP processor/model if available
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    HAS_BLIP = True
except Exception:
    HAS_BLIP = False


DEFAULT_INSTRUCTION = (
    "Generate a caption (20–30 words), no numbering, no newlines. Emphasize discriminative visual cues for leaf disease: "
    "lesion color and center-to-edge gradient; margin character (halo, necrotic rim, serrated, blurred); shape and size; spatial distribution "
    "(scattered, coalescing, along veins, near margin/apex); adaxial vs abaxial; chlorosis vs necrosis; vein reactions; presence of mildew/mycelium/sooty growth. "
    "Contrast symptomatic with adjacent healthy tissue. Use precise terms like angular, concentric rings, shot-hole, target-like, water-soaked, olive-green, "
    "reddish-brown, grayish center, yellow halo, vein-limited. Do not start with 'Diseased leaf' or 'A leaf'."
)


def safe_load_blip(device):
    """Return a BLIP pipeline (image-captioning) if possible, else None."""
    if HAS_BLIP:
        try:
            processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
            model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')
            if device == 'cuda':
                model.to('cuda')
            return processor, model
        except Exception:
            return None
    # try HF pipeline fallback
    try:
        picap = pipeline('image-captioning', model='Salesforce/blip-image-captioning-base', device=0 if torch.cuda.is_available() else -1)
        return picap
    except Exception:
        return None


class CaptionGenerator:
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        # load flan-t5 base for rewriting
        self.rewriter_name = 'google/flan-t5-large'
        try:
            self.rewriter = AutoModelForSeq2SeqLM.from_pretrained(self.rewriter_name).to(self.device)
            self.rewriter_tokenizer = AutoTokenizer.from_pretrained(self.rewriter_name)
        except Exception:
            # fallback to smaller model
            self.rewriter_name = 'google/flan-t5-base'
            self.rewriter = AutoModelForSeq2SeqLM.from_pretrained(self.rewriter_name).to(self.device)
            self.rewriter_tokenizer = AutoTokenizer.from_pretrained(self.rewriter_name)

        # attempt BLIP
        self.blip = safe_load_blip(self.device)

    def rewrite_caption(self, caption, instruction, tries=3):
        # Compose a prompt for the rewriter
        prompt = (
            "Rewrite the following image caption to be 10-20 words, with no numbering or newlines, and emphasize discriminative visual cues as instructed. "
            f"Instruction: {instruction}\nCaption: {caption}\n\nRewritten caption:"
        )
        inputs = self.rewriter_tokenizer(prompt, return_tensors='pt', truncation=True).to(self.device)
        gen_kwargs = dict(max_new_tokens=64, do_sample=True, top_p=0.95, temperature=0.7)
        for _ in range(tries):
            out = self.rewriter.generate(**inputs, **gen_kwargs)
            text = self.rewriter_tokenizer.decode(out[0], skip_special_tokens=True).strip()
            text = ' '.join(text.split())
            if 10 <= len(text.split()) <= 20:
                return text
        # final best-effort: adjust length heuristically
        words = text.split()
        if len(words) < 10:
            # append adjectives to reach ~12 words
            extras = ['scattered', 'small', 'confluent', 'along veins', 'near margin', 'distinct']
            needed = 12 - len(words)
            words += extras[:needed]
            return ' '.join(words)
        if len(words) > 20:
            return ' '.join(words[:20])
        return text

    def generate_for_image(self, image_path, instruction):
        # Step 1: get base caption via BLIP (or pipeline)
        base_caption = None
        img = Image.open(image_path).convert('RGB')
        # If blip returned processor+model tuple
        if isinstance(self.blip, tuple):
            try:
                processor, model = self.blip
                inputs = processor(images=img, return_tensors='pt').to(self.device)
                out_ids = model.generate(**inputs, max_new_tokens=32)
                base_caption = processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
            except Exception:
                base_caption = None
        elif self.blip is not None:
            try:
                res = self.blip(img)
                if isinstance(res, list) and 'generated_text' in res[0]:
                    base_caption = res[0]['generated_text']
                elif isinstance(res, list) and isinstance(res[0], dict):
                    base_caption = list(res[0].values())[0]
            except Exception:
                base_caption = None

        if base_caption is None:
            # as last resort, create a simple placeholder using filename
            base_caption = f"An image of a tea leaf showing lesions"

        # Step 2: rewrite using flan-t5 to follow instruction and length
        final = self.rewrite_caption(base_caption, instruction)
        # cleanup: remove any leading verbs like 'A leaf with...' ensure not starting with 'Diseased leaf'
        if final.lower().startswith('diseased leaf'):
            final = final[len('diseased leaf'):].strip().lstrip('.,;:')
        return final


def collect_class_image_paths(data_root):
    data_root = Path(data_root)
    classes = {}
    # assume each immediate subdirectory is a class containing images
    immediate_dirs = [p for p in sorted(data_root.iterdir()) if p.is_dir()]
    for p in immediate_dirs:
        images = [str(x) for x in p.rglob('*') if x.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']]
        if images:
            classes[p.name] = images
    # If we found only one immediate dir (e.g. data/tea/tea) and that dir itself contains subdirs that look like classes,
    # prefer the deeper level: data_root/<child>/* -> treat each subdir as a class.
    if len(classes) == 1:
        # check if any immediate dir contains multiple subdirectories that each have images
        deeper_found = False
        for p in immediate_dirs:
            child_dirs = [q for q in sorted(p.iterdir()) if q.is_dir()]
            candidate = {}
            for q in child_dirs:
                imgs = [str(x) for x in q.rglob('*') if x.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']]
                if imgs:
                    candidate[q.name] = imgs
            if candidate and len(candidate) > 1:
                classes = candidate
                deeper_found = True
                break
        # also handle case where there is a single nested directory with many class subdirs
        if not deeper_found:
            # try scanning two levels deep: data_root/*/* (if those are class folders)
            two_level = {}
            for p in immediate_dirs:
                for q in [d for d in sorted(p.iterdir()) if d.is_dir()]:
                    imgs = [str(x) for x in q.rglob('*') if x.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']]
                    if imgs:
                        two_level[q.name] = imgs
            if two_level:
                classes = two_level
    # if no subdirs found, try to treat images in data_root as single class
    if not classes:
        images = [str(x) for x in data_root.rglob('*') if x.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']]
        if images:
            classes[data_root.name] = images
    return classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True, help='Path to dataset root (folders per class)')
    parser.add_argument('--out-file', required=True, help='Output JSON file to write mapping')
    parser.add_argument('--n-prompts', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-retries', type=int, default=3)
    parser.add_argument('--instruction', default=DEFAULT_INSTRUCTION)
    args = parser.parse_args()

    random.seed(args.seed)
    classes = collect_class_image_paths(args.data_root)
    if not classes:
        print(f"No class subfolders with images found under {args.data_root}")
        return
    print(f"Found {len(classes)} classes. Sampling images and generating {args.n_prompts} captions per class.")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gen = CaptionGenerator(device=device)

    out = {}
    for classname, images in classes.items():
        captions = []
        if not images:
            continue
        # shuffle for variety
        imgs = images.copy()
        random.shuffle(imgs)
        idx = 0
        pbar = tqdm(total=args.n_prompts, desc=f"Generating for {classname}")
        attempts = 0
        while len(captions) < args.n_prompts and attempts < args.n_prompts * args.max_retries:
            img_path = imgs[idx % len(imgs)]
            try:
                caption = gen.generate_for_image(img_path, args.instruction)
                # simple postprocess
                caption = caption.replace('\n', ' ').strip()
                # remove trailing periods
                if caption.endswith('.'):
                    caption = caption[:-1]
                words = caption.split()
                if 10 <= len(words) <= 20:
                    if caption not in captions:
                        captions.append(caption)
                        pbar.update(1)
                else:
                    # allow a few retries; the rewriter tries to enforce but sometimes fails
                    pass
            except Exception as e:
                # skip image on error
                pass
            idx += 1
            attempts += 1
        pbar.close()
        # if not enough captions generated, fill with variants by simple templating
        if len(captions) < args.n_prompts:
            print(f"Warning: only generated {len(captions)} captions for class {classname}; filling with variants.")
            base = captions[0] if captions else f"leaf with small concentrated lesions near margin"
            i = 0
            while len(captions) < args.n_prompts:
                captions.append(f"{base} variant {i}")
                i += 1
        out[classname] = captions[:args.n_prompts]

    # write out
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote generated prompts to {out_path}")


if __name__ == '__main__':
    main()
