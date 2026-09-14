"""
Generate per-class lesion-focused LLaVA captions for a support set.

Policy:
- Looks for class subdirectories under --data-root (each subdir = class name).
- For each class, visits support images in round-robin order until it has N
  unique captions (default 50).
- Each visit applies a lesion-preserving random resized crop, optional flip,
  and mild color jitter before stochastic LLaVA decoding.
- Enforces 10-20 word constraint via regeneration attempts and light postprocessing.
- Saves JSON mapping: { class_name: [caption1, caption2, ...], ... }.
- BLIP + Flan-T5 is available only through --allow-fallback; it is not the
  VLTCoOp language teacher described in the paper.

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
import pickle
from collections import defaultdict
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm
from torchvision import transforms as T

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
    def __init__(self, device='cuda', llava_model='llava-hf/llava-1.5-7b-hf', allow_fallback=False):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.llava_processor = None
        self.llava = None
        try:
            # Dynamic imports keep this utility importable with legacy
            # transformers versions while producing an actionable error.
            import transformers
            processor_class = getattr(transformers, 'AutoProcessor')
            model_class = getattr(transformers, 'AutoModelForVision2Seq', None)
            if model_class is None:
                model_class = getattr(transformers, 'LlavaForConditionalGeneration')
            dtype = torch.float16 if self.device == 'cuda' else torch.float32
            self.llava_processor = processor_class.from_pretrained(llava_model)
            self.llava = model_class.from_pretrained(llava_model, torch_dtype=dtype).to(self.device).eval()
            print(f'Loaded LLaVA caption model: {llava_model}')
        except Exception as exc:
            if not allow_fallback:
                raise RuntimeError(
                    'Unable to load LLaVA. Install a transformers version with LLaVA support or pass '
                    '--allow-fallback to use the non-method BLIP + Flan-T5 fallback.'
                ) from exc
            print(f'Warning: LLaVA unavailable ({exc}); using BLIP + Flan-T5 fallback.')

        if self.llava is not None:
            return

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
            "Rewrite the following image caption to be 20-30 words, with no numbering or newlines, and emphasize discriminative visual cues as instructed. "
            f"Instruction: {instruction}\nCaption: {caption}\n\nRewritten caption:"
        )
        inputs = self.rewriter_tokenizer(prompt, return_tensors='pt', truncation=True).to(self.device)
        gen_kwargs = dict(max_new_tokens=64, do_sample=True, top_p=0.95, temperature=0.7)
        for _ in range(tries):
            out = self.rewriter.generate(**inputs, **gen_kwargs)
            text = self.rewriter_tokenizer.decode(out[0], skip_special_tokens=True).strip()
            text = ' '.join(text.split())
            if 20 <= len(text.split()) <= 30:
                return text
        # final best-effort: adjust length heuristically
        words = text.split()
        if len(words) < 20:
            # Append conservative visual modifiers to reach the requested range.
            extras = ['scattered', 'small', 'confluent', 'distinct', 'marginal', 'vein-adjacent', 'irregular', 'chlorotic', 'necrotic', 'textured', 'localized', 'visible']
            needed = 20 - len(words)
            words += extras[:needed]
            while len(words) < 20:
                words.append('visible')
            return ' '.join(words)
        if len(words) > 30:
            return ' '.join(words[:30])
        return text

    def _generate_llava(self, image, instruction):
        conversation = [{
            'role': 'user',
            'content': [
                {'type': 'image'},
                {'type': 'text', 'text': instruction},
            ],
        }]
        if hasattr(self.llava_processor, 'apply_chat_template'):
            prompt = self.llava_processor.apply_chat_template(conversation, add_generation_prompt=True)
        else:
            prompt = f'USER: <image>\\n{instruction}\\nASSISTANT:'
        inputs = self.llava_processor(text=prompt, images=image, return_tensors='pt')
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            output_ids = self.llava.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
            )
        generated_ids = output_ids[:, inputs['input_ids'].shape[1]:]
        return self.llava_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    def generate_for_image(self, image, instruction):
        if self.llava is not None:
            return self._generate_llava(image, instruction)

        # Step 1: get base caption via BLIP (or pipeline)
        base_caption = None
        img = image
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


def render_lesion_preserving_view(image):
    """Vary framing and illumination without synthesizing lesion content."""
    height, width = image.height, image.width
    image = T.RandomResizedCrop((height, width), scale=(0.80, 1.0), ratio=(0.9, 1.1))(image)
    if random.random() < 0.5:
        image = T.functional.hflip(image)
    return T.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.02)(image)


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


def collect_support_cache_paths(cache_path):
    """Read the exact train split saved by the dataset's few-shot sampler."""
    with open(cache_path, 'rb') as handle:
        cached = pickle.load(handle)
    train_items = cached.get('train') if isinstance(cached, dict) else None
    if not train_items:
        raise ValueError(f'Few-shot cache has no non-empty train split: {cache_path}')
    classes = defaultdict(list)
    for item in train_items:
        classname = getattr(item, 'classname', None)
        impath = getattr(item, 'impath', None)
        if not classname or not impath:
            raise ValueError(f'Invalid support item in {cache_path}')
        classes[classname].append(impath)
    return dict(classes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True, help='Path to dataset root (folders per class)')
    parser.add_argument('--out-file', required=True, help='Output JSON file to write mapping')
    parser.add_argument('--support-cache', help='Optional split_fewshot/shot_*-seed_*.pkl; captions then use the exact training support set')
    parser.add_argument('--n-prompts', type=int, default=50)
    parser.add_argument('--llava-model', default='llava-hf/llava-1.5-7b-hf')
    parser.add_argument('--allow-fallback', action='store_true', help='Allow BLIP + Flan-T5 if LLaVA cannot load')
    parser.add_argument('--allow-incomplete', action='store_true', help='Write fewer than N captions when unique decoding is exhausted')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-retries', type=int, default=3)
    parser.add_argument('--instruction', default=DEFAULT_INSTRUCTION)
    args = parser.parse_args()

    random.seed(args.seed)
    classes = collect_support_cache_paths(args.support_cache) if args.support_cache else collect_class_image_paths(args.data_root)
    if not classes:
        print(f"No class subfolders with images found under {args.data_root}")
        return
    print(f"Found {len(classes)} classes. Sampling images and generating {args.n_prompts} captions per class.")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gen = CaptionGenerator(device=device, llava_model=args.llava_model, allow_fallback=args.allow_fallback)

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
                with Image.open(img_path) as source:
                    view = render_lesion_preserving_view(source.convert('RGB'))
                caption = gen.generate_for_image(view, args.instruction)
                # simple postprocess
                caption = caption.replace('\n', ' ').strip()
                # remove trailing periods
                if caption.endswith('.'):
                    caption = caption[:-1]
                words = caption.split()
                if 20 <= len(words) <= 30:
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
        # Do not manufacture textual variants: N_cap counts unique generated
        # captions in the language-teacher definition.
        if len(captions) < args.n_prompts:
            message = f"Only generated {len(captions)}/{args.n_prompts} unique captions for class {classname}."
            if not args.allow_incomplete:
                raise RuntimeError(message + ' Increase --max-retries or use --allow-incomplete explicitly.')
            print('Warning: ' + message)
        out[classname] = captions[:args.n_prompts]

    # write out
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote generated prompts to {out_path}")


if __name__ == '__main__':
    main()
