"""PromptSRC trainer for the OpenAI CLIP backbone.

Prompt placement and objectives follow the authors' MIT-licensed implementations:
https://github.com/muzairkhattak/PromptSRC
https://github.com/muzairkhattak/multimodal-prompt-learning
See MULTIMODAL_LICENSE for attribution. All trainable tensors live in the
prompt learner; the same frozen CLIP backbone also supplies PromptSRC's teacher.
"""

import copy
import hashlib
import json
import logging
import math
import os
import pickle
import random
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T

from clip import clip
from clip.model import convert_weights
from dassl.engine import TRAINER_REGISTRY, TrainerX
from datasets.utils import DatasetWrapper, Datum
from trainers.promptsrc_templates import IMAGENET_TEMPLATES


DEFAULTS = {
    "MAPLE": dict(N_CTX=2, CTX_INIT="a photo of a", PROMPT_DEPTH=9, PREC="amp"),
    "PROMPTSRC": dict(
        N_CTX_TEXT=4, N_CTX_VISION=4, CTX_INIT="a photo of a",
        PROMPT_DEPTH_TEXT=9, PROMPT_DEPTH_VISION=9, PREC="amp",
        TEXT_LOSS_WEIGHT=25.0, IMAGE_LOSS_WEIGHT=10.0, GPA_MEAN=30.0, GPA_STD=30.0,
    ),
}


def extend_multimodal_cfg(cfg):
    for name, defaults in DEFAULTS.items():
        if not hasattr(cfg.TRAINER, name):
            setattr(cfg.TRAINER, name, defaults.copy())
        else:
            node = getattr(cfg.TRAINER, name)
            for key, value in defaults.items():
                if not hasattr(node, key):
                    setattr(node, key, value)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


class FewShotUnpickler(pickle.Unpickler):
    """Read original Dassl Datum objects without resampling historic splits."""
    def find_class(self, module, name):
        if name == "Datum" and module in ("dassl.data.datasets.base_dataset", "dassl.data.datasets"):
            return Datum
        return super().find_class(module, name)


def load_fewshot(cfg, dataset):
    if cfg.DATASET.NUM_SHOTS < 1:
        raise ValueError("A positive few-shot count is required")
    path = Path(dataset.split_fewshot_dir) / "shot_{}-seed_{}.pkl".format(cfg.DATASET.NUM_SHOTS, cfg.SEED)
    if path.exists():
        with path.open("rb") as stream:
            data = FewShotUnpickler(stream).load()
        logging.info("Reusing original few-shot samples: %s", path)
    else:
        data = {"train": dataset.generate_fewshot_dataset(dataset.train_x, num_shots=cfg.DATASET.NUM_SHOTS),
                "val": dataset.generate_fewshot_dataset(dataset.val, num_shots=min(cfg.DATASET.NUM_SHOTS, 4))}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    dataset._train_x, dataset._val = data["train"], data["val"]
    return path


def prepare_data(cfg, dataset):
    """Relocate cached paths in memory and verify samples against the fixed split."""
    image_root = Path(dataset.image_dir).resolve()
    split_path = Path(dataset.split_path)
    original = json.loads(split_path.read_text())
    manifest = {"split_file": str(split_path),
                "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest()}
    groups = {"train": dataset.train_x, "val": dataset.val, "test": dataset.test}
    for group, items in groups.items():
        allowed = {str((image_root / entry[0]).resolve()): (int(entry[1]), entry[2])
                   for entry in original[group]}
        records = []
        for item in items:
            path = Path(item.impath)
            if str(path.resolve()) not in allowed:
                marker = "/{0}/{0}/".format(cfg.DATASET.NAME)
                if marker not in str(path):
                    raise ValueError("Cannot relocate cached image: {}".format(path))
                path = image_root / str(path).rsplit(marker, 1)[1]
            path = path.resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            if allowed.get(str(path)) != (item.label, item.classname):
                raise ValueError("Cached sample disagrees with {} split: {}".format(group, path))
            item._impath = str(path)
            records.append({"path": str(path.relative_to(image_root)),
                            "label": item.label, "classname": item.classname})
        manifest[group] = records
    train_paths = {r["path"] for r in manifest["train"]}
    test_paths = {r["path"] for r in manifest["test"]}
    val_paths = {r["path"] for r in manifest["val"]}
    if train_paths & test_paths or train_paths & val_paths or val_paths & test_paths:
        raise ValueError("Train/validation/test splits overlap")
    counts = Counter(item.label for item in dataset.train_x)
    if set(counts) != set(range(dataset.num_classes)):
        raise ValueError("Few-shot cache is missing one or more classes")
    if cfg.DATASET.NUM_SHOTS > 0 and any(n != cfg.DATASET.NUM_SHOTS for n in counts.values()):
        raise ValueError("Few-shot cache does not contain k samples per class: {}".format(counts))
    manifest["train_count_per_class"] = dict(counts)
    manifest["train_unique_per_class"] = dict(Counter(r["label"] for r in {
        r["path"]: r for r in manifest["train"]}.values()))
    manifest["sampling_note"] = "Existing DatasetBase repeats samples when a class has fewer than k images."
    return manifest


class PromptSRCPromptLearner(nn.Module):
    def __init__(self, backbone, classnames, options, method):
        super().__init__()
        self.method = method
        self.n_ctx = options.N_CTX if method == "MaPLe" else options.N_CTX_TEXT
        depth = options.PROMPT_DEPTH if method == "MaPLe" else options.PROMPT_DEPTH_TEXT
        vision_depth = depth if method == "MaPLe" else options.PROMPT_DEPTH_VISION
        if not 1 <= depth <= len(backbone.transformer.resblocks):
            raise ValueError("Text prompt depth is outside the backbone")
        if not 1 <= vision_depth <= len(backbone.visual.transformer.resblocks):
            raise ValueError("Vision prompt depth is outside the backbone")
        if not 1 <= self.n_ctx < backbone.context_length - 2:
            raise ValueError("Invalid text context length")
        text_width = backbone.ln_final.weight.numel()
        vision_width = backbone.visual.conv1.out_channels
        prefix = options.CTX_INIT.replace("_", " ")
        if prefix and self.n_ctx <= 4:
            with torch.no_grad():
                tokens = clip.tokenize(prefix).to(backbone.positional_embedding.device)
                ctx = backbone.token_embedding(tokens)[0, 1:1 + self.n_ctx].float().clone()
        else:
            prefix = " ".join(["X"] * self.n_ctx)
            ctx = torch.empty(self.n_ctx, text_width).normal_(std=0.02)
        self.ctx = nn.Parameter(ctx)
        self.deep_text = nn.ParameterList([
            nn.Parameter(torch.empty(self.n_ctx, text_width).normal_(std=0.02))
            for _ in range(depth - 1)
        ])
        if method == "MaPLe":
            self.proj = nn.Linear(text_width, vision_width)
            projection = nn.Linear(text_width, vision_width)
            self.deep_projections = nn.ModuleList([
                copy.deepcopy(projection) for _ in range(depth - 1)
            ])
        else:
            if options.N_CTX_VISION < 1:
                raise ValueError("Vision context length must be positive")
            self.vision = nn.ParameterList([
                nn.Parameter(torch.empty(options.N_CTX_VISION, vision_width).normal_(std=0.02))
                for _ in range(vision_depth)
            ])
        names = [name.replace("_", " ") for name in classnames]
        tokens = clip.tokenize([prefix + " " + name + "." for name in names])
        with torch.no_grad():
            embeddings = backbone.token_embedding(tokens.to(backbone.positional_embedding.device)).float()
        self.register_buffer("token_prefix", embeddings[:, :1], persistent=False)
        self.register_buffer("token_suffix", embeddings[:, 1 + self.n_ctx:], persistent=False)
        self.register_buffer("tokenized_prompts", tokens, persistent=False)

    def text_prompts(self):
        return torch.cat([self.token_prefix,
                          self.ctx.unsqueeze(0).expand(len(self.token_prefix), -1, -1),
                          self.token_suffix], dim=1)

    def vision_prompts(self):
        if self.method == "MaPLe":
            return [self.proj(self.ctx)] + [projection(prompt) for projection, prompt
                                            in zip(self.deep_projections, self.deep_text)]
        return list(self.vision)


class PromptSRCCLIP(nn.Module):
    def __init__(self, backbone, classnames, options, method):
        super().__init__()
        self.clip = backbone.requires_grad_(False).eval()
        self.method = method
        self.options = options
        self.prompt_learner = PromptSRCPromptLearner(backbone, classnames, options, method)
        self.register_buffer("teacher_text", torch.empty(0), persistent=False)
        self._eval_text = None

    @torch.no_grad()
    def initialize_teacher(self, classnames):
        # Official PromptSRC averages raw embeddings across its 60 templates,
        # then normalizes the mean. Compute these frozen targets in float32.
        names = [name.replace("_", " ") for name in classnames]
        features = []
        device = self.clip.positional_embedding.device
        for template in IMAGENET_TEMPLATES:
            tokens = clip.tokenize([template.format(name) for name in names]).to(device)
            features.append(self.clip.encode_text(tokens).float())
        self.teacher_text = F.normalize(torch.stack(features).mean(0), dim=-1)

    def train(self, mode=True):
        super().train(mode)
        self.clip.eval()
        self._eval_text = None
        return self

    def encode_prompted_text(self):
        learner, backbone = self.prompt_learner, self.clip
        x = learner.text_prompts().to(backbone.dtype) + backbone.positional_embedding.to(backbone.dtype)
        x = x.permute(1, 0, 2)
        for index, block in enumerate(backbone.transformer.resblocks):
            if 0 < index <= len(learner.deep_text):
                prompt = learner.deep_text[index - 1].to(x.dtype)
                prompt = prompt[:, None, :].expand(-1, x.shape[1], -1)
                x = torch.cat([x[:1], prompt, x[1 + learner.n_ctx:]], dim=0)
            x = block(x)
        x = backbone.ln_final(x.permute(1, 0, 2)).to(backbone.dtype)
        rows = torch.arange(x.shape[0], device=x.device)
        return x[rows, learner.tokenized_prompts.argmax(-1)] @ backbone.text_projection

    def encode_prompted_image(self, images):
        visual = self.clip.visual
        prompts = self.prompt_learner.vision_prompts()
        x = visual.conv1(images.to(self.clip.dtype))
        x = x.flatten(2).permute(0, 2, 1)
        cls = visual.class_embedding.to(x.dtype)[None, None, :].expand(x.shape[0], 1, -1)
        x = torch.cat([cls, x], dim=1) + visual.positional_embedding.to(x.dtype)
        # Official implementations append visual prompts AFTER patch positions.
        x = torch.cat([x, prompts[0].to(x.dtype)[None].expand(x.shape[0], -1, -1)], dim=1)
        x = visual.ln_pre(x).permute(1, 0, 2)
        n_ctx = prompts[0].shape[0]
        for index, block in enumerate(visual.transformer.resblocks):
            if 0 < index < len(prompts):
                prompt = prompts[index].to(x.dtype)[:, None, :].expand(-1, x.shape[1], -1)
                x = torch.cat([x[:-n_ctx], prompt], dim=0)
            x = block(x)
        x = visual.ln_post(x[0])
        return x @ visual.proj

    def forward(self, images, labels=None):
        if self.training or self._eval_text is None:
            text = F.normalize(self.encode_prompted_text().float(), dim=-1)
            if not self.training:
                self._eval_text = text.detach()
        else:
            text = self._eval_text
        image = F.normalize(self.encode_prompted_image(images).float(), dim=-1)
        scale = self.clip.logit_scale.exp().float()
        logits = scale * image @ text.t()
        if labels is None:
            return logits
        # Always evaluate losses in float32, including under autocast.
        with torch.autocast(device_type=images.device.type, enabled=False):
            logits = logits.float()
            losses = {"ce": F.cross_entropy(logits, labels)}
            if self.method == "PromptSRC":
                with torch.no_grad():
                    teacher_image = F.normalize(self.clip.encode_image(images.to(self.clip.dtype)).float(), dim=-1)
                    teacher_logits = scale * teacher_image @ self.teacher_text.t()
                losses["text_l1"] = self.options.TEXT_LOSS_WEIGHT * F.l1_loss(text, self.teacher_text)
                losses["image_l1"] = self.options.IMAGE_LOSS_WEIGHT * F.l1_loss(image, teacher_image)
                losses["logit_kl"] = F.kl_div(F.log_softmax(logits, dim=1),
                                               F.log_softmax(teacher_logits, dim=1),
                                               reduction="sum", log_target=True) / logits.numel()
            return sum(losses.values()), logits, losses


class PromptSRCTrainer(TrainerX):
    method = None

    def __init__(self, cfg):
        if cfg.DATASET.SUBSAMPLE_CLASSES != "all":
            raise ValueError("These few-shot trainers currently use all classes")
        self._experiment_cfg = cfg
        # Load the existing full split first. The repository's ordinary loader
        # would silently resample when encountering legacy Dassl pickles.
        data_cfg = cfg.clone()
        data_cfg.DATASET.NUM_SHOTS = -1
        super().__init__(data_cfg)

    def build_model(self):
        cfg = self._experiment_cfg
        self.cfg = cfg
        self.dm.cfg = cfg
        self.options = getattr(cfg.TRAINER, self.method.upper())
        if self.options.PREC not in ("fp16", "fp32", "amp"):
            raise ValueError("PREC must be fp16, fp32 or amp")
        if cfg.MODEL.BACKBONE.NAME not in ("ViT-B/16", "ViT-B/32"):
            raise ValueError("These trainers support CLIP ViT-B/16 and ViT-B/32")
        if cfg.OPTIM.MAX_EPOCH < 1:
            raise ValueError("MAX_EPOCH must be positive")
        cache = load_fewshot(cfg, self.dm.dataset)
        self.data_manifest = prepare_data(cfg, self.dm.dataset)
        self.data_manifest["fewshot_cache"] = str(cache)
        self.data_manifest["fewshot_sha256"] = hashlib.sha256(cache.read_bytes()).hexdigest()
        self._make_loaders()
        path = clip._download(clip._MODELS[cfg.MODEL.BACKBONE.NAME])
        jit = torch.jit.load(path, map_location="cpu").eval()
        backbone = clip.build_model(jit.state_dict()).float()
        del jit
        if cfg.INPUT.SIZE[0] != backbone.visual.input_resolution:
            raise ValueError("Input size does not match CLIP")
        self.model = PromptSRCCLIP(backbone, self.dm.dataset.classnames, self.options, self.method).to(self.device)
        if self.method == "PromptSRC":
            self.model.initialize_teacher(self.dm.dataset.classnames)
        if self.options.PREC == "fp16":
            if self.device.type != "cuda":
                raise ValueError("Use fp32 for CPU execution")
            # CLIP LayerNorm computes in float32 and must retain float32 weights.
            convert_weights(self.model.clip)
        # Explicit optimizer: the local dassl.optim does not honor CfgNode.
        self.optim = torch.optim.SGD(self.model.prompt_learner.parameters(), lr=float(cfg.OPTIM.LR),
                                     momentum=float(cfg.OPTIM.get("MOMENTUM", 0.9)),
                                     weight_decay=float(cfg.OPTIM.get("WEIGHT_DECAY", 0.0005)))
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.options.PREC == "amp" and self.device.type == "cuda")
        self.register_model("prompt_learner", self.model, self.optim)
        self.gpa = {}
        self.gpa_weights = None
        if self.method == "PromptSRC":
            sigma = float(self.options.GPA_STD)
            if sigma <= 0:
                raise ValueError("GPA_STD must be positive")
            epochs = torch.arange(1, cfg.OPTIM.MAX_EPOCH + 1, dtype=torch.float64)
            self.gpa_weights = torch.softmax(-0.5 * ((epochs - self.options.GPA_MEAN) / sigma).square(), dim=0)
        self.started = time.monotonic()
        count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info("%s: backbone=%s, trainable parameters=%s, classes=%s",
                     self.method, cfg.MODEL.BACKBONE.NAME, count, self.dm.dataset.classnames)

    def _make_loaders(self):
        cfg = self.cfg
        size = cfg.INPUT.SIZE[0]
        normalize = T.Normalize(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD)
        train_transform = T.Compose([T.RandomResizedCrop(size, interpolation=T.InterpolationMode.BICUBIC),
                                     T.RandomHorizontalFlip(), T.ToTensor(), normalize])
        # Retain the repository's square-resize test preprocessing.
        test_transform = T.Compose([T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
                                    T.ToTensor(), normalize])
        workers = int(cfg.DATALOADER.NUM_WORKERS)
        for split, items in (("train", self.dm.dataset.train_x), ("test", self.dm.dataset.test)):
            batch_size = cfg.DATALOADER.TRAIN_X.BATCH_SIZE if split == "train" else cfg.DATALOADER.TEST.BATCH_SIZE
            transform = train_transform if split == "train" else test_transform
            generator = torch.Generator().manual_seed(cfg.SEED)
            loader = DataLoader(DatasetWrapper(items, input_size=size, transform=transform, is_train=split == "train"),
                                batch_size=batch_size, shuffle=split == "train", num_workers=workers,
                                pin_memory=self.device.type == "cuda", drop_last=False,
                                persistent_workers=workers > 0, worker_init_fn=seed_worker, generator=generator)
            if not len(loader):
                raise ValueError("Empty {} loader".format(split))
            setattr(self.dm, "_{}_loader".format(split), loader)

    def autocast(self):
        if self.options.PREC == "amp" and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def before_train(self):
        if self.cfg.RESUME:
            raise ValueError("Use the batch runner to restart incomplete runs; mid-epoch resume is not supported")
        output = Path(self.cfg.OUTPUT_DIR)
        atomic_json(output / "config.json", self.cfg.to_dict())
        atomic_json(output / "data_manifest.json", self.data_manifest)

    def before_epoch(self):
        self.model.train()
        # Matches Dassl's ConstantWarmupScheduler + CosineAnnealingLR(T_max=N).
        warmup = int(self.cfg.OPTIM.get("WARMUP_EPOCH", 1))
        if self.epoch < warmup:
            lr = float(self.cfg.OPTIM.get("WARMUP_CONS_LR", 1e-5))
        else:
            lr = float(self.cfg.OPTIM.LR) * (1 + math.cos(math.pi * (self.epoch - warmup)
                                                         / self.cfg.OPTIM.MAX_EPOCH)) / 2
        for group in self.optim.param_groups:
            group["lr"] = lr

    def forward_backward(self, batch):
        images = batch["img"].to(self.device, non_blocking=True)
        labels = batch["label"].to(self.device, non_blocking=True)
        self.optim.zero_grad(set_to_none=True)
        with self.autocast():
            loss, logits, components = self.model(images, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optim)
        self.scaler.update()
        return {"loss": loss.item(), "acc": (logits.argmax(-1) == labels).float().mean().item() * 100,
                **{name: value.item() for name, value in components.items()}}

    def after_epoch(self):
        # Frozen weights are identical across epochs; average prompts only.
        if self.gpa_weights is not None:
            weight = float(self.gpa_weights[self.epoch])
            for name, parameter in self.model.prompt_learner.named_parameters():
                if name not in self.gpa:
                    self.gpa[name] = torch.zeros_like(parameter, dtype=torch.float32)
                self.gpa[name].add_(parameter.detach().float(), alpha=weight)

    def model_inference(self, images):
        with self.autocast():
            return self.model(images)

    def after_train(self):
        if self.gpa_weights is not None:
            with torch.no_grad():
                for name, parameter in self.model.prompt_learner.named_parameters():
                    parameter.copy_(self.gpa[name])
            logging.info("Using Gaussian parameter average for final evaluation")
        checkpoint = Path(self.cfg.OUTPUT_DIR) / "prompt_learner" / "model.pth.tar"
        atomic_checkpoint(checkpoint, {
            "state_dict": {k: v.detach().cpu() for k, v in self.model.prompt_learner.state_dict().items()},
            "epoch": self.cfg.OPTIM.MAX_EPOCH, "method": self.method,
            "backbone": self.cfg.MODEL.BACKBONE.NAME, "classnames": self.dm.dataset.classnames,
            "config": self.cfg.to_dict(), "gpa_applied": self.gpa_weights is not None,
        })
        # Let evaluation errors propagate. A failed evaluation is never complete.
        results = self.test()
        if not all(math.isfinite(value) for value in results.values()):
            raise FloatingPointError("Non-finite evaluation metric")
        results.update(status="complete", model=self.method, dataset=self.cfg.DATASET.NAME,
                       shots=self.cfg.DATASET.NUM_SHOTS, seed=self.cfg.SEED,
                       backbone=self.cfg.MODEL.BACKBONE.NAME, epochs=self.cfg.OPTIM.MAX_EPOCH,
                       precision=self.options.PREC, test_size=len(self.dm.dataset.test),
                       elapsed_seconds=time.monotonic() - self.started, checkpoint=str(checkpoint),
                       gpa_applied=self.gpa_weights is not None)
        atomic_json(Path(self.cfg.OUTPUT_DIR) / "results.json", results)

    def load_model(self, directory, epoch=None):
        path = Path(directory) / "prompt_learner" / "model.pth.tar"
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint["method"] != self.method or checkpoint["backbone"] != self.cfg.MODEL.BACKBONE.NAME:
            raise ValueError("Checkpoint method/backbone does not match the evaluation config")
        if epoch is not None and checkpoint["epoch"] != epoch:
            raise ValueError("Only the final checkpoint is retained")
        self.model.prompt_learner.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model._eval_text = None


@TRAINER_REGISTRY.register("PromptSRC_CLIP")
class PromptSRC(PromptSRCTrainer):
    method = "PromptSRC"
