import json
import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import load_checkpoint, load_pretrained_weights
from trainers.vltcoop_templates import VLTCOOP_TEMPLATES

from clip import clip


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    model_path = clip._download(clip._MODELS[backbone_name])
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    return clip.build_model(state_dict or model.state_dict())


def _normalized_mean(features):
    """Average feature vectors and normalize the resulting class prototype."""
    return F.normalize(features.mean(dim=0), dim=0)


def select_caption_groups(caption_features, visual_prototypes, mad_threshold, eps=1e-6):
    """Apply the paper's group-wise MAD filter to CLIP caption embeddings.

    ``caption_features[i, j]`` is caption group ``j`` for class ``i``. A group
    is retained or rejected jointly for all classes, exactly as defined by s_j.
    The returned language prototype deliberately remains a mean of normalized
    captions; cosine-based distillation normalizes it only when scoring logits.
    """
    if caption_features.ndim != 3:
        raise ValueError("caption_features must have shape [classes, captions, dimension]")
    if caption_features.shape[0] != visual_prototypes.shape[0]:
        raise ValueError("caption and visual prototype class counts must match")

    caption_features = F.normalize(caption_features, dim=-1)
    visual_prototypes = F.normalize(visual_prototypes, dim=-1)
    scores = (caption_features * visual_prototypes[:, None, :]).sum(dim=-1).mean(dim=0)
    median = scores.median()
    mad = (scores - median).abs().median()
    robust_z = (scores - median) / (mad + eps)
    mask = robust_z.abs() <= mad_threshold

    # A very small threshold or degenerate score distribution must still leave
    # one valid caption group for every class.
    if not mask.any():
        mask[scores.sub(median).abs().argmin()] = True

    language_prototypes = caption_features[:, mask, :].mean(dim=1)
    return language_prototypes, mask, scores


def _find_caption_file(cfg):
    """Find generated_prompts.json without coupling the trainer to one layout."""
    root = getattr(cfg.DATASET, "ROOT", "")
    dataset_name = getattr(cfg.DATASET, "NAME", "")
    if not root:
        return None

    candidates = []
    if dataset_name:
        candidates.append(osp.join(root, dataset_name, "generated_prompts.json"))
    candidates.append(osp.join(root, "generated_prompts.json"))
    candidates.append(osp.join(root, "generated_prompts", f"{dataset_name}.json"))
    for candidate in candidates:
        if osp.isfile(candidate):
            return candidate
    return None


def _load_caption_pools(cfg, classnames):
    path = _find_caption_file(cfg)
    generated = {}
    if path:
        with open(path, "r", encoding="utf-8") as handle:
            generated = json.load(handle)
        if not isinstance(generated, dict):
            raise ValueError(f"Caption file must contain a class-to-captions mapping: {path}")
        print(f"Loaded LLaVA captions from {path}")
    else:
        print("No generated_prompts.json found; using curated descriptions as language-teacher fallback.")

    pools = []
    origins = []
    for classname in classnames:
        alternatives = (classname, classname.replace(" ", "_"), classname.replace("_", " "))
        captions = next((generated[key] for key in alternatives if key in generated), None)
        if isinstance(captions, dict):
            captions = captions.get("captions")
        if isinstance(captions, str):
            captions = [captions]
        captions = [caption.strip() for caption in (captions or []) if isinstance(caption, str) and caption.strip()]
        if captions:
            pools.append(captions)
            origins.append("generated")
            continue

        templates = VLTCOOP_TEMPLATES.get(classname)
        if templates is None:
            templates = VLTCOOP_TEMPLATES.get(classname.replace(" ", "_"))
        if not templates:
            raise KeyError(f"No generated captions or curated descriptions for class '{classname}'")
        pools.append(templates)
        origins.append("template")

    return pools, origins


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = self.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        indices = torch.arange(x.shape[0], device=x.device)
        return x[indices, tokenized_prompts.argmax(dim=-1)] @ self.text_projection


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, visual_prototypes, build_teachers=True):
        super().__init__()
        self.n_cls = len(classnames)
        self.n_ctx = cfg.TRAINER.VLTCOOP.N_CTX
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        if cfg.INPUT.SIZE[0] != clip_model.visual.input_resolution:
            raise ValueError("INPUT.SIZE must equal the CLIP visual input resolution")

        ctx_init = cfg.TRAINER.VLTCOOP.CTX_INIT.replace("_", " ")
        if ctx_init and self.n_ctx <= 4:
            tokenized_init = clip.tokenize(ctx_init)
            with torch.no_grad():
                ctx_vectors = clip_model.token_embedding(tokenized_init).type(dtype)[0, 1:1 + self.n_ctx]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(self.n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * self.n_ctx)
        self.ctx = nn.Parameter(ctx_vectors)
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words: {self.n_ctx}")

        prompt_texts = [f"{prompt_prefix} {name.replace('_', ' ')}." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(text) for text in prompt_texts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx:, :])
        self.register_buffer("tokenized_prompts", tokenized_prompts)

        feature_dim = visual_prototypes.shape[-1]
        language_prototypes = torch.zeros(self.n_cls, feature_dim, dtype=visual_prototypes.dtype)
        if build_teachers:
            pools, origins = _load_caption_pools(cfg, classnames)
            n_captions = cfg.TRAINER.VLTCOOP.N_CAPTIONS
            caption_texts = [
                pool[group_index % len(pool)]
                for group_index in range(n_captions)
                for pool in pools
            ]
            with torch.no_grad():
                tokens = torch.cat([clip.tokenize(text) for text in caption_texts])
                features = F.normalize(clip_model.encode_text(tokens).float(), dim=-1)
            caption_features = features.reshape(n_captions, self.n_cls, -1).permute(1, 0, 2)
            caption_features = caption_features.to(visual_prototypes.dtype)
            language_prototypes, mask, scores = select_caption_groups(
                caption_features,
                visual_prototypes.cpu(),
                cfg.TRAINER.VLTCOOP.MAD_THRESHOLD,
            )
            print(
                f"Language teacher: retained {int(mask.sum())}/{n_captions} caption groups "
                f"with MAD threshold {cfg.TRAINER.VLTCOOP.MAD_THRESHOLD}."
            )
            for classname, origin in zip(classnames, origins):
                print(f"  {classname}: {origin}")
            self.register_buffer("caption_group_scores", scores, persistent=False)
            self.register_buffer("caption_group_mask", mask, persistent=False)
        else:
            self.register_buffer("caption_group_scores", torch.empty(0), persistent=False)
            self.register_buffer("caption_group_mask", torch.empty(0, dtype=torch.bool), persistent=False)

        # Targets are rebuilt from the current support set at training startup
        # and intentionally omitted from checkpoints/inference construction.
        self.register_buffer("language_prototypes", language_prototypes, persistent=False)

    def forward(self):
        ctx = self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        return torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, visual_prototypes, build_teachers=True):
        super().__init__()
        self.cfg = cfg
        self.prompt_learner = PromptLearner(
            cfg, classnames, clip_model, visual_prototypes, build_teachers=build_teachers
        )
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.dtype = clip_model.dtype
        self.temperature = float(cfg.TRAINER.VLTCOOP.TEMPERATURE)
        if self.temperature <= 0:
            raise ValueError("TRAINER.VLTCOOP.TEMPERATURE must be positive")
        self.register_buffer("visual_prototypes", visual_prototypes, persistent=False)

    def _distribution_logits(self, image_features, class_features):
        return image_features @ F.normalize(class_features, dim=-1).t() / self.temperature

    def forward(self, image, label=None):
        prompts = self.prompt_learner()
        text_features = F.normalize(self.text_encoder(prompts, self.prompt_learner.tokenized_prompts), dim=-1)
        image_features = F.normalize(self.image_encoder(image.type(self.dtype)), dim=-1)
        student_logits = self._distribution_logits(image_features, text_features)

        if not self.training:
            return student_logits
        if label is None:
            raise ValueError("Labels are required while training VLTCoOp")

        visual_prototypes = F.normalize(self.visual_prototypes, dim=-1)
        language_prototypes = self.prompt_learner.language_prototypes
        vision_teacher_logits = self._distribution_logits(image_features, visual_prototypes)
        language_teacher_logits = self._distribution_logits(image_features, language_prototypes)

        loss_ce = F.cross_entropy(student_logits, label)
        loss_scv = F.mse_loss(text_features, visual_prototypes)
        loss_sct = F.mse_loss(text_features, language_prototypes)
        loss_kdv = F.kl_div(
            F.log_softmax(student_logits, dim=1), F.softmax(vision_teacher_logits.detach(), dim=1), reduction="batchmean"
        )
        loss_kdt = F.kl_div(
            F.log_softmax(student_logits, dim=1), F.softmax(language_teacher_logits.detach(), dim=1), reduction="batchmean"
        )

        weights = self.cfg.TRAINER.VLTCOOP
        return (
            student_logits,
            loss_ce,
            weights.LAMBDA_SCV * loss_scv,
            weights.LAMBDA_SCT * loss_sct,
            weights.LAMBDA_KDV * loss_kdv,
            weights.LAMBDA_KDT * loss_kdt,
        )


@TRAINER_REGISTRY.register()
class VLTCoOp_CLIP(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.VLTCOOP.PREC in ["fp16", "fp32", "amp"]

    def _build_visual_prototypes(self, clip_model, classnames):
        print("Computing frozen vision-teacher prototypes from the support set...")
        features, labels = [], []
        with torch.no_grad():
            for batch in self.dm.train_loader:
                images = batch["img"].to(self.device)
                feature = F.normalize(clip_model.visual(images.type(clip_model.dtype)), dim=-1)
                features.append(feature)
                labels.append(batch["label"].to(self.device))
        features = torch.cat(features)
        labels = torch.cat(labels)
        prototypes = []
        for class_index in range(len(classnames)):
            class_features = features[labels == class_index]
            if class_features.numel() == 0:
                raise ValueError(f"Support set has no image for class index {class_index}")
            prototypes.append(_normalized_mean(class_features))
        return torch.stack(prototypes)

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.VLTCOOP.PREC in ["fp32", "amp"]:
            clip_model.float()
        clip_model.eval().to(self.device)

        build_teachers = not getattr(cfg, "EVAL_ONLY", False)
        if build_teachers:
            visual_prototypes = self._build_visual_prototypes(clip_model, classnames)
        else:
            # Evaluation uses only the learned context and frozen CLIP encoders.
            feature_dim = clip_model.text_projection.shape[-1]
            visual_prototypes = torch.zeros(len(classnames), feature_dim, device=self.device, dtype=clip_model.dtype)
            print("Evaluation-only mode: skipped teacher and caption construction.")

        clip_model.to("cpu")
        self.model = CustomCLIP(cfg, classnames, clip_model.eval(), visual_prototypes.cpu(), build_teachers)
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(name == "prompt_learner.ctx")
        enabled = {name for name, parameter in self.model.named_parameters() if parameter.requires_grad}
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.VLTCOOP.PREC == "amp" else None
        if torch.cuda.device_count() > 1:
            print(f"Multiple GPUs detected (n_gpus={torch.cuda.device_count()}), using DataParallel.")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        if self.cfg.TRAINER.VLTCOOP.PREC == "amp":
            with autocast():
                logits, loss_ce, loss_scv, loss_sct, loss_kdv, loss_kdt = self.model(image, label)
                loss = loss_ce + loss_scv + loss_sct + loss_kdv + loss_kdt
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            logits, loss_ce, loss_scv, loss_sct, loss_kdv, loss_kdt = self.model(image, label)
            loss = loss_ce + loss_scv + loss_sct + loss_kdv + loss_kdt
            self.model_backward_and_update(loss)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        return {
            "loss": loss.item(),
            "acc": compute_accuracy(logits, label)[0].item(),
            "loss_ce": loss_ce.item(),
            "loss_scv": loss_scv.item(),
            "loss_sct": loss_sct.item(),
            "loss_kdv": loss_kdv.item(),
            "loss_kdt": loss_kdt.item(),
        }

    def parse_batch_train(self, batch):
        return batch["img"].to(self.device), batch["label"].to(self.device)

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return
        candidates = [f"model.pth.tar-{epoch}", "model.pth.tar"] if epoch is not None else ["model.pth.tar", "model-best.pth.tar"]
        for name in self.get_model_names():
            model_path = next((osp.join(directory, name, candidate) for candidate in candidates if osp.isfile(osp.join(directory, name, candidate))), None)
            if model_path is None:
                raise FileNotFoundError(f"No checkpoint found under {osp.join(directory, name)}")
            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            # Prompt token buffers depend on the active classes and are rebuilt.
            state_dict.pop("prompt_learner.token_prefix", None)
            state_dict.pop("prompt_learner.token_suffix", None)
            current = self._models[name].state_dict()
            compatible = {key: value for key, value in state_dict.items() if key in current and tuple(value.shape) == tuple(current[key].shape)}
            print(f"Loading weights to {name} from {model_path} (epoch = {checkpoint['epoch']})")
            self._models[name].load_state_dict(compatible, strict=False)
