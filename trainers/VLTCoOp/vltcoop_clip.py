import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import random

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.metrics import compute_accuracy
from trainers.vltcoop_templates import VLTCOOP_TEMPLATES
import json
import os

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

    model = clip.build_model(state_dict or model.state_dict())

    return model


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
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.VLTCOOP.N_CTX
        ctx_init = cfg.TRAINER.VLTCOOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and n_ctx <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        self.ctx = nn.Parameter(ctx_vectors)

        # classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        # Create frozen CLIP teachers for language and visual supervision.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clip_model_temp = load_clip_to_cpu(cfg).float().to(device)
        clip_model_temp_image = load_clip_to_cpu(cfg).float().to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            self.ZS_image_encoder = clip_model_temp_image.visual
            # Now pre-compute the frozen VL embeddings
            all_teacher_features = []

            # Load optional VLM-generated class descriptions from the dataset tree.
            gen_prompts = {}
            try:
                dataset_root = getattr(cfg.DATASET, 'ROOT', None)
                dataset_name = getattr(cfg.DATASET, 'NAME', None)
                load_path = None

                def try_generated_prompt_base(base_dir):
                    nonlocal gen_prompts, load_path
                    cand1 = os.path.join(base_dir, 'generated_prompts.json')
                    if os.path.isfile(cand1):
                        with open(cand1, 'r', encoding='utf-8') as f:
                            gen_prompts = json.load(f)
                            load_path = cand1
                            return True
                    cand2_dir = os.path.join(base_dir, 'generated_prompts')
                    if os.path.isdir(cand2_dir):
                        label_for_file = dataset_name or os.path.basename(os.path.normpath(base_dir))
                        if label_for_file:
                            cand_ds = os.path.join(cand2_dir, f"{label_for_file}.json")
                            if os.path.isfile(cand_ds):
                                with open(cand_ds, 'r', encoding='utf-8') as f:
                                    gen_prompts = json.load(f)
                                    load_path = cand_ds
                                    return True
                        merged = {}
                        for fn in os.listdir(cand2_dir):
                            if fn.lower().endswith('.json'):
                                path = os.path.join(cand2_dir, fn)
                                try:
                                    with open(path, 'r', encoding='utf-8') as f:
                                        data = json.load(f)
                                        if isinstance(data, dict):
                                            merged.update(data)
                                except Exception:
                                    continue
                        if merged:
                            gen_prompts = merged
                            load_path = cand2_dir
                            return True
                    return False

                if dataset_root:
                    dirs_to_check = []
                    if dataset_name:
                        ds_dir = os.path.abspath(os.path.join(dataset_root, dataset_name))
                        if os.path.isdir(ds_dir):
                            dirs_to_check.append(ds_dir)
                    for up in range(0, 3):
                        dir_to_check = os.path.abspath(os.path.join(dataset_root, *(['..'] * up)))
                        if dir_to_check not in dirs_to_check:
                            dirs_to_check.append(dir_to_check)
                    for base_dir in dirs_to_check:
                        if try_generated_prompt_base(base_dir):
                            break
                if load_path:
                    try:
                        # prefer logging when available; otherwise print
                        import logging
                        logging.info(f"Loaded generated prompts from {load_path}")
                    except Exception:
                        print(f"Loaded generated prompts from {load_path}")
            except Exception:
                gen_prompts = {}

            def prompts_for_class(name):
                # try exact key
                if name in gen_prompts:
                    return gen_prompts[name]
                # try replacing spaces/underscores
                k = name.replace(' ', '_')
                if k in gen_prompts:
                    return gen_prompts[k]
                k2 = name.replace('_', ' ')
                if k2 in gen_prompts:
                    return gen_prompts[k2]
                # fallback empty
                return None

            prompt_pool = {}
            prompt_origin = {}
            resolved_prompt_usage = {classname: [] for classname in classnames}

            for classname in classnames:
                gp = prompts_for_class(classname)
                if gp and len(gp) > 0:
                    prompt_pool[classname] = gp
                    prompt_origin[classname] = "generated"
                else:
                    tmpl = VLTCOOP_TEMPLATES.get(classname, None)
                    if tmpl is None:
                        tmpl = VLTCOOP_TEMPLATES.get(classname.replace(' ', '_'), None)
                    if tmpl is None:
                        raise KeyError(f"No template or generated prompts for class '{classname}'")
                    prompt_pool[classname] = tmpl
                    prompt_origin[classname] = "template"

            for i in range(cfg.TRAINER.VLTCOOP.N_PROMPTS):
                tokens = []
                for classname in classnames:
                    pool = prompt_pool[classname]
                    txt = pool[i % len(pool)]
                    resolved_prompt_usage[classname].append(txt)
                    tokens.append(clip.tokenize(txt))
                x_tokenized = torch.cat(tokens)
                text_features = clip_model_temp.encode_text(x_tokenized.to(device))
                all_teacher_features.append(text_features.unsqueeze(1))

        self.fixed_embeddings = torch.cat(all_teacher_features, dim=1)
        try:
            import logging
            log = logging.getLogger(__name__)
            log_fn = log.info if log.handlers else print
        except Exception:
            log_fn = print

        log_fn("Resolved prompts per class:")
        for classname in classnames:
            origin = prompt_origin.get(classname, 'template')
            log_fn(f"  {classname} (source: {origin}):")
            for idx, text in enumerate(resolved_prompt_usage.get(classname, [])):
                log_fn(f"    [{idx:02d}] {text}")
        # These buffers are reconstructed from the current class names on load.
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, visual_class_means):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.n_cls = len(classnames)
        self.cfg = cfg
        self.register_buffer("visual_class_means", visual_class_means)

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts = self.prompt_learner()

        # Compute the prompted image and text features
        text_features = self.text_encoder(prompts, tokenized_prompts)
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # Compute the prompted logits
        logits = logit_scale * image_features @ text_features.t()
        if self.prompt_learner.training:
            
            # Visual Teacher Branch
            visual_class_means = self.visual_class_means
            proto_logits = logit_scale * image_features @ visual_class_means.t()
            #proto_logits = logit_scale * text_features @ visual_class_means.t()
            # Now calculate the frozen pre-trained features
            fixed_embeddings = self.prompt_learner.fixed_embeddings  # precomputed pre-trained frozen textual features
            fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)
            with torch.no_grad():
                zero_shot_features = self.prompt_learner.ZS_image_encoder(image.type(self.dtype))
                zero_shot_features = zero_shot_features / zero_shot_features.norm(dim=-1, keepdim=True)

                scores = []
                for i in range(fixed_embeddings.shape[1]):
                    temp_logits = logit_scale * visual_class_means @ fixed_embeddings[:, i, :].to(visual_class_means.device).t()
                    scores.append(torch.max(temp_logits, dim=1).values.mean())

                scores = torch.stack(scores)
                median = scores.median()
                mad = (scores - median).abs().median()
                robust_z = (scores - median) / mad.clamp_min(torch.finfo(scores.dtype).eps)
                tau = self.cfg.TRAINER.VLTCOOP.TAU
                spread = robust_z.std(unbiased=False).clamp_min(torch.finfo(scores.dtype).eps)
                mask = ((robust_z - robust_z.mean()).abs() / spread) <= tau
                if not mask.any():
                    mask = torch.ones_like(mask, dtype=torch.bool)
                selected_embeddings = fixed_embeddings[:,mask].mean(dim=1)
                selected_embeddings = selected_embeddings / selected_embeddings.norm(dim=-1, keepdim=True)
                
            fixed_embeddings = fixed_embeddings.mean(dim=1)
            fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)
            zero_shot_logits = logit_scale * zero_shot_features @ selected_embeddings.to(zero_shot_features.device).t()
            loss_ce = F.cross_entropy(logits,
                                   label)
            
            loss_mse = torch.nn.MSELoss()
            loss_sccm = loss_mse(text_features, selected_embeddings.to(text_features.device)) * self.cfg.TRAINER.VLTCOOP.SCCM_LAMBDA

            loss_kdsp = F.kl_div(
                F.log_softmax(logits, dim=1),
                F.log_softmax(zero_shot_logits, dim=1),
                reduction='sum',
                log_target=True
            ) / logits.numel()
            loss_kdsp = loss_kdsp * self.cfg.TRAINER.VLTCOOP.KDSP_LAMBDA
            
            # Visual Losses
            # Visual SCCM: MSE between text_features (learnable) and visual_class_means (frozen)
            loss_sccm_vis = loss_mse(text_features, visual_class_means) * getattr(self.cfg.TRAINER.VLTCOOP, 'VIS_SCCM_LAMBDA', 0.0)
            
            # Visual KDSP: KL Div between logits (student) and proto_logits (teacher)
            loss_kdsp_vis = F.kl_div(
                F.log_softmax(logits, dim=1),
                F.log_softmax(proto_logits, dim=1),
                reduction='sum',
                log_target=True
            ) / logits.numel()
            loss_kdsp_vis = loss_kdsp_vis * getattr(self.cfg.TRAINER.VLTCOOP, 'VIS_KDSP_LAMBDA', 0.0)

            return logits, loss_ce, loss_sccm, loss_kdsp, loss_sccm_vis, loss_kdsp_vis
        else:
            return logits


@TRAINER_REGISTRY.register()
class VLTCoOp_CLIP(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.VLTCOOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.VLTCOOP.PREC == "fp32" or cfg.TRAINER.VLTCOOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Computing visual prototypes...")
        
        # Preserve stochastic training behavior while constructing frozen prototypes.
        rng_state = torch.get_rng_state()
        if torch.cuda.is_available():
            cuda_rng_state = torch.cuda.get_rng_state()
        np_rng_state = np.random.get_state()
        py_rng_state = random.getstate()

        visual_features = []
        labels = []
        
        clip_model.to(self.device)
        
        with torch.no_grad():
            for batch in self.dm.train_loader:
                input = batch["img"].to(self.device)
                label = batch["label"].to(self.device)
                image_features = clip_model.visual(input.type(clip_model.dtype))
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                visual_features.append(image_features)
                labels.append(label)
        
        # Restore RNG state
        torch.set_rng_state(rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(cuda_rng_state)
        np.random.set_state(np_rng_state)
        random.setstate(py_rng_state)

        visual_features = torch.cat(visual_features, dim=0)
        labels = torch.cat(labels, dim=0)
        
        visual_class_means = []
        for i in range(len(classnames)):
            idx = (labels == i).nonzero(as_tuple=True)[0]
            if len(idx) == 0:
                mean_feat = torch.zeros(visual_features.shape[1], device=self.device, dtype=visual_features.dtype)
            else:
                mean_feat = visual_features[idx].mean(dim=0)
            mean_feat = mean_feat / mean_feat.norm()
            visual_class_means.append(mean_feat)
            
        visual_class_means = torch.stack(visual_class_means)
        print(f"Computed visual class prototypes for {len(visual_class_means)} classes")
        
        clip_model.to("cpu")

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model.eval(), visual_class_means)

        print("Turning off gradients in both the image and the text encoder")
        names_to_update = ["prompt_learner.ctx"]

        for name, param in self.model.named_parameters():
            if name not in names_to_update:
                param.requires_grad_(False)


        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        print(f"Parameters count: {len(enabled)}")
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model, self.optim, self.sched)
        # Cosine scheduler
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.step_counter = 1
        self.scaler = GradScaler() if cfg.TRAINER.VLTCOOP.PREC == "amp" else None
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.VLTCOOP.PREC
        if prec == "amp":
            with autocast():
                logits, loss_ce, loss_sccm, loss_kdsp, loss_sccm_vis, loss_kdsp_vis = model(image, label)
                loss = loss_ce + loss_sccm + loss_kdsp + loss_sccm_vis + loss_kdsp_vis
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            logits, loss_ce, loss_sccm, loss_kdsp, loss_sccm_vis, loss_kdsp_vis = model(image, label)
            
            loss = loss_ce + loss_sccm + loss_kdsp + loss_sccm_vis + loss_kdsp_vis
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(logits, label)[0].item(),
            "loss_ce": loss_ce.item(),
            "loss_sccm": loss_sccm.item(),
            "loss_kdsp": loss_kdsp.item(),
            "loss_sccm_vis": loss_sccm_vis.item(),
            "loss_kdsp_vis": loss_kdsp_vis.item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # Build candidate list in priority order
        candidates = []
        if epoch is not None:
            candidates.append(f"model.pth.tar-{epoch}")
            candidates.append("model.pth.tar")
        else:
            # Prefer rolling latest checkpoint if available, then fall back to best
            candidates.append("model.pth.tar")
            candidates.append("model-best.pth.tar")

        for name in names:
            model_path = None
            for cand in candidates:
                p = osp.join(directory, name, cand)
                if osp.exists(p):
                    model_path = p
                    break
            if model_path is None:
                # Compose an informative error
                raise FileNotFoundError(
                    'No checkpoint found under "{}" (tried: {})'.format(
                        osp.join(directory, name), ", ".join(candidates)
                    )
                )

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            # Filter out keys with shape mismatch (e.g., class-dependent buffers)
            current_state = self._models[name].state_dict()
            filtered_state = {}
            skipped = []
            for k, v in state_dict.items():
                if k in current_state:
                    try:
                        if tuple(current_state[k].shape) == tuple(v.shape):
                            filtered_state[k] = v
                        else:
                            skipped.append((k, tuple(v.shape), tuple(current_state[k].shape)))
                    except Exception:
                        # If shape is not available/comparable, skip conservatively
                        skipped.append((k, None, None))
                else:
                    # Key not in current model; safe to skip under strict=False
                    continue

            if skipped:
                msg = "; ".join([f"{kk} ckpt_shape={cs} current_shape={ms}" for kk, cs, ms in skipped])
                print(f"[load_model] Skipping {len(skipped)} mismatched keys: {msg}")

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # Load only compatible keys with strict=False to allow missing ones
            self._models[name].load_state_dict(filtered_state, strict=False)
