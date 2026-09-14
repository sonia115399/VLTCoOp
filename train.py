import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer


import datasets

import trainers.Zeroshot.zeroshot
import trainers.CoOp.coop_clip
import trainers.CoCoOp.cocoop_clip
import trainers.KgCoOp.kgcoop_clip
import trainers.ProGrad.prograd_clip
import trainers.BiomedCoOp.biomedcoop_clip
import trainers.VLTCoOp.vltcoop_clip
import trainers.PromptSRC.promptsrc_clip
import trainers.MaPLe.maple_clip
from trainers.PromptSRC.promptsrc_clip import extend_multimodal_cfg


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed is not None:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head



def extend_cfg(cfg):
    """Add defaults required by VLTCoOp and comparison trainers."""
    extend_multimodal_cfg(cfg)
    try:
        has_field = hasattr(cfg.DATASET, 'SUBSAMPLE_CLASSES')
        current = cfg.DATASET.SUBSAMPLE_CLASSES if has_field else None
    except Exception:
        has_field = False
        current = None
    if (not has_field) or (current is None) or (str(current).strip() == ""):
        cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    vltcoop_defaults = {
        'CTX_INIT': "a photo of a",
        'CSC': False,
        'CLASS_TOKEN_POSITION': "end",
        'N_CTX': 4,
        'PREC': "fp32",
        'LAMBDA_SCV': 0.0,
        'LAMBDA_SCT': 1.0,
        'LAMBDA_KDV': 0.0,
        'LAMBDA_KDT': 1.0,
        'TEMPERATURE': 0.01,
        'MAD_THRESHOLD': 1.5,
        'N_CAPTIONS': 50,
    }

    if hasattr(cfg.TRAINER, 'VLTCOOP'):
        current = cfg.TRAINER.VLTCOOP
        # Configurations released before the dual-teacher formulation used
        # implementation-specific names. Keep them as migration aliases.
        aliases = {
            'VIS_SCCM_LAMBDA': 'LAMBDA_SCV',
            'SCCM_LAMBDA': 'LAMBDA_SCT',
            'VIS_KDSP_LAMBDA': 'LAMBDA_KDV',
            'KDSP_LAMBDA': 'LAMBDA_KDT',
            'TAU': 'MAD_THRESHOLD',
            'N_PROMPTS': 'N_CAPTIONS',
        }
        for old_name, new_name in aliases.items():
            if not hasattr(current, new_name) and hasattr(current, old_name):
                setattr(current, new_name, getattr(current, old_name))
        for k, v in vltcoop_defaults.items():
            if not hasattr(current, k):
                setattr(current, k, v)
    else:
        cfg.TRAINER.VLTCOOP = vltcoop_defaults

    cfg.TRAINER.COOP = {
        'N_CTX': 4,
        'CSC': False,
        'CTX_INIT': "",
        'PREC': "fp32",
        'CLASS_TOKEN_POSITION': "end",
    }
    cfg.TRAINER.COCOOP = {
        'N_CTX': 4,
        'CSC': False,
        'CTX_INIT': "",
        'PREC': "fp32",
        'CLASS_TOKEN_POSITION': "end",
    }
    biomedcoop_defaults = {
        'CTX_INIT': "a photo of a",
        'CSC': False,
        'CLASS_TOKEN_POSITION': "end",
        'N_CTX': 4,
        'PREC': "fp32",
        'SCCM_LAMBDA': 1.0,
        'KDSP_LAMBDA': 1.0,
        'VIS_SCCM_LAMBDA': 0.0,
        'VIS_KDSP_LAMBDA': 0.0,
        'TAU': 1.5,
        'N_PROMPTS': 50,
    }
    if hasattr(cfg.TRAINER, 'BIOMEDCOOP'):
        current = cfg.TRAINER.BIOMEDCOOP
        for key, value in biomedcoop_defaults.items():
            if not hasattr(current, key):
                setattr(current, key, value)
    else:
        cfg.TRAINER.BIOMEDCOOP = biomedcoop_defaults
    cfg.TRAINER.KGCOOP = {
        'CTX_INIT': "a photo of a",
        'CSC': False,
        'N_CTX': 4,
        'CLASS_TOKEN_POSITION': "end",
        'PREC': "fp32",
        'W': 1.0,
    }
    prograd_defaults = {
        'CTX_INIT': "a photo of a",
        'CSC': False,
        'CLASS_TOKEN_POSITION': "end",
        'N_CTX': 4,
        'PREC': "fp32",
        'GM': False,
        'NAME': "",
        'ALPHA': 0.0,
        'T': 1.0,
        'LAMBDA': 1.0,
    }
    if hasattr(cfg.TRAINER, 'PROGRAD'):
        current = cfg.TRAINER.PROGRAD
        for key, value in prograd_defaults.items():
            if not hasattr(current, key):
                setattr(current, key, value)
    else:
        cfg.TRAINER.PROGRAD = prograd_defaults

def setup_cfg(args):
    cfg = get_cfg_default()
    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    extend_cfg(cfg)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    cfg.EVAL_ONLY = args.eval_only
    trainer = build_trainer(cfg)
    print("Trainer built successfully.")

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
