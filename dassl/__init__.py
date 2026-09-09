"""Lightweight shim implementing a minimal subset of the original `dassl`
API used by this project. The goal is to avoid depending on the external
`Dassl.pytorch` package while providing the small surface area trainers expect.

This package implements minimal versions of:
- config.get_cfg_default
- engine.REGISTRY / TRAINER_REGISTRY / build_trainer / TrainerX
- data.DataManager
- utils (logging, seeding, checkpoints)
- optim (simple optimizer & scheduler builders)
- metrics.compute_accuracy

This is intentionally small and focused on getting the project's training
scripts working. It doesn't try to reimplement the full Dassl feature set.
"""

from . import engine, config, utils, data, optim, metrics

__all__ = ["engine", "config", "utils", "data", "optim", "metrics"]
