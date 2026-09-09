"""Minimal DataManager that uses repository's `datasets` package."""
import logging
import torch
from datasets import build_dataset
from datasets.utils import build_data_loader

class DataManager:
    def __init__(self, cfg):
        # Build dataset object using project's datasets.build_dataset(cfg)
        self.dataset = build_dataset(cfg)
        # Build data loaders lazily
        self._train_loader = None
        self._val_loader = None
        self._test_loader = None
        self.cfg = cfg

    @property
    def train_loader(self):
        if self._train_loader is None:
            # Determine batch size from multiple possible config locations.
            # Prefer DATALOADER.TRAIN_X.BATCH_SIZE (used in trainer YAMLs),
            # fall back to SOLVER.IMS_PER_BATCH or top-level BATCH_SIZE.
            try:
                batch_size = int(self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE)
            except Exception:
                batch_size = self.cfg.get('SOLVER', {}).get('IMS_PER_BATCH', self.cfg.get('BATCH_SIZE', 64))

            self._train_loader = build_data_loader(
                data_source=self.dataset.train_x,
                batch_size=batch_size,
                input_size=self.cfg.get('INPUT', {}).get('SIZE', [224])[0],
                tfm=None,
                is_train=True,
                shuffle=True,
            )
            # Log runtime info to help debug mismatches between config and
            # the actual DataLoader used during training (e.g. episodic
            # samplers that collapse an epoch into a single batch).
            log = logging.getLogger()
            try:
                dataset_len = len(self._train_loader.dataset)
            except Exception:
                dataset_len = 'N/A'
            try:
                num_batches = len(self._train_loader)
            except Exception:
                num_batches = 'N/A'
            sampler = type(getattr(self._train_loader, 'sampler', None)).__name__
            log.info('Built train_loader: dataset_len=%s, num_batches=%s, sampler=%s', dataset_len, num_batches, sampler)
        return self._train_loader

    @property
    def val_loader(self):
        if self._val_loader is None:
            try:
                batch_size = int(self.cfg.DATALOADER.TEST.BATCH_SIZE)
            except Exception:
                batch_size = self.cfg.get('BATCH_SIZE', 64)

            self._val_loader = build_data_loader(
                data_source=self.dataset.val,
                batch_size=batch_size,
                input_size=self.cfg.get('INPUT', {}).get('SIZE', [224])[0],
                tfm=None,
                is_train=False,
                shuffle=False,
            )
        return self._val_loader

    @property
    def test_loader(self):
        if self._test_loader is None:
            try:
                batch_size = int(self.cfg.DATALOADER.TEST.BATCH_SIZE)
            except Exception:
                batch_size = self.cfg.get('BATCH_SIZE', 64)

            self._test_loader = build_data_loader(
                data_source=self.dataset.test,
                batch_size=batch_size,
                input_size=self.cfg.get('INPUT', {}).get('SIZE', [224])[0],
                tfm=None,
                is_train=False,
                shuffle=False,
            )
        return self._test_loader

    def get_num_classes(self):
        return self.dataset.num_classes
