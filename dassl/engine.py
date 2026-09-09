"""Minimal engine: registry and a simple TrainerX base used by trainers.

This is intentionally small and implements the minimal lifecycle to let
`train.py` call `build_trainer(cfg)` and run `trainer.train()`.
"""
import types
import inspect
import time
import logging
import torch
import os
import os.path as osp
from collections import OrderedDict
try:
    from sklearn.metrics import f1_score
except Exception:
    f1_score = None


class Registry:
    def __init__(self, name):
        self._name = name
        self._dict = {}

    def register(self, name=None):
        def _register(cls):
            key = name or cls.__name__
            self._dict[key] = cls
            return cls
        return _register

    def registered_names(self):
        return list(self._dict.keys())

    def get(self, name):
        return self._dict.get(name)


TRAINER_REGISTRY = Registry('TRAINER')


def build_trainer(cfg):
    name = cfg.get('TRAINER', {}).get('NAME', '')
    if name == '':
        raise ValueError('No trainer name specified in cfg.TRAINER.NAME')
    trainer_cls = TRAINER_REGISTRY.get(name)
    if trainer_cls is None:
        raise ValueError(f'Trainer {name} is not registered. Available: {TRAINER_REGISTRY.registered_names()}')
    return trainer_cls(cfg)


class TrainerX:
    """A simplified trainer base class.

    Subclasses (in `trainers/`) are expected to implement at least:
    - build_model(self)
    - forward_backward(self, batch)
    - parse_batch_train(self, batch)
    - (optional) load_model(self, directory, epoch=None)
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() and cfg.get('USE_CUDA', True) else 'cpu')
        # Lazy attributes expected by many trainers
        self._models = {}
        self._optims = {}
        self._scheds = {}
        self.dm = None
        # Trainer-specific fields
        self.epoch = 0
        self.batch_idx = 0
        self.num_batches = 0
        # Allow subclass to build datamanager and model
        # Subclasses should call self.register_model() in build_model().
        # Many trainers expect a DataManager assigned to self.dm
        from dassl.data import DataManager
        self.dm = DataManager(cfg)
        self.build_model()

    def build_model(self):
        # to be implemented by subclass
        pass

    def register_model(self, name, model, optim=None, sched=None):
        self._models[name] = model
        self._optims[name] = optim
        self._scheds[name] = sched

    def get_model_names(self):
        return list(self._models.keys())

    def model_backward_and_update(self, loss, names=None):
        if names is None:
            names = self.get_model_names()
        for name in names:
            if self._optims.get(name) is not None:
                self._optims[name].zero_grad()
        loss.backward()
        for name in names:
            if self._optims.get(name) is not None:
                self._optims[name].step()

    def update_lr(self):
        for name, sched in self._scheds.items():
            if sched is not None:
                try:
                    sched.step()
                except Exception:
                    pass

    def train(self):
        # Simple training loop with per-iteration logging
        max_epoch = self.cfg.get('OPTIM', {}).get('MAX_EPOCH', 1)
        log = logging.getLogger()
        self.before_train()
        for epoch in range(max_epoch):
            self.epoch = epoch
            self.before_epoch()
            # iterate over train loader
            loader = self.dm.train_loader
            self.num_batches = len(loader)

            # simple meters
            class AverageMeter:
                def __init__(self):
                    self.reset()
                def reset(self):
                    self.val = 0
                    self.avg = 0
                    self.sum = 0
                    self.count = 0
                def update(self, val, n=1):
                    self.val = val
                    self.sum += val * n
                    self.count += n
                    self.avg = self.sum / self.count if self.count != 0 else 0

            batch_time_meter = AverageMeter()
            data_time_meter = AverageMeter()
            loss_meter = AverageMeter()
            acc_meter = AverageMeter()

            end = time.time()
            for self.batch_idx, batch in enumerate(loader):
                # measure data loading time
                data_time = time.time() - end
                data_time_meter.update(data_time)

                # forward/backward
                start = time.time()
                summary = self.forward_backward(batch)
                batch_time = time.time() - start
                batch_time_meter.update(batch_time)

                # update meters from summary
                loss_val = 0.0
                acc_val = 0.0
                if summary is not None:
                    loss_val = float(summary.get('loss', 0.0))
                    acc_val = float(summary.get('acc', 0.0))
                    loss_meter.update(loss_val, n=1)
                    acc_meter.update(acc_val, n=1)

                # estimate ETA (based on avg batch time)
                remaining_batches = self.num_batches - (self.batch_idx + 1)
                eta_seconds = batch_time_meter.avg * remaining_batches
                eta_min = int(eta_seconds // 60)
                eta_sec = int(eta_seconds % 60)
                eta = f"{eta_min}:{eta_sec:02d}"

                # get lr from first optimizer if available
                lr = 0.0
                for optim in self._optims.values():
                    if optim is not None:
                        try:
                            lr = optim.param_groups[0].get('lr', lr)
                        except Exception:
                            pass
                        break

                # format message similar to original DA/Dassl output
                msg = (
                    f"epoch [{epoch+1}/{max_epoch}][{self.batch_idx+1}/{self.num_batches}]\t"
                    f"time {batch_time:.3f} ({batch_time_meter.avg:.3f})\t"
                    f"data {data_time:.3f} ({data_time_meter.avg:.3f})\t"
                    f"eta {eta}\t"
                    f"loss {loss_val:.4f} ({loss_meter.avg:.4f})\t"
                    f"acc {acc_val:.4f} ({acc_meter.avg:.4f})\t"
                    f"lr {lr:.6e}"
                )

                # log to configured handlers (stdout + train.log + log.txt)
                log.info(msg)

                end = time.time()

            self.after_epoch()
        self.after_train()

    # Hooks
    def before_train(self):
        pass

    def after_train(self):
        # By default, run evaluation on test set after training unless
        # explicitly disabled in cfg (cfg.TEST.NO_TEST == True).
        do_test = not self.cfg.get('TEST', {}).get('NO_TEST', False)
        if do_test:
            try:
                logging.getLogger().info('Do evaluation on test set')
                results = self.test()
                # results is an OrderedDict like {'accuracy': acc, 'error_rate': err, 'macro_f1': mf}
            except Exception as e:
                logging.getLogger().warning('Evaluation failed: %s', e)

    def before_epoch(self):
        pass

    def after_epoch(self):
        """Save checkpoints at the end of each epoch.

        Saves each registered model into a subdirectory under cfg.OUTPUT_DIR
        using the filename pattern 'model.pth.tar-<epoch>'. This matches the
        expected pattern in scripts that load a specific epoch (e.g., 50).
        """
        try:
            from dassl.utils import save_checkpoint, mkdir_if_missing
        except Exception:
            # Fallback: do nothing if utilities are unavailable
            return

        outdir = self.cfg.get('OUTPUT_DIR', '')
        if not outdir:
            return

        # Ensure base output directory exists
        try:
            mkdir_if_missing(outdir)
        except Exception:
            pass

        # Save each registered model state dict along with optimizer/scheduler
        for name in self.get_model_names():
            model = self._models.get(name)
            optim = self._optims.get(name)
            sched = self._scheds.get(name)
            if model is None:
                continue
            state = {
                'state_dict': getattr(model, 'state_dict', lambda: {})(),
                'epoch': self.epoch + 1,
                'optimizer': getattr(optim, 'state_dict', lambda: None)() if optim is not None else None,
                'scheduler': getattr(sched, 'state_dict', lambda: None)() if sched is not None else None,
            }
            subdir = osp.join(outdir, name)
            try:
                mkdir_if_missing(subdir)
                # Only save/overwrite a unified latest checkpoint 'model.pth.tar'
                # so external scripts can always load the most recent one without
                # needing an epoch-specific filename.
                save_checkpoint(state, subdir, is_best=False, model_name='model.pth.tar')
            except Exception as e:
                logging.getLogger().warning('Failed to save checkpoint for %s: %s', name, e)

    def test(self):
        """Run a simple classification evaluation over self.dm.test_loader.

        Returns an OrderedDict with keys similar to Dassl.pytorch's evaluator:
        'accuracy', 'error_rate', 'macro_f1'. Also prints the detailed block so
        it appears in both terminal and log.txt (when setup_logger is used).
        """
        # Ensure model is in eval mode. Try to pick the first registered model.
        names = list(self._models.keys())
        if len(names) == 0:
            raise RuntimeError('No model registered for evaluation')
        model = self._models[names[0]]
        model.eval()

        total = 0
        correct = 0
        y_true = []
        y_pred = []

        with torch.no_grad():
            for batch in self.dm.test_loader:
                # Expect batch to be a dict with 'img' and 'label'
                imgs = batch.get('img')
                labels = batch.get('label')
                if imgs is None or labels is None:
                    continue
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                if hasattr(self, "model_inference"):
                    out = self.model_inference(imgs)
                else:
                    out = model(imgs)
                
                # If model returns tuple (logits, ...), extract logits
                if isinstance(out, tuple) or isinstance(out, list):
                    logits = out[0]
                else:
                    logits = out

                preds = logits.max(1)[1]
                matches = preds.eq(labels).cpu()
                correct += int(matches.sum().item())
                total += labels.shape[0]

                y_true.extend(labels.cpu().numpy().tolist())
                y_pred.extend(preds.cpu().numpy().tolist())

        # Compute metrics
        acc = 100.0 * correct / total if total > 0 else 0.0
        err = 100.0 - acc
        if f1_score is not None and len(y_true) > 0:
            try:
                import numpy as _np
                labels_unique = _np.unique(_np.array(y_true))
                macro = 100.0 * f1_score(y_true, y_pred, average='macro', labels=labels_unique)
            except Exception:
                macro = 0.0
        else:
            macro = 0.0

        # Print block (via logging) matching original format so it appears
        # in train.log and log.txt
        logging.getLogger().info(
            "=> result\n"
            f"* total: {total:,}\n"
            f"* correct: {correct:,}\n"
            f"* accuracy: {acc:.2f}%\n"
            f"* error: {err:.2f}%\n"
            f"* macro_f1: {macro:.2f}%"
        )

        res = OrderedDict()
        res['accuracy'] = acc
        res['error_rate'] = err
        res['macro_f1'] = macro
        return res

    def forward_backward(self, batch):
        raise NotImplementedError

    def parse_batch_train(self, batch):
        raise NotImplementedError

    def load_model(self, directory, epoch=None):
        # Subclasses may override
        pass
