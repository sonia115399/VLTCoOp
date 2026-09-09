"""Small utilities used by train script and trainers."""
import os
import logging
import random
import platform
import sys
import time
import torch


class StdoutTee:
    """Duplicate sys.stdout to both console and a text file.

    Use append mode so we don't truncate previous logs when running
    multiple seeds.
    """

    def __init__(self, filepath):
        self.console = sys.stdout
        try:
            self.file = open(filepath, "a")
        except Exception:
            self.file = None

    def write(self, msg):
        try:
            self.console.write(msg)
        except Exception:
            pass
        if self.file is not None:
            try:
                self.file.write(msg)
            except Exception:
                pass

    def flush(self):
        try:
            self.console.flush()
        except Exception:
            pass
        if self.file is not None:
            try:
                self.file.flush()
                os.fsync(self.file.fileno())
            except Exception:
                pass

    def close(self):
        if self.file is not None:
            try:
                self.file.close()
            except Exception:
                pass


def setup_logger(output=None):
    if output:
        os.makedirs(output, exist_ok=True)
        logfile = os.path.join(output, 'train.log')
        logtxt = os.path.join(output, 'log.txt')
    else:
        logfile = None

    # If requested, set up stdout tee so print(...) is captured in log.txt
    if output:
        try:
            tee = StdoutTee(logtxt)
            sys.stdout = tee
        except Exception:
            pass

    handlers = [logging.StreamHandler(sys.stdout)]
    if logfile:
        handlers.append(logging.FileHandler(logfile))
        try:
            handlers.append(logging.FileHandler(logtxt))
        except Exception:
            pass

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', handlers=handlers)
    logging.getLogger().info('Logger initialized. Output dir: %s', output)


def set_random_seed(seed):
    if seed is None or seed < 0:
        return
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def collect_env_info():
    info = {
        'platform': platform.platform(),
        'python': sys.version.replace('\n', ' '),
        'torch': torch.__version__,
        'time': time.asctime()
    }
    return '\n'.join([f'{k}: {v}' for k, v in info.items()])


def mkdir_if_missing(dirname):
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)


def check_isfile(path):
    """Return whether a path exists and is a regular file."""
    try:
        return os.path.exists(path) and os.path.isfile(path)
    except Exception:
        return False


def listdir_nohidden(path, sort=False):
    """List directory entries excluding hidden files (those starting with '.')

    Returns a list of names (not full paths). If the directory can't be
    listed, returns an empty list.
    """
    try:
        names = [n for n in os.listdir(path) if not n.startswith('.')]
    except Exception:
        return []
    if sort:
        try:
            names.sort()
        except Exception:
            pass
    return names


def load_checkpoint(path, map_location=None):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if map_location is None and not torch.cuda.is_available():
        map_location = "cpu"
    return torch.load(path, map_location=map_location)


def load_pretrained_weights(model, path, strict=False):
    # Accept path or state_dict
    import torch
    if path is None or path == '':
        return
    if isinstance(path, dict):
        sd = path
    else:
        sd = torch.load(path, map_location='cpu')
        if 'state_dict' in sd:
            sd = sd['state_dict']
    model.load_state_dict(sd, strict=strict)


def save_checkpoint(state, directory, is_best=False, model_name=''):
    """Save a training checkpoint.

    If model_name is provided, it will be used as the filename. Otherwise,
    the default name will be 'model-best.pth.tar' for best checkpoints or
    'model.pth.tar' for regular checkpoints.
    """
    mkdir_if_missing(directory)
    # Honor explicit model_name if provided
    if isinstance(model_name, str) and len(model_name) > 0:
        fname = model_name
    else:
        fname = 'model-best.pth.tar' if is_best else 'model.pth.tar'
    path = os.path.join(directory, fname)
    torch.save(state, path)
    try:
        logging.getLogger().info('Checkpoint saved to %s', path)
        if is_best:
            logging.getLogger().info('Best checkpoint (is_best=True) saved to %s', path)
    except Exception:
        pass


def resume_from_checkpoint(path, model, optim=None, sched=None):
    # Find latest checkpoint file
    # This minimal implementation looks for model-best or model.pth.tar
    best = os.path.join(path, 'model-best.pth.tar')
    p = best if os.path.exists(best) else os.path.join(path, 'model.pth.tar')
    if not os.path.exists(p):
        return 0
    checkpoint = load_checkpoint(p)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    if optim is not None and 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
        try:
            optim.load_state_dict(checkpoint['optimizer'])
        except Exception:
            pass
    if sched is not None and 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
        try:
            sched.load_state_dict(checkpoint['scheduler'])
        except Exception:
            pass
    return checkpoint.get('epoch', 0)
