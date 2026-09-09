"""Minimal config helper to replace `dassl.config.get_cfg_default()`.

This uses a very small, project-specific dict-based config. It supports
`merge_from_file` for YAML and `merge_from_list` for command-line overrides.
"""
import copy
import yaml
import ast

class CfgNode:
    def __init__(self, initial=None):
        self._store = {}
        if initial:
            for k, v in initial.items():
                if isinstance(v, dict):
                    self._store[k] = CfgNode(v)
                else:
                    self._store[k] = v

    def __getattr__(self, name):
        if name in self._store:
            return self._store[name]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name == '_store':
            super().__setattr__(name, value)
            return
        if isinstance(value, dict):
            self._store[name] = CfgNode(value)
        else:
            self._store[name] = value

    def __getitem__(self, key):
        return self._store[key]

    def get(self, key, default=None):
        return self._store.get(key, default)

    def clone(self):
        return CfgNode(self.to_dict())

    def to_dict(self):
        out = {}
        for k, v in self._store.items():
            if isinstance(v, CfgNode):
                out[k] = v.to_dict()
            else:
                out[k] = v
        return out

    def merge_from_file(self, filepath):
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f) or {}
        for k, v in data.items():
            # If a YAML scalar looks like a tuple/list in parentheses, try to parse it
            if isinstance(v, str) and (v.strip().startswith('(') or v.strip().startswith('[')):
                try:
                    v = ast.literal_eval(v)
                except Exception:
                    pass
            if hasattr(self, k) and isinstance(getattr(self, k), CfgNode) and isinstance(v, dict):
                # merge into existing node
                node = getattr(self, k)
                for kk, vv in v.items():
                    if isinstance(vv, str) and (vv.strip().startswith('(') or vv.strip().startswith('[')):
                        try:
                            vv = ast.literal_eval(vv)
                        except Exception:
                            pass
                    setattr(node, kk, vv)
            else:
                setattr(self, k, v)

    def merge_from_list(self, opts):
        # opts is a list like ["A.B", value, ...]
        it = iter(opts)
        for k, v in zip(it, it):
            keys = k.split('.')
            target = self
            for sub in keys[:-1]:
                if not hasattr(target, sub):
                    setattr(target, sub, {})
                target = getattr(target, sub)
            # parse value
            try:
                val = yaml.safe_load(v)
            except Exception:
                val = v
            setattr(target, keys[-1], val)

    def freeze(self):
        # no-op for this minimal CfgNode
        return

    def defrost(self):
        # no-op for this minimal CfgNode
        return

def get_cfg_default():
    # Provide a minimal default configuration structure expected by the project.
    cfg = CfgNode()
    cfg.DATASET = {
        'ROOT': '',
        'NAME': '',
        'NUM_SHOTS': 0,
        'SUBSAMPLE_CLASSES': 'all'
    }
    cfg.OUTPUT_DIR = ''
    cfg.RESUME = ''
    cfg.SEED = -1
    cfg.USE_CUDA = True
    cfg.VERBOSE = True
    cfg.MODEL = {
        'BACKBONE': {'NAME': '', 'PRETRAINED': False},
        'HEAD': {'NAME': '', 'HIDDEN_LAYERS': None},
        'INIT_WEIGHTS': False
    }
    cfg.TRAINER = {'NAME': ''}
    cfg.INPUT = {'SIZE': [224]}
    cfg.OPTIM = {'MAX_EPOCH': 1}
    return cfg
