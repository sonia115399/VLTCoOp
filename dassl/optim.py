"""Simple optimizer / scheduler builders used by trainers."""
import torch

def build_optimizer(model, optim_cfg):
    # optim_cfg is expected to be a dict-like object
    params = [p for p in model.parameters() if p.requires_grad]
    lr = optim_cfg.get('LR', optim_cfg.get('LR_BASE', 1e-3)) if isinstance(optim_cfg, dict) else 1e-3
    optim_name = optim_cfg.get('NAME', 'Adam') if isinstance(optim_cfg, dict) else 'Adam'
    if optim_name.lower() == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=optim_cfg.get('MOMENTUM', 0.9))
    else:
        return torch.optim.Adam(params, lr=lr)


def build_lr_scheduler(optim, optim_cfg):
    # minimal stub: return None or a StepLR if requested
    if not isinstance(optim_cfg, dict):
        return None
    if optim_cfg.get('SCHED', '') == 'StepLR':
        step_size = optim_cfg.get('STEP_SIZE', 30)
        gamma = optim_cfg.get('GAMMA', 0.1)
        return torch.optim.lr_scheduler.StepLR(optim, step_size=step_size, gamma=gamma)
    return None
