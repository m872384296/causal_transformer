from torch import optim as optim

def build_all_opt(net):
    opt_cls = build_optimizer(net[0], lr=1e-4, weight_decay=0.05)
    opt_spl = build_optimizer(net[1], lr=1e-5, weight_decay=0.05)
    return opt_cls, opt_spl

def build_optimizer(model, lr=1e-4, weight_decay=0.05):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    parameters = set_weight_decay(model, skip, skip_keywords)
    optimizer = optim.AdamW(parameters, eps=1e-8, betas=(0.9, 0.999),
                            lr=lr, weight_decay=weight_decay)
    return optimizer

def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []
    split_lr = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]

def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin