def adam(parameters, **kwargs):
    from torch.optim import Adam

    return Adam(parameters, **kwargs)


def adamw(parameters, **kwargs):
    from torch.optim import AdamW

    return AdamW(parameters, **kwargs)


def sgd(parameters, **kwargs):
    from torch.optim import SGD

    return SGD(parameters, **kwargs)


OPTIMIZERS = {
    'adam': adam,
    'adamw': adamw,
    'sgd': sgd,
}
