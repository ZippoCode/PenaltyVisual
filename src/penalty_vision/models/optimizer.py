import torch.optim as optim


def get_optimizer(model_parameters, lr: float = 0.001, weight_decay: float = 0.0, betas: tuple = (0.9, 0.999),
                  eps: float = 1e-8) -> optim.Adam:
    return optim.Adam(
        model_parameters,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay
    )


def get_scheduler(
        optimizer: optim.Optimizer,
        mode: str = 'min',
        factor: float = 0.1,
        patience: int = 10,
        min_lr: float = 1e-6,
        threshold: float = 1e-4
) -> optim.lr_scheduler.ReduceLROnPlateau:
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=factor,
        patience=patience,
        min_lr=min_lr,
        threshold=threshold,
    )
