import torch
from torch import nn

from engine.training_engine.training.dataset_bundle import DatasetBundle

from engine.training_engine.config.config import EPOCHS

def _forward(model: nn.Module, data) -> torch.Tensor:

    edge_weight = getattr(data, "edge_weight", None)

    if edge_weight is None:
        return model(
            data.x,
            data.edge_index,
        )

    return model(
        data.x,
        data.edge_index,
        edge_weight=edge_weight,
    )

def full_batch_node_trainer(
        *,
        model: nn.Module,
        bundle: DatasetBundle,
        optimizer: torch.optim.Optimizer,
        loss_function,
        metric_function,
        device: torch.device
) -> dict:

    data = bundle.data.to(device)

    train_split = bundle.train_split.to(device)
    val_split = bundle.val_split.to(device)
    test_split = bundle.test_split.to(device)

    if data.y is None:
        raise ValueError("The dataset does not contain target labels.")

    labels = torch.as_tensor(data.y, device=device).reshape(-1)

    history = []

    for epoch in range(1, EPOCHS + 1):


        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits = _forward(
            model,
            data,
        )

        train_loss = loss_function(
            logits[train_split],
            labels[train_split],
        )

        train_loss.backward()
        optimizer.step()


        model.eval()

        with torch.no_grad():

            logits = _forward(
                model,
                data,
            )

            train_metric = metric_function(
                logits[train_split],
                labels[train_split],
                evaluator=bundle.evaluator,
            )

            val_loss = loss_function(
                logits[val_split],
                labels[val_split],
            )

            val_metric = metric_function(
                logits[val_split],
                labels[val_split],
                evaluator=bundle.evaluator,
            )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss.item(),
            "train_metric": train_metric,
            "val_loss": val_loss.item(),
            "val_metric": val_metric,
        })


    model.eval()

    with torch.no_grad():

        logits = _forward(
            model,
            data,
        )

        test_loss = loss_function(
            logits[test_split],
            labels[test_split],
        )

        test_metric = metric_function(
            logits[test_split],
            labels[test_split],
            evaluator=bundle.evaluator,
        )

    return {
        "history": history,
        "test_loss": test_loss.item(),
        "test_metric": test_metric,
    }