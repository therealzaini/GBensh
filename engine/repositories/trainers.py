def full_batch_node_trainer(**kwargs):
    from engine.training_engine.training.trainers.full_batch_node_trainer import (
        full_batch_node_trainer as train,
    )

    return train(**kwargs)


TRAINER_REGISTRY = {
    "full_batch_node_trainer": full_batch_node_trainer,
}
