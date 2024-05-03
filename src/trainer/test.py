from lightning.pytorch.loggers import WandbLogger
from .session_preparation import prepare_session
from settings import ALL_CLASSES


def test(config: dict, audio_dir: str, wandb_logger: WandbLogger, model_checkpoint: str) -> tuple[list[str], list[str]]:
    trainer, pl_model, data = prepare_session(config, audio_dir, wandb_logger)
    data.setup("test")
    pl_model.load_local(model_checkpoint)
    trainer.test(pl_model, data)

    return (
        data.test.tensors[1].numpy().tolist(),
        [ALL_CLASSES[class_id] for class_id in pl_model.test_class_ids.numpy().tolist()],
    )
