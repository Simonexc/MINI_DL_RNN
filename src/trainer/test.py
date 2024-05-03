from lightning.pytorch.loggers import WandbLogger
from .session_preparation import prepare_session
from settings import ALL_CLASSES
from dataset.training_dataset import KaggleTestDataset


def test(config: dict, audio_dir: str, wandb_logger: WandbLogger, model_checkpoint: str) -> tuple[list[str], list[str]]:
    trainer, pl_model, data = prepare_session(config, audio_dir, wandb_logger, KaggleTestDataset)
    data.setup("test")
    pl_model.load_local(model_checkpoint)
    pl_model.log_test = False
    trainer.test(pl_model, data)

    return (
        data.file_names,
        [ALL_CLASSES[class_id] for class_id in pl_model.test_class_ids.numpy().tolist()],
    )
