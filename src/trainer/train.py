from lightning.pytorch.loggers import WandbLogger
from .session_preparation import prepare_session


def train(config: dict, audio_dir: str, wandb_logger: WandbLogger):
    trainer, pl_model, data = prepare_session(config, audio_dir, wandb_logger)
    data.setup()
    trainer.fit(pl_model, data)
    pl_model.load_best_model()
    trainer.validate(pl_model, data)
    trainer.test(pl_model, data)
