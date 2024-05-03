from lightning.pytorch.loggers import WandbLogger
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor

from dataset.training_dataset import SpeechDataset
import models
from models.lightning_model import LightningModel
import dataset.feature_processors as feature_processors


def prepare_session(
    config: dict,
    audio_dir: str,
    wandb_logger: WandbLogger
) -> tuple[pl.Trainer, LightningModel, SpeechDataset]:
    feature_processor: str | None = config.get("feature_processor", None)
    data = SpeechDataset(
        audio_dir,
        config["batch_size"],
        feature_processor and getattr(feature_processors, feature_processor)(
            **{
                key: config[key]
                for key in config.get("feature_processor_params", [])
            }
        ),
        config.get("train_num_samples", None),
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(
        logger=wandb_logger,
        log_every_n_steps=10,
        max_epochs=config["epochs"],
        callbacks=[lr_monitor]
    )

    model = getattr(models, config["model_class"])(
        **{
            key: config[key]
            for key in config["model_params"]
        }
    )
    pl_model = LightningModel(
        model,
        config["model_name"],
        config["lr"],
        config.get("l2_penalty", 0),
        (config.get("beta1", 0.9), config.get("beta2", 0.999)),
        config.get("scheduler_factor", None),
        config.get("scheduler_patience", None),
    )

    return trainer, pl_model, data
