from lightning.pytorch.loggers import WandbLogger
import lightning.pytorch as pl
import wandb

from dataset.training_dataset import SpeechDataset
import models
from models.lightning_model import LightningModel
import dataset.feature_processors as feature_processors


def train(config: wandb.sdk.Config, audio_dir: str, wandb_logger: WandbLogger):
    feature_processor: str | None = getattr(config, "feature_processor", None)
    data = SpeechDataset(
        audio_dir,
        config.batch_size,
        feature_processor and getattr(feature_processors, feature_processor)(),
        getattr(config, "train_num_samples", None),
    )
    data.setup()

    trainer = pl.Trainer(
        logger=wandb_logger,
        log_every_n_steps=10,
        max_epochs=config.epochs,
    )

    model = getattr(models, config.model_class)(
        **{
            key: getattr(config, key)
            for key in config.model_params
        }
    )
    pl_model = LightningModel(
        model,
        config.model_name,
        config.lr,
        getattr(config, "l2_penalty", 0),
        (getattr(config, "beta1", 0.9), getattr(config, "beta2", 0.999)),
    )
    trainer.fit(pl_model, data)
    pl_model.load_best_model()
    trainer.validate(pl_model, data)
    trainer.test(pl_model, data)
