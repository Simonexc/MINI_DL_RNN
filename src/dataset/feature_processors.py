from abc import ABC, abstractmethod
from typing import Callable

from torch import Tensor
from transformers import ASTFeatureExtractor, ASTConfig

from settings import AUDIO_FILE_METADATA, NUM_CLASSES
from .utils import load_ast_config


class BaseProcessor(ABC):
    @abstractmethod
    def __call__(self, features: Tensor) -> Tensor:
        """
        Convert features to correct format.
        """


class ASTProcessor(BaseProcessor):
    def __init__(self, config_dir: str, **kwargs):
        config = load_ast_config(config_dir, **kwargs)
        self.feature_extractor = ASTFeatureExtractor.from_dict(config.to_dict())

    def __call__(self, features: Tensor) -> Tensor:
        return self.feature_extractor(
            features.numpy(),
            return_tensors="pt",
            sampling_rate=AUDIO_FILE_METADATA["sample_rate"],
        ).input_values


class ASTNormalizedProcessor(BaseProcessor):
    def __call__(self, features: Tensor) -> Tensor:
        return (super().__call__(features) + 0.4722) / (2 * 0.54427)
