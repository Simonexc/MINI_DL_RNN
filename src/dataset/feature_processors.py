from abc import ABC, abstractmethod
from typing import Callable

from torch import Tensor
from transformers import ASTFeatureExtractor

from settings import AUDIO_FILE_METADATA


class BaseProcessor(ABC):
    @abstractmethod
    def __call__(self, features: Tensor) -> Tensor:
        """
        Convert features to correct format.
        """


class ASTProcessor(BaseProcessor):
    def __init__(self, pretrained_path: str):
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(pretrained_path, local_files_only=True)

    def __call__(self, features: Tensor) -> Tensor:
        return self.feature_extractor(
            features.numpy(),
            return_tensors="pt",
            sampling_rate=AUDIO_FILE_METADATA["sample_rate"],
        ).input_values
