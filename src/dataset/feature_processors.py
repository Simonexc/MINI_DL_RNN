from abc import ABC, abstractmethod
from typing import Callable
import torch
from torch import Tensor
from transformers import ASTFeatureExtractor, ASTConfig

from settings import AUDIO_FILE_METADATA, NUM_CLASSES
from .utils import load_ast_config

import librosa
import numpy as np
import torchaudio.transforms as T

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
        numpy_features = features.numpy()
        sr = AUDIO_FILE_METADATA.get("sample_rate", 16000)
        return self.feature_extractor(
            numpy_features,
            return_tensors="pt",
            sampling_rate=sr
        ).input_values

class ASTAugmenter(ASTProcessor):
    def __init__(self, config_dir: str, max_length: int, time_stretch: float, freq_mask: int, time_mask: int, **kwargs):
        super().__init__(config_dir, max_length=max_length, **kwargs)
        self.time_stretch = time_stretch
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.max_length = max_length

    def apply_time_stretch(self, audio, rate):
        if rate != 1.0:  # Only apply if the rate is not the default value
            return librosa.effects.time_stretch(audio, rate=rate)
        return audio

    def __call__(self, features: Tensor) -> Tensor:
        numpy_features = features.numpy()
        sr = AUDIO_FILE_METADATA.get("sample_rate", 16000)

        if self.time_stretch is not None:
            numpy_features = self.apply_time_stretch(audio=numpy_features, rate=self.time_stretch)

        if self.freq_mask is not None:
            freq_mask = T.FrequencyMasking(self.freq_mask)
            tensor_features = torch.tensor(numpy_features, dtype=torch.float32).clone().detach()  # clone and detach
            numpy_features = freq_mask(tensor_features).numpy()

        if self.time_mask is not None:
            time_mask = T.TimeMasking(self.time_mask)
            tensor_features = torch.tensor(numpy_features, dtype=torch.float32).clone().detach()  # clone and detach
            numpy_features = time_mask(tensor_features).numpy()

        features_tensor = torch.tensor(numpy_features, dtype=torch.float32).clone().detach()  # clone and detach
        return super().__call__(features_tensor)



class ASTNormalizedProcessor(ASTAugmenter):
    def __call__(self, features: Tensor) -> Tensor:
        return (super().__call__(features) + 0.4722) / (2 * 0.54427)


