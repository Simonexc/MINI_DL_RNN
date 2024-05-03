from abc import ABC, abstractmethod
from typing import Callable

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
        
        if hasattr(config, 'to_dict'):
            config_dict = config.to_dict()
        else:
            config_dict = config.__dict__  # Fallback to converting attributes directly
        
        self.augmentations = config_dict.get('augmentations', {})

    def __call__(self, features: Tensor) -> Tensor:
        numpy_features = features.numpy()
        sr = AUDIO_FILE_METADATA.get("sample_rate", 16000)
        
        if 'spectrogram' in self.augmentations:
            numpy_features = self.spectrogram(numpy_features, sr)
        
        if 'time_stretch' in self.augmentations:
            stretch_rate = self.augmentations['time_stretch']
            numpy_features = self.apply_time_stretch(numpy_features, stretch_rate)
        
        if 'freq_mask' in self.augmentations:
            mask_param = self.augmentations['freq_mask']
            freq_mask = T.FrequencyMasking(mask_param)
            numpy_features = freq_mask(numpy_features)

        if 'time_mask' in self.augmentations:
            mask_param = self.augmentations['time_mask']
            time_mask = T.TimeMasking(mask_param)
            numpy_features = time_mask(numpy_features)
        
        return self.feature_extractor(
            numpy_features,
            return_tensors="pt",
            sampling_rate=sr
        ).input_values


class ASTNormalizedProcessor(ASTProcessor):
    def __call__(self, features: Tensor) -> Tensor:
        return (super().__call__(features) + 0.4722) / (2 * 0.54427)


