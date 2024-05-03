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


class ASTAugmenterProcessor(ASTProcessor):
    def __init__(self, config_dir: str, **kwargs):
        self.time_stretch = kwargs.pop("time_stretch", None)
        self.freq_mask = kwargs.pop("freq_mask", None)
        self.time_mask = kwargs.pop("time_mask", None)

        super().__init__(config_dir, **kwargs)

    def __call__(self, features: Tensor) -> Tensor:
        if self.time_stretch is not None:
            rate = np.clip(np.random.standard_normal((1,)), -1, 1)[0] * self.time_stretch
            features = torch.tensor(
                librosa.effects.time_stretch(y=features.numpy(), rate=1 + rate),
                dtype=torch.float32
            )
            if features.shape[1] > 16000:
                idxs = np.random.randint(0, features.shape[1] - 16000, features.shape[0])
                features = features[:, idxs:idxs+16000]
            elif features.shape[1] < 16000:
                features = torch.nn.functional.pad(features, (0, 16000 - features.shape[1]))
        print(features.shape)
        mel_spectogram = super().__call__(features)

        if self.freq_mask is not None:
            freq_mask = T.FrequencyMasking(freq_mask_param=self.freq_mask)
            mel_spectogram = freq_mask(mel_spectogram.transpose(1, 2)).transpose(1, 2)

        if self.time_mask is not None:
            time_mask = T.TimeMasking(time_mask_param=self.time_mask)
            mel_spectogram = time_mask(mel_spectogram.transpose(1, 2)).transpose(1, 2)

        return mel_spectogram


class ASTNormalizedProcessor(ASTProcessor):
    def __call__(self, features: Tensor) -> Tensor:
        return (super().__call__(features) + 0.4722) / (2 * 0.54427)
