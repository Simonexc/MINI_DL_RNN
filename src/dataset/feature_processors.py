from typing import Callable

from torch import Tensor
from transformers import ASTFeatureExtractor

from settings import AUDIO_FILE_METADATA


def curried_ast_processor(
    pretrained_path: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
) -> Callable[[Tensor], Tensor]:
    feature_extractor = ASTFeatureExtractor.from_pretrained(pretrained_path)

    def ast_processor(features: Tensor) -> Tensor:
        return feature_extractor(
            features.numpy(),
            return_tensors="pt",
            sampling_rate=AUDIO_FILE_METADATA["sample_rate"],
        )

    return ast_processor
