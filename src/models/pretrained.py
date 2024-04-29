from transformers import ASTConfig, ASTForAudioClassification

from settings import NUM_CLASSES


class FineTunedAST(ASTForAudioClassification):
    def __init__(
        self, pretrained_path: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    ):
        config = ASTConfig.from_pretrained(pretrained_path)
        config.num_labels = NUM_CLASSES

        super().__init__(config)
