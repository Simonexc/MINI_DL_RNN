from torch import nn, Tensor

from settings import NUM_CLASSES
from .embeddings import FCEmbedding


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        embedding_hidden: int,
        hidden_vector_size: int,
        layers_num: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding = FCEmbedding(input_size, hidden_vector_size, embedding_hidden)
        self.lstm = nn.LSTM(
            hidden_vector_size, hidden_vector_size, layers_num, batch_first=True, dropout=dropout
        )
        self.linear = nn.Linear(hidden_vector_size, NUM_CLASSES)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor) -> Tensor:
        src = self.embedding(src)
        output, (hidden, _) = self.lstm(src)
        output = output.mean(dim=1)
        output = self.linear(output)
        return output
