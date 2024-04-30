import math

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers.models.audio_spectrogram_transformer.modeling_audio_spectrogram_transformer import ASTEmbeddings
from transformers import ASTConfig

from settings import NUM_CLASSES
from .embeddings import FCEmbedding


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        embedding_hidden: int,
        hidden_vector_size: int,
        heads_num: int,
        hidden_layer_size: int,
        layers_num: int,
        max_input_size: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.pos_encoder = PositionalEncoding(hidden_vector_size, dropout, max_len=max_input_size)
        encoder_layers = TransformerEncoderLayer(hidden_vector_size, heads_num, hidden_layer_size, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, layers_num)
        self.embedding = FCEmbedding(input_size, hidden_vector_size, embedding_hidden)
        self.hidden_vector_size = hidden_vector_size
        self.linear = nn.Linear(hidden_vector_size, NUM_CLASSES)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[batch_size, waveform_length]``

        Returns:
            output Tensor of shape ``[batch_size, seq_len, ntoken]``
        """
        src = self.embedding(src) * math.sqrt(self.hidden_vector_size)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.mean(dim=1)
        output = self.linear(output)
        return output


class TransformerASTModel(nn.Module):
    def __init__(
        self,
        hidden_vector_size: int,
        heads_num: int,
        hidden_layer_size: int,
        layers_num: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        config = ASTConfig(
            hidden_size=hidden_vector_size,
            patch_size=16,
            num_mel_bins=128,
            frequency_stride=10,
            time_stride=10,
            max_length=1024,
            hidden_dropout_prob=dropout,
            num_labels=NUM_CLASSES,
        )

        self.embedding = ASTEmbeddings(config)
        encoder_layers = TransformerEncoderLayer(hidden_vector_size, heads_num, hidden_layer_size, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, layers_num)
        self.hidden_vector_size = hidden_vector_size
        self.linear = nn.Linear(hidden_vector_size, NUM_CLASSES)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[batch_size, waveform_length]``

        Returns:
            output Tensor of shape ``[batch_size, seq_len, ntoken]``
        """
        src = self.embedding(src)
        output = self.transformer_encoder(src)
        output = output.mean(dim=1)
        output = self.linear(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, sequence_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.shape[1]]
        return self.dropout(x)
