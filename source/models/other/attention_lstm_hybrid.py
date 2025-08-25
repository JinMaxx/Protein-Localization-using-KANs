"""
This module defines the `AttentionLstmHybrid` model, an architecture that
leverages both self-attention and recurrent layers to process sequence data.

The model is structured as a sequential pipeline:
1. A Multi-Head Attention layer first processes the input sequence to weigh
   the importance of different elements and capture contextual relationships.
2. The output of the attention layer is then fed into a bidirectional LSTM,
   which models the temporal or sequential dependencies within the feature-rich sequence.
3. Finally, the last hidden states from both directions of the LSTM are
   concatenated and passed to a Feed-Forward Network (e.g., FastKAN)
   to produce the final output logits for classification or regression.
"""

import torch
import torch.nn as nn

from typing_extensions import cast, override, Optional, Type, List

from source.models.abstract import AbstractSequenceModel
from source.models.ffn import AbstractFFN, FastKAN
from source.training.utils.collate_functions import per_residue_collate_function
from source.training.utils.hidden_layers import HiddenLayers

from source.config import ConfigType, AttentionLstmHybridConfig

from source.custom_types import (
    Encodings_Batch_T,
    AttentionMask_Batch_T,
    Data_T,
    Labels_Batch_T,
    Logits_Batch_T
)



class AttentionLstmHybrid(AbstractSequenceModel):
    """
    A hybrid model using Multi-Head Attention, a bidirectional LSTM, and a KAN.

    This model processes sequences by first applying self-attention to extract
    contextual features, then feeding the result into a bidirectional LSTM to
    capture sequential dependencies. The final hidden states of the LSTM are
    concatenated and passed to a KAN for final classification or regression.
    """


    def __init__(self,
            in_channels: int,
            in_seq_len: int = None,
            out_channels: Optional[int] = None,
            attention_num_heads: int = None,
            lstm_hidden_size: int = None,
            lstm_num_layers: int = None,
            hidden_layers: HiddenLayers = None,
            ffn_layer_class: Type[AbstractFFN] = FastKAN):
        """
        Initializes the AttentionLstmHybrid model.

        Args:
            in_channels: The number of features for each element in the input sequence.
            in_seq_len: The length of the input sequences.
            out_channels: The number of output classes.
            attention_num_heads: The number of heads in the Multi-Head Attention layer.
            lstm_hidden_size: The number of features in the LSTM hidden state.
            lstm_num_layers: The number of recurrent LSTM layers.
            hidden_layers: The configuration for the hidden layers of the final FFN.
            ffn_layer_class: The class to use for the final feed-forward network (e.g., FastKAN).
        """
        config: AttentionLstmHybridConfig = cast(AttentionLstmHybridConfig, self.get_config())

        super().__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            in_seq_len = in_seq_len if in_seq_len is not None else config.in_seq_len
        )

        self.attention_num_heads = attention_num_heads if attention_num_heads is not None else config.attention_num_heads
        self.lstm_hidden_size = lstm_hidden_size if lstm_hidden_size is not None else config.lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers if lstm_num_layers is not None else config.lstm_num_layers
        self.hidden_layers = hidden_layers if hidden_layers is not None else config.hidden_layers
        self.ffn_layer_class: Type[AbstractFFN] = ffn_layer_class

        # Multi-Head Attention Layer
        self.attention = nn.MultiheadAttention(
            embed_dim = self.in_channels,
            num_heads = self.attention_num_heads,
            batch_first = True
        )
        self.layer_norm1 = nn.LayerNorm(self.in_channels)

        # 2. bi-LSTM Layer to understand global context
        # input: (batch_size, seq_len, in_channels)
        self.lstm = nn.LSTM(
            input_size  = self.in_channels,
            hidden_size = self.lstm_hidden_size,
            num_layers  = self.lstm_num_layers,
            bidirectional = True,
            batch_first   = True
        )

        # 3. KAN for final classification
        # concatenation of last layers from forward and backward pass in LSTM:
        # (batch_size, lstm_hidden_size) -> (batch_size, lstm_hidden_size * 2)
        self.ffn_layer: AbstractFFN = ffn_layer_class(
            in_channels = 1,  # Output from LSTM is flat
            in_seq_len = self.lstm_hidden_size * 2,  # *2 because bidirectional
            hidden_layers = self.hidden_layers,
            out_channels = self.out_channels,
        )


    @classmethod
    @override
    def config_type(cls) -> ConfigType: return ConfigType.AttentionLstmHybrid


    @override
    def forward(self, x: Encodings_Batch_T, attention_mask: AttentionMask_Batch_T) -> Logits_Batch_T:
        """
        Defines the forward pass for the hybrid model.

        The process is as follows:
        1.  Input sequence is passed through a Multi-Head Attention layer. A residual
            connection and Layer Normalization are applied.
        2.  The output from the attention block is passed to the bidirectional LSTM.
        3.  The final hidden states from the last layer of the LSTM (one for the
            forward pass, one for the backward pass) are concatenated.
        4.  The resulting vector is passed to the FFN to produce the final logits.

        Args:
            x: Input batch of sequences, shape: (batch_size, seq_len, in_channels).
            attention_mask: An mask indicating valid sequence elements.

        Returns:
            The final output logits from the FFN.
        """

        # Input shape: [batch_size, seq_len, in_channels]

        # Create key_padding_mask from the attention_mask
        # MultiheadAttention expects True for positions to be ignored.
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None

        # Multi-Head Attention
        # The query, key, and value are all the same in self-attention
        attn_output, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)

        # Residual connection and Layer Normalization
        x = self.layer_norm1(x + attn_output)

        # LSTM processing
        # h_n shape: (num_layers * num_directions, batch, hidden_size)
        _, (h_n, _) = self.lstm(x)

        # Concatenate the final forward and backward hidden states
        # Get the last layer's hidden state (h_n[-2,:,:] for fwd, h_n[-1,:,:] for bwd)
        x = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        # disregards the full output sequence of LSTM -> maybe that is holding the model back?

        # KAN for classification
        logits = self.ffn_layer(x)
        return logits


    @override
    def collate_function(self, batch: List[Data_T]) -> tuple[Encodings_Batch_T, Labels_Batch_T, AttentionMask_Batch_T]:
        return per_residue_collate_function(batch, max_seq_len=self.in_seq_len, flatten=False)  # unflattened!