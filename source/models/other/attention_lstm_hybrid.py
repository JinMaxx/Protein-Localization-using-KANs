"""
This module defines an abstract base class, `AbstractAttentionLstmHybrid`, for creating
hybrid models that leverage both self-attention and recurrent layers.

The architecture is structured as a sequential pipeline:
1. A Multi-Head Attention layer with a residual connection processes the input
   sequence to capture contextual relationships.
2. The output is then fed into a bidirectional LSTM to model sequential dependencies.
3. The LSTM's full output sequence is aggregated using both mean and max pooling
   to create a rich, fixed-size summary vector.
4. Finally, a pluggable Feed-Forward Network (FFN) produces the final output logits.

This module also provides a concrete implementation, `AttentionLstmHybridFastKAN`,
which uses a FastKAN as the final FFN.
"""

import torch
import torch.nn as nn

from abc import ABC
from typing_extensions import cast, override, Optional, Type, List, Dict

from source.models.abstract import AbstractSequenceModel
from source.models.ffn import AbstractFFN, FastKAN
from source.training.utils.collate_functions import per_residue_collate_function
from source.training.utils.hidden_layers import HiddenLayers

from source.config import ConfigType

from source.custom_types import (
    Encodings_Batch_T,
    AttentionMask_Batch_T,
    Data_T,
    Labels_Batch_T,
    Logits_Batch_T
)



class AbstractAttentionLstmHybrid(AbstractSequenceModel, ABC):
    """
    A hybrid model using Multi-Head Attention, a bidirectional LSTM, and a KAN.

    This model processes sequences by first applying self-attention to extract
    contextual features, then feeding the result into a bidirectional LSTM to
    capture sequential dependencies. The final hidden states of the LSTM are
    concatenated and passed to a KAN for final classification or regression.
    """


    def __init__(self,
            in_channels: int,
            in_seq_len: int,
            attention_num_heads: int,
            lstm_hidden_size: int,
            lstm_num_layers: int,
            dropout1_rate: float,
            dropout2_rate: float,
            hidden_layers: HiddenLayers,
            ffn_layer_class: Type[AbstractFFN],
            ffn_layer_kwargs: Optional[Dict] = None,
            out_channels: Optional[int] = None):
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

        super().__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            in_seq_len = in_seq_len
        )

        self.attention_num_heads = attention_num_heads
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.hidden_layers = hidden_layers
        self.dropout1_rate: float = dropout1_rate
        self.dropout2_rate: float = dropout2_rate
        self.ffn_layer_class: Type[AbstractFFN] = ffn_layer_class
        self.ffn_layer_kwargs: Dict = ffn_layer_kwargs if ffn_layer_class is not None else {}


        # Multi-Head Attention Layer
        self.attention = nn.MultiheadAttention(
            embed_dim = self.in_channels,
            num_heads = self.attention_num_heads,
            batch_first = True
        )
        self.layer_norm1 = nn.LayerNorm(self.in_channels)
        self.dropout1 = nn.Dropout(self.dropout1_rate)

        # 2. bi-LSTM Layer to understand global context
        # input: (batch_size, seq_len, in_channels)
        self.lstm = nn.LSTM(
            input_size  = self.in_channels,
            hidden_size = self.lstm_hidden_size,
            num_layers  = self.lstm_num_layers,
            bidirectional = True,
            batch_first   = True
        )
        self.dropout2 = nn.Dropout(self.dropout2_rate)

        # 3. KAN for final classification
        # concatenation of last layers from forward and backward pass in LSTM:
        # (batch_size, lstm_hidden_size) -> (batch_size, lstm_hidden_size * 2)
        self.ffn_layer: AbstractFFN = ffn_layer_class(
            in_channels = 1,  # Output from LSTM is flat
            in_seq_len = self.lstm_hidden_size * 4,  # *2 because bidirectional and * 2 because mean + max pooling
            hidden_layers = self.hidden_layers,
            out_channels = self.out_channels,
            **self.ffn_layer_kwargs
        )


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
        x = self.dropout1(x)

        # LSTM processing
        # h_n shape: [num_layers * num_directions, batch, hidden_size]
        # _, (h_n, _) = self.lstm(x) # that produces a severe bottleneck

        # Concatenate the final forward and backward hidden states
        # Get the last layer's hidden state (h_n[-2,:,:] for fwd, h_n[-1,:,:] for bwd)
        # x = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        # disregards the full output sequence of LSTM -> maybe that is holding the model back?

        # Revised logic:
        # Get the FULL output sequence from the LSTM
        # lstm_output shape: [batch_size, seq_len, lstm_hidden_size * 2]
        lstm_output, _ = self.lstm(x)
        lstm_output = self.dropout2(lstm_output)

        # Mask out padded values before pooling.
        if attention_mask is not None:
            # Expand mask to match lstm_output dimensions for broadcasting
            expanded_mask = attention_mask.unsqueeze(-1).expand_as(lstm_output)
            lstm_output = lstm_output * expanded_mask

        mean_pooled = torch.mean(lstm_output, dim=1)
        max_pooled, _ = torch.max(lstm_output, dim=1)
        x = torch.cat((mean_pooled, max_pooled), dim=1)  # rich summery vector

        # KAN for classification
        logits = self.ffn_layer(x)

        return logits


    @override
    def collate_function(self, batch: List[Data_T]) -> tuple[Encodings_Batch_T, Labels_Batch_T, AttentionMask_Batch_T]:
        return per_residue_collate_function(batch, max_seq_len=self.in_seq_len, flatten=False)  # unflattened!



class AttentionLstmHybridFastKAN(AbstractAttentionLstmHybrid):
    """
    A concrete implementation of the AbstractAttentionLstmHybrid that uses a
    FastKAN for the final classification layer.
    """

    def __init__(self,
            in_channels: int,
            in_seq_len: int = None,
            out_channels: Optional[int] = None,
            attention_num_heads: int = None,
            lstm_hidden_size: int = None,
            lstm_num_layers: int = None,
            dropout1_rate: float = None,
            dropout2_rate: float = None,
            hidden_layers: HiddenLayers = None,
            grid_diff: int = None,
            num_grids: int = None):
        from source.config import AttentionLstmHybridFastKANConfig
        config: AttentionLstmHybridFastKANConfig = cast(AttentionLstmHybridFastKANConfig, self.get_config())
        self.grid_diff = grid_diff if grid_diff is not None else config.grid_diff
        self.num_grids = num_grids if num_grids is not None else config.num_grids
        super().__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            in_seq_len = in_seq_len if in_seq_len is not None else config.in_seq_len,
            attention_num_heads = attention_num_heads if attention_num_heads is not None else config.attention_num_heads,
            lstm_hidden_size = lstm_hidden_size if lstm_hidden_size is not None else config.lstm_hidden_size,
            lstm_num_layers = lstm_num_layers if lstm_num_layers is not None else config.lstm_num_layers,
            dropout1_rate = dropout1_rate if dropout1_rate is not None else config.dropout1_rate,
            dropout2_rate = dropout2_rate if dropout2_rate is not None else config.dropout2_rate,
            hidden_layers = hidden_layers if hidden_layers is not None else config.hidden_layers,
            ffn_layer_class = FastKAN,
            ffn_layer_kwargs = dict(
                grid_diff = self.grid_diff,
                num_grids = self.num_grids
            )
        )


    @classmethod
    @override
    def config_type(cls) -> ConfigType: return ConfigType.AttentionLstmHybridFastKAN
