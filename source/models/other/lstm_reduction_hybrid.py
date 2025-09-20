"""
This module defines an abstract base class, `AbstractLstmReductionHybrid`, for creating
hybrid sequence models. It provides a reusable pipeline that combines a recurrent
LSTM layer with a pluggable sequence reduction layer and a final feed-forward network (FFN).

The model is designed as a multi-stage architecture:
1. A bidirectional LSTM processes the input sequence to capture contextual and
   temporal dependencies.
2. The full output sequence from the LSTM is then passed to a pluggable reduction layer
   (e.g., AttentionLayer) to condense the sequence into a fixed-size summary.
3. Finally, a feed-forward network (e.g., FastKAN) takes this flattened summary and
   produces the final output logits.

This architecture allows for flexible experimentation by easily swapping out the
reduction and FFN components.
"""


import torch
import torch.nn as nn

from abc import ABC
from typing_extensions import override, cast, Type, List, Dict, Optional

from source.models.ffn import AbstractFFN, FastKAN
from source.models.abstract import AbstractSequenceModel
from source.training.utils.hidden_layers import HiddenLayers
from source.models.reduction_layers import AbstractReductionLayer, AttentionLayer
from source.training.utils.collate_functions import per_residue_collate_function

from source.config import ConfigType

from source.custom_types import (
    Encodings_Batch_T,
    AttentionMask_Batch_T,
    Data_T,
    Labels_Batch_T,
    Logits_Batch_T
)



class AbstractLstmReductionHybrid(AbstractSequenceModel, ABC):
    """
    An abstract base class for a hybrid model combining a bidirectional LSTM,
    a reduction layer, and a final feed-forward network (FFN).

    This model processes sequences in three main stages:
    1.  A bidirectional LSTM captures sequential dependencies from the input.
    2.  The full output sequence from the LSTM is normalized and regularized.
    3.  A pluggable reduction layer (e.g., AttentionLayer) intelligently aggregates
        the LSTM output into a fixed-size summary representation.
    4.  A final FFN maps the flattened, reduced representation to the output logits.
    """

    def __init__(self,
            in_channels: int,
            in_seq_len: int,
            reduced_seq_len: int,
            lstm_hidden_size: int,
            lstm_num_layers: int,
            dropout_rate: float,
            hidden_layers: HiddenLayers,
            ffn_layer_class: Type[AbstractFFN],
            reduction_layer_class: Type[AbstractReductionLayer],
            ffn_layer_kwargs: Optional[Dict] = None,
            reduction_layer_kwargs: Optional[Dict] = None,
            out_channels: Optional[int] = None):
        """
        Initializes the AbstractLstmReductionHybrid model.

        Args:
            in_channels: The number of features for each element in the input sequence.
            in_seq_len: The length of the input sequences.
            reduced_seq_len: The target sequence length after the reduction layer.
            lstm_hidden_size: The number of features in the LSTM hidden state.
            lstm_num_layers: The number of recurrent LSTM layers.
            dropout_rate: The dropout rate applied for regularization after the LSTM.
            hidden_layers: The configuration for the hidden layers of the final FFN.
            ffn_layer_class: The class to use for the final feed-forward network.
            reduction_layer_class: The class to use for sequence reduction.
            ffn_layer_kwargs: Optional dictionary of keyword arguments for the FFN class.
            reduction_layer_kwargs: Optional dictionary of keyword arguments for the reduction layer class.
            out_channels: The number of output classes.
        """

        super().__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            in_seq_len = in_seq_len
        )

        self.lstm_hidden_size: int = lstm_hidden_size
        self.lstm_num_layers: int = lstm_num_layers
        self.reduced_seq_len: int = reduced_seq_len
        self.dropout_rate: float = dropout_rate
        self.hidden_layers: HiddenLayers = hidden_layers
        self.ffn_layer_class: Type[AbstractFFN] = ffn_layer_class
        self.ffn_layer_kwargs: Dict = ffn_layer_kwargs if ffn_layer_kwargs else {}
        self.reduction_layer_class: Type[AbstractReductionLayer] = reduction_layer_class
        self.reduction_layer_kwargs: Dict = reduction_layer_kwargs if reduction_layer_kwargs else {}

        # 1. Bi-LSTM Layer to first understand sequential context
        self.lstm = nn.LSTM(
            input_size = self.in_channels,
            hidden_size = self.lstm_hidden_size,
            num_layers = self.lstm_num_layers,
            bidirectional = True,
            batch_first = True
        )

        # 2. Normalization and Dropout for regularization after the LSTM
        self.layer_norm = nn.LayerNorm(self.lstm_hidden_size * 2)
        self.dropout = nn.Dropout(self.dropout_rate)

        # 3. Reduction Layer
        self.reduction_layer = reduction_layer_class(
            in_channels = self.lstm_hidden_size * 2,  # Takes output from bi-LSTM
            in_seq_len = self.in_seq_len,
            out_seq_len = self.reduced_seq_len,
            device = self.device,
            **self.reduction_layer_kwargs
        )

        # 4. KAN for final classification
        self.ffn_layer: AbstractFFN = self.ffn_layer_class(
            in_channels = 1,
            in_seq_len = self.reduced_seq_len * (self.lstm_hidden_size * 2),  # FFN in channels
            hidden_layers = self.hidden_layers,
            out_channels = self.out_channels,
            **self.ffn_layer_kwargs
        )


    @override
    def forward(self, x: Encodings_Batch_T, attention_mask: AttentionMask_Batch_T) -> Logits_Batch_T:
        """
        Defines the forward pass for the hybrid model.

        The process is as follows:
        1. The input sequence is packed and passed through the bidirectional LSTM.
        2. The full output sequence from the LSTM is unpacked, normalized by LayerNorm,
           and regularized with Dropout.
        3. The result is passed through the reduction layer to shorten the sequence.
        4. The reduced sequence is flattened and passed to the FFN to get the final logits.

        Args:
            x: Input batch of sequences, shape: (batch_size, seq_len, in_channels).
            attention_mask: mask for the input sequence.
        """
        # Input shape: (batch_size, seq_len, in_channels)

        # Get actual sequence lengths from the mask to handle padding in the LSTM
        seq_lengths = attention_mask.sum(dim=1).cpu()

        # Pack the sequence to ignore padded elements during LSTM processing
        packed_input = nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)

        # LSTM processing
        packed_lstm_output, _ = self.lstm(packed_input)

        # Unpack the sequence back to its padded tensor form
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_lstm_output, batch_first=True, total_length=self.in_seq_len)
        # lstm_output shape: (batch_size, seq_len, lstm_hidden_size * 2)

        # Apply the residual connection and normalize.
        x_norm = self.layer_norm(lstm_output)

        # Apply dropout to the normalized output.
        x_dropout = self.dropout(x_norm)

        # Reduce the sequence with the learnable ReductionLayer
        # reduced_output shape: (batch, out_seq_len, lstm_hidden_size * 2)
        reduced_output = self.reduction_layer(x_dropout, attention_mask)

        # Flatten the output from the reduction layer before the FFN.
        # Reshape from [batch, seq_len, features] to [batch, seq_len * features]
        flattened_output = torch.flatten(reduced_output, start_dim=1)

        logits = self.ffn_layer(flattened_output)

        return logits


    @override
    def collate_function(self, batch: List[Data_T]) -> tuple[Encodings_Batch_T, Labels_Batch_T, AttentionMask_Batch_T]:
        return per_residue_collate_function(batch, max_seq_len=self.in_seq_len, flatten=False)  # unflattened!



class LstmAttentionReductionHybridFastKAN(AbstractLstmReductionHybrid):
    """
    A concrete implementation of the LstmReductionHybrid that uses
    an AttentionLayer for sequence reduction and a FastKAN FFN for the final classification.
    """

    def __init__(self,
            in_channels: int,
            in_seq_len: int = None,
            out_channels: Optional[int] = None,
            reduced_seq_len: int = None,
            lstm_hidden_size: int = None,
            lstm_num_layers: int = None,
            dropout_rate: float = None,
            hidden_layers: HiddenLayers = None,
            num_heads: int = None,
            grid_diff: int = None,
            num_grids: int = None):
        from source.config import LstmAttentionReductionHybridFastKANConfig
        config: LstmAttentionReductionHybridFastKANConfig = cast(LstmAttentionReductionHybridFastKANConfig, self.get_config())
        self.num_heads: int = num_heads if num_heads else config.num_heads
        self.grid_diff = grid_diff if grid_diff is not None else config.grid_diff
        self.num_grids = num_grids if num_grids is not None else config.num_grids
        super().__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            in_seq_len = in_seq_len if in_seq_len is not None else config.in_seq_len,
            reduced_seq_len = reduced_seq_len if reduced_seq_len is not None else config.reduced_seq_len,
            lstm_hidden_size = lstm_hidden_size if lstm_hidden_size is not None else config.lstm_hidden_size,
            lstm_num_layers = lstm_num_layers if lstm_num_layers is not None else config.lstm_num_layers,
            dropout_rate = dropout_rate if dropout_rate is not None else config.dropout_rate,
            hidden_layers = hidden_layers if hidden_layers is not None else config.hidden_layers,
            ffn_layer_class = FastKAN,
            ffn_layer_kwargs = dict(
                grid_diff = self.grid_diff,
                num_grids = self.num_grids
            ),
            reduction_layer_class = AttentionLayer,
            reduction_layer_kwargs = dict(
                num_heads = self.num_heads
            )
        )


    @classmethod
    @override
    def config_type(cls) -> ConfigType: return ConfigType.LstmAttentionReductionHybridFastKAN