"""
This module defines a hybrid sequence model, `LstmReductionHybrid`,
that combines recurrent, attention, and feed-forward components to process sequence data.

The model is designed as a multi-stage pipeline:
1. A bidirectional LSTM processes the input sequence to capture contextual and
   temporal dependencies.
2. The output from the LSTM is then passed to a pluggable reduction layer
   (e.g., Attention) to condense the sequence into a fixed-size summary.
3. Finally, a feed-forward network (e.g., FastKAN) takes this summary and
   produces the final output logits.

This architecture aims to leverage the strengths of each component:
LSTMs for sequential patterns, attention for targeted information aggregation,
and KANs for effective feature mapping in the final classification/regression step.
"""

import torch
import torch.nn as nn

from typing_extensions import override, cast, Type, List

from source.models.ffn import AbstractFFN, FastKAN
from source.models.abstract import AbstractSequenceModel
from source.models.reduction_layers import AbstractReductionLayer, AttentionLayer
from source.training.utils.collate_functions import per_residue_collate_function
from source.training.utils.hidden_layers import HiddenLayers

from source.config import ConfigType, LstmReductionHybridConfig

from source.custom_types import (
    Encodings_Batch_T,
    AttentionMask_Batch_T,
    Data_T,
    Labels_Batch_T,
    Logits_Batch_T
)



class LstmReductionHybrid(AbstractSequenceModel):
    """
    A hybrid model combining a bidirectional LSTM, a reduction layer, and an FFN.

    This model processes sequences in three main stages:
    1.  A bidirectional LSTM captures sequential dependencies in the input.
    2.  A reduction layer (e.g., Attention) aggregates the LSTM outputs into a
        fixed-size representation. A residual connection with LayerNorm is
        applied before this step to stabilize training.
    3.  A final FFN (e.g., FastKAN) maps the reduced representation to output logits.
    """

    def __init__(self,
            in_channels: int,
            in_seq_len: int = None,
            out_channels: int = None,
            reduced_seq_len: int = None,
            lstm_hidden_size: int = None,
            lstm_num_layers: int = None,
            dropout_rate: float = None,
            hidden_layers: HiddenLayers = None,
            ffn_layer_class: Type[AbstractFFN] = FastKAN,
            reduction_layer_class: Type[AbstractReductionLayer] = AttentionLayer):
        """
        Initializes the LstmReductionHybrid model.

        Args:
            in_channels: The number of features for each element in the input sequence.
            in_seq_len: The length of the input sequences.
            out_channels: The number of output classes.
            reduced_seq_len: The target sequence length after the reduction layer.
            lstm_hidden_size: The number of features in the LSTM hidden state.
            lstm_num_layers: The number of recurrent LSTM layers.
            dropout_rate: The dropout rate applied after the LSTM layer.
            hidden_layers: The configuration for the hidden layers of the final FFN.
            ffn_layer_class: The class to use for the final feed-forward network (e.g., FastKAN).
            reduction_layer_class: The class to use for sequence reduction (e.g., AttentionLayer).
        """
        config: LstmReductionHybridConfig = cast(LstmReductionHybridConfig, self.get_config())

        super().__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            in_seq_len = in_seq_len if in_seq_len is not None else config.in_seq_len
        )

        self.lstm_hidden_size: int = lstm_hidden_size if lstm_hidden_size is not None else config.lstm_hidden_size
        self.lstm_num_layers: int = lstm_num_layers if lstm_num_layers is not None else config.lstm_num_layers
        self.reduced_seq_len: int = reduced_seq_len if reduced_seq_len is not None else config.reduced_seq_len
        self.dropout_rate: float = dropout_rate if dropout_rate is not None else config.dropout_rate
        self.hidden_layers: HiddenLayers = hidden_layers if hidden_layers is not None else config.hidden_layers
        self.ffn_layer_class: Type[AbstractFFN] = ffn_layer_class
        self.reduction_layer_class: Type[AbstractReductionLayer] = reduction_layer_class

        # 1. Bi-LSTM Layer to first understand sequential context
        self.lstm = nn.LSTM(
            input_size = self.in_channels,
            hidden_size = self.lstm_hidden_size,
            num_layers = self.lstm_num_layers,
            bidirectional = True,
            batch_first = True
        )

        # 2. Dropout for regularization after the LSTM
        self.dropout = nn.Dropout(self.dropout_rate)

        # 3. AttentionLayer for intelligent reduction
        self.reduction_layer = reduction_layer_class(
            in_channels = self.lstm_hidden_size * 2,  # Takes output from bi-LSTM
            in_seq_len = self.in_seq_len,
            out_seq_len = self.reduced_seq_len,
            device = self.device
        )
        self.layer_norm = nn.LayerNorm(self.lstm_hidden_size * 2)

        # 4. KAN for final classification
        self.ffn_layer: AbstractFFN = self.ffn_layer_class(
            in_channels = 1,
            in_seq_len = self.reduced_seq_len * (self.lstm_hidden_size * 2),  # FFN in channels
            hidden_layers = self.hidden_layers,
            out_channels = self.out_channels
        )


    @classmethod
    @override
    def config_type(cls) -> ConfigType: return ConfigType.LstmReductionHybrid


    @override
    def forward(self, x: Encodings_Batch_T, attention_mask: AttentionMask_Batch_T) -> Logits_Batch_T:
        """
        Defines the forward pass for the hybrid model.

        The process is as follows:
        1. The input sequence is passed through the bidirectional LSTM.
        2. Dropout is applied to the LSTM output for regularization.
        3. A residual connection from the original LSTM output is added to the
           dropout-modified output, followed by Layer Normalization. This
           stabilizes training.
        4. The result is passed through the reduction layer to shorten the sequence.
        5. The reduced sequence is flattened and passed to the FFN to get logits.

        Args:
            x: Input batch of sequences, shape: (batch_size, seq_len, in_channels).
            attention_mask: mask for the input sequence.

        Returns:
            The final output logits from the FFN.
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

        # Apply Dropout for regularization
        x_dropout = self.dropout(lstm_output)

        # Add a residual connection and LayerNorm (recommended)
        # This helps stabilize training by allowing the model to bypass the reduction if needed
        x_norm = self.layer_norm(x_dropout + lstm_output)  # Pre-dropout output for the residual

        # Reduce the sequence with the learnable ReductionLayer
        # reduced_output shape: (batch, out_seq_len, lstm_hidden_size * 2)
        reduced_output = self.reduction_layer(x_norm, attention_mask)

        # Flatten the output from the reduction layer before the FFN.
        # Reshape from [batch, seq_len, features] to [batch, seq_len * features]
        flattened_output = torch.flatten(reduced_output, start_dim=1)

        logits = self.ffn_layer(flattened_output)

        return logits


    @override
    def collate_function(self, batch: List[Data_T]) -> tuple[Encodings_Batch_T, Labels_Batch_T, AttentionMask_Batch_T]:
        return per_residue_collate_function(batch, max_seq_len=self.in_seq_len, flatten=False)  # unflattened!