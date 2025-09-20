"""
This module provides an abstract base class for the Light Attention model,
a neural architecture designed for protein localization prediction based on the paper:
"Light attention predicts protein location from the language of life" by Hannes Stark et al.

The implementation is based on the official code release:
https://github.com/HannesStark/protein-localization/blob/master/models/light_attention.py

The module includes:
- AbstractLightAttention: An ABC that defines the core Light Attention mechanism.
- LightAttentionFFN: A specific FFN designed to replicate the classifier from the original paper.
- LightAttentionLAMLP: A concrete implementation that combines AbstractLightAttention with the original LightAttentionFFN.
- LightAttentionFastKAN: A concrete implementation that combines AbstractLightAttention with FastKAN.
"""


import torch
import torch.nn as nn

from abc import ABC
from typing_extensions import override, cast, Type, Optional, List, Tuple

from source.models.ffn import AbstractFFN, FastKAN
from source.models.abstract import AbstractSequenceModel
from source.training.utils.hidden_layers import HiddenLayers
from source.training.utils.collate_functions import per_residue_collate_function
from source.config import ConfigType

from source.custom_types import (
    Encodings_Batch_T,
    AttentionMask_Batch_T,
    Logits_Batch_T, Data_T, Labels_Batch_T
)


class LightAttentionFFN(AbstractFFN):
    """
    A specific Feed-Forward Network designed to match the classifier in the original LightAttention implementation.

    This FFN consists of blocks of `Linear -> Dropout -> ReLU -> BatchNorm1d` for each hidden layer,
    followed by a final linear layer for the output.
    This structure is explicitly defined to replicate the architecture described in the source paper.
    """

    def __init__(self,
                 in_channels: int,
                 hidden_layers: HiddenLayers = None,
                 dropout_rate: float = None,
                 out_channels: Optional[int] = None,
                 **_):
        """
        Initializes the LightAttentionFFN.

        Args:
            in_channels: The number of input features.
            hidden_layers: Configuration for the hidden layers.
            dropout_rate: Dropout rate to be used within the FFN blocks.
            out_channels: The number of output classes.
        """
        from source.config import LightAttentionFFNConfig
        config: LightAttentionFFNConfig = cast(LightAttentionFFNConfig, self.get_config())
        self.dropout_rate: float = dropout_rate if dropout_rate is not None else config.dropout_rate

        # FFN takes a single flat vector as an input.
        super().__init__(
            in_channels = in_channels,
            in_seq_len = 1,
            hidden_layers = hidden_layers if hidden_layers is not None else config.hidden_layers,
            out_channels = out_channels
        )

    
    @classmethod
    @override
    def config_type(cls) -> ConfigType: return ConfigType.LightAttentionFFN


    @override
    def ffn_model_supplier(self, layers: List[int]) -> nn.Module:
        """
        Builds the sequential model based on the Light Attention paper's design.

        For each hidden layer transition, it creates a block of:
        `Linear -> Dropout -> ReLU -> BatchNorm1d`

        The final transition is a single Linear layer for the output logits.
        For example, `layers=[in_dim, 32, out_dim]` will produce the exact
        classifier architecture from the paper.

        Args:
            layers (List[int]): A list of layer sizes, starting with the input dimension,
                                followed by hidden layer dimensions, and ending with the
                                output dimension.
                                E.g., [input_dim, hidden_1, hidden_2, output_dim]
        """
        modules = []
        num_transitions = len(layers) - 1

        for i, (in_dim, out_dim) in enumerate(zip(layers[:-1], layers[1:])):
            modules.append(nn.Linear(in_dim, out_dim))

            # Add the Dropout, ReLU, and BatchNorm block for all layers except the last one
            is_output_layer = (i == num_transitions - 1)
            if not is_output_layer:
                modules.append(nn.Dropout(self.dropout_rate))
                modules.append(nn.ReLU())
                modules.append(nn.BatchNorm1d(out_dim))

        return nn.Sequential(*modules)



class AbstractLightAttention(AbstractSequenceModel, ABC):
    """
    An abstract base class for the Light Attention (LA) model from the paper:
    "Light attention predicts protein location from the language of life".

    The model uses two parallel 1D convolutional layers to process sequence embeddings.
    One layer generates attention weights, and the other generates values.
    The final representation is a combination of the attention-weighted sum values and a max-pooled feature vector.

    Based on https://github.com/HannesStark/protein-localization/blob/master/models/light_attention.py
    """

    def __init__(self,
            in_channels: int,
            in_seq_len: int,
            out_channels: Optional[int],
            kernel_size: int,
            conv_dropout_rate: float,
            hidden_layers: HiddenLayers,
            ffn_layer_class: Type[AbstractFFN],
            ffn_layer_kwargs: Optional[dict]):
        """
        Initializes the abstract LightAttention model.

        Args:
            in_channels: The number of embedding dimensions for each sequence element.
            in_seq_len: The length of the input sequences.
            kernel_size: The kernel size for the 1D convolutions.
            conv_dropout_rate: The dropout rate applied after the value convolution.
            hidden_layers: The configuration for the hidden layers of the final FFN.
            ffn_layer_class: The class to use for the final feed-forward network (e.g., MLP, FastKAN).
            ffn_layer_kwargs: Optional dictionary of keyword arguments for the FFN class.
            out_channels: The number of output classes.
        """

        super().__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            in_seq_len = in_seq_len
        )

        self.kernel_size = kernel_size
        self.conv_dropout_rate = conv_dropout_rate
        self.hidden_layers = hidden_layers
        self.ffn_layer_class = ffn_layer_class
        self.ffn_layer_kwargs = ffn_layer_kwargs if ffn_layer_kwargs is not None else {}

        # This convolution learns the "values" (v)
        self.feature_convolution = nn.Conv1d(
            in_channels = self.in_channels,
            out_channels = self.in_channels,
            kernel_size = self.kernel_size,
            padding = self.kernel_size // 2  # Preserves sequence length
        )

        # This convolution learns the "attention coefficients" (e)
        self.attention_convolution = nn.Conv1d(
            in_channels = self.in_channels,
            out_channels = self.in_channels,
            kernel_size = self.kernel_size,
            padding = self.kernel_size // 2  # Preserves sequence length
        )

        self.conv_dropout = nn.Dropout(self.conv_dropout_rate)
        self.softmax = nn.Softmax(dim=-1)

        # The final classifier FFN
        self.ffn_layer: AbstractFFN = ffn_layer_class(
            in_channels = self.in_channels * 2,  # self.feature_convolution.out_channels
            in_seq_len  = 1,
            hidden_layers = self.hidden_layers,
            out_channels = self.out_channels,
            **self.ffn_layer_kwargs
        )


    @override
    def forward(self, x: Encodings_Batch_T, attention_mask: AttentionMask_Batch_T) -> Logits_Batch_T:
        """
        Performs the forward pass of the LightAttention model.

        Args:
            x (torch.Tensor): Input embeddings [batch_size, sequence_length, embeddings_dim].
                              The model expects [batch_size, embeddings_dim, sequence_length],
                              so we permute it.
            attention_mask (torch.Tensor): Boolean mask [batch_size, sequence_length]
                                           where True is a real token and False is padding.

        Returns:
            torch.Tensor: The output logits of shape [batch_size, out_channels].
        """
        # (batch, seq, embed) -> (batch, embed, seq) for Conv1d
        x = x.permute(0, 2, 1)

        v = self.feature_convolution(x)
        v = self.conv_dropout(v)
        e = self.attention_convolution(x)

        if attention_mask is not None:
            # Add a dimension for broadcasting: [batch_size, 1, sequence_length]
            broadcastable_mask = attention_mask.unsqueeze(1)

            # Use the smallest value for the tensor's dtype to prevent overflow with float16
            fill_value = torch.finfo(e.dtype).min
            e.masked_fill_(broadcastable_mask == False, fill_value)

        alpha = self.softmax(e)
        x_prime = torch.sum(v * alpha, dim=-1)
        v_max, _ = torch.max(v, dim=-1)
        z = torch.cat([x_prime, v_max], dim=-1)
        logits = self.ffn_layer(z)
        return logits


    @override
    def collate_function(self, batch: List[Data_T]) -> Tuple[Encodings_Batch_T, Labels_Batch_T, AttentionMask_Batch_T]:
        return per_residue_collate_function(batch, max_seq_len=self.in_seq_len, flatten=False)



class LightAttentionLAMLP(AbstractLightAttention):
    """
    A concrete implementation of the original Light Attention architecture.

    This class combines the `AbstractLightAttention` base with the `LightAttentionFFN`,
    faithfully replicating the model design from the source paper.
    """

    def __init__(self,
            in_channels: int,
            in_seq_len: int,
            out_channels: Optional[int] = None,
            kernel_size: int = None,
            conv_dropout_rate: float = None,
            hidden_layers: HiddenLayers = None,
            ffn_dropout_rate: float = None):
        from source.config import LightAttentionLAMLPConfig
        config: LightAttentionLAMLPConfig = cast(LightAttentionLAMLPConfig, self.get_config())
        self.ffn_dropout_rate = ffn_dropout_rate if ffn_dropout_rate is not None else config.ffn_dropout_rate
        super().__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            in_seq_len = in_seq_len if in_seq_len is not None else config.in_seq_len,
            kernel_size = kernel_size if kernel_size is not None else config.kernel_size,
            conv_dropout_rate = conv_dropout_rate if kernel_size is not None else config.conv_dropout_rate,
            hidden_layers = hidden_layers if hidden_layers is not None else config.hidden_layers,
            ffn_layer_class = LightAttentionFFN,
            ffn_layer_kwargs = dict(
                dropout_rate = self.ffn_dropout_rate
            )
        )


    @classmethod
    @override
    def config_type(cls) -> ConfigType: return ConfigType.LightAttentionLAMLP



class LightAttentionFastKAN(AbstractLightAttention):
    """
    A concrete implementation of `AbstractLightAttention` that replaces the
    original FFN classifier with a `FastKAN`.

    This allows for a direct comparison of the `LightAttentionFFN` against a
    `FastKAN` in the final classification stage.
    """

    def __init__(self,
            in_channels: int,
            in_seq_len: int = None,
            out_channels: Optional[int] = None,
            kernel_size: int = None,
            conv_dropout_rate: float = None,
            hidden_layers: HiddenLayers = None,
            grid_diff: int = None,
            num_grids: int = None):
        from source.config import LightAttentionFastKANConfig
        config: LightAttentionFastKANConfig = cast(LightAttentionFastKANConfig, self.get_config())
        self.grid_diff = grid_diff if grid_diff is not None else config.grid_diff
        self.num_grids = num_grids if num_grids is not None else config.num_grids
        super().__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            in_seq_len = in_seq_len if in_seq_len is not None else config.in_seq_len,
            kernel_size = kernel_size if kernel_size is not None else config.kernel_size,
            conv_dropout_rate = conv_dropout_rate if kernel_size is not None else config.conv_dropout_rate,
            hidden_layers = hidden_layers if hidden_layers is not None else config.hidden_layers,
            ffn_layer_class = FastKAN,
            ffn_layer_kwargs = dict(
                grid_diff = self.grid_diff,
                num_grids = self.num_grids
            )
        )


    @classmethod
    @override
    def config_type(cls) -> ConfigType: return ConfigType.LightAttentionFastKAN