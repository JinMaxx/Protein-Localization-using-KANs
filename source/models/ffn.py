"""
This module provides abstract and concrete implementations for Feed-Forward Network (FFN) models,
building upon the abstract model framework.

The classes are designed to handle sequence data by first flattening it, then
passing it through a series of dense layers.
The module includes support for standard MLPs and modern architectures like KANs,
with built-in support for memory-saving techniques like gradient checkpointing.

- `AbstractFFN`: The base class for all FFNs, handling layer size calculations.
- `AbstractCheckpointFFN`: Extends `AbstractFFN` to add gradient checkpointing.
- `FastKAN`: A concrete implementation using the Fast KAN architecture.
- `MLP`: A concrete implementation of a standard Multi-Layer Perceptron.
- `MLPpp`: A specialized MLP for per-protein analysis with a custom data collator.
"""

from typing import Optional

import torch.nn as nn

from abc import ABC, abstractmethod

from torch.utils import checkpoint as cp
from typing_extensions import List, override, Tuple, cast, Sequence

from source.models.abstract import AbstractSequenceModel
from source.training.utils.hidden_layers import HiddenLayers
from source.data_scripts.read_data import num_classes
from source.training.utils.collate_functions import per_protein_collate_function, per_residue_collate_function

from source.config import (
    AbstractFFNConfig,
    ConfigType,
    FastKANConfig
)

from source.custom_types import (
    Encodings_Batch_T,
    AttentionMask_Batch_T,
    Data_T,
    Labels_Batch_T,
    Logits_Batch_T
)



class AbstractFFN(AbstractSequenceModel, ABC):
    """
    Abstract base class for Feed-Forward Network models that operate on flattened sequence data.

    This class handles the common logic for FFNs,
    such as calculating the input dimension from sequence length and channel count,
    and resolving the `HiddenLayers` configuration into a concrete list of layer sizes.
    """

    def __init__(self,
            in_channels: int,
            in_seq_len: int,
            hidden_layers: HiddenLayers,
            out_channels: int = num_classes,
            **_):
        """
        Initializes the AbstractFFN.

        :param in_channels: The number of features per token in the input sequence.
        :param in_seq_len: The length of the input sequences.
        :param hidden_layers: A `HiddenLayers` object defining the hidden layer architecture.
        :param out_channels: The number of output classes.
        """
        super().__init__(
            in_channels = in_channels,
            in_seq_len  = in_seq_len,
            out_channels = out_channels
        )

        self.input_dim: int = self.in_channels * self.in_seq_len
        # print(f"in_channels: {self.in_channels} * in_seq_len: {self.in_seq_len} = {self.in_channels * self.in_seq_len}")

        self.hidden_layers: HiddenLayers = hidden_layers
        self._hidden_layers_exact: List[int] = hidden_layers.calculate_layers(input_dim=self.input_dim)
        self.layers: List[int] = ([self.input_dim] +
                                  self._hidden_layers_exact +
                                  [self.out_channels])
        print(f"Layers: {self.layers}")

        self.ffn_layer: nn.Module = self.ffn_model_supplier(layers=self.layers).to(self.device)


    @staticmethod
    @abstractmethod
    def ffn_model_supplier(layers: List[int]) -> nn.Module:
        """
        A factory method that subclasses must implement to construct the FFN.

        :param layers: A list of integers defining the network's dimensions,
                       starting with the input dimension and ending with the output dimension.
        :return: An `nn.Module` instance representing the feed-forward network.
        """
        raise NotImplementedError


    @override
    def forward(self, x: Encodings_Batch_T, **_) -> Logits_Batch_T:
        """
        Performs the forward pass by feeding the input through the FFN.

        :param x: The input batch of flattened sequence encodings.
        :return: The output logits from the model.
        """
        logits = self.ffn_layer(x)
        return logits


    @classmethod
    @override
    def get_config(cls, force_reload: bool = False) -> AbstractFFNConfig:
        """Retrieves and caches the parsed FFN-specific configuration."""
        return cast(AbstractFFNConfig, super().get_config(force_reload=force_reload))


    @override
    def collate_function(self, batch: List[Data_T]) -> Tuple[Encodings_Batch_T, Labels_Batch_T, AttentionMask_Batch_T]:
        """
        A default collate function for FFN models.

        It pads/truncates all sequences in a batch to `self.in_seq_len` and
        flattens the resulting tensor for consumption by dense layers.

        :param batch: A list of data samples.
        :return: A tuple of batched encodings, labels, and an attention mask.
        """
        return per_residue_collate_function(batch, self.in_seq_len, flatten=True)



# To save memory consumption during training
class AbstractCheckpointFFN(AbstractFFN, ABC):
    """An abstract FFN that uses gradient checkpointing to save memory on CUDA devices."""

    def __init__(self,
            in_channels: int,
            in_seq_len: int,
            hidden_layers: HiddenLayers,
            out_channels: Optional[int] = None):
        super().__init__(
            in_channels = in_channels,
            in_seq_len  = in_seq_len,
            hidden_layers = hidden_layers,
            out_channels = out_channels
        )
                               # ↓ Input layer
        self.num_chunks = 1 + len(hidden_layers)  # Divide the model into chunks for checkpointing
        # Should ignore the last layer


    @override
    def forward(self, x: Encodings_Batch_T, **_) -> Logits_Batch_T:
        """
        Performs the forward pass using gradient checkpointing if on a CUDA device.
        This trades computation for a significant reduction in memory usage.
        """
        logits = cp.checkpoint_sequential(
            self.layers_seq(),  # Sequence(nn.Module)
            self.num_chunks,
            x,
            use_reentrant = False
        ) if self.device.type == "cuda" else self.ffn_layer(x)  # No improvements when using cpu.
        return logits


    @abstractmethod
    def layers_seq(self) -> Sequence[nn.Module]:
        """
        Provides the sequence of modules required for `checkpoint_sequential`.

        :return: A sequence (e.g., list) of `nn.Module` objects.
        """
        pass



# Old naming. Maybe change to FasterKAN...
class FastKAN(AbstractFFN):  # AbstractCheckpointFFN
    """
    A concrete FFN implementation using the Fast Kolmogorov-Arnold Network (KAN)
    implementation.
    Based on https://github.com/ZiyaoLi/fast-kan.git
    Using improved version https://github.com/AthanasiosDelis/faster-kan.git
    """  # with support for gradient checkpointing.
    # from fastkan import FastKAN as __KAN_impl
    from fasterkan import FasterKAN as __KAN_impl  # 1.5x faster

    def __init__(self,
            in_channels: int,
            in_seq_len: int = None,
            hidden_layers: HiddenLayers = None,
            out_channels: int = num_classes,
            grid_diff = None,
            num_grids = None
        ):
        config: FastKANConfig = cast(FastKANConfig, self.get_config())

        super().__init__(
            in_channels = in_channels,
            in_seq_len  = in_seq_len if in_seq_len is not None else config.in_seq_len,
            hidden_layers = hidden_layers if hidden_layers is not None else config.hidden_layers,
            out_channels = out_channels
        )
        self.grid_diff = grid_diff if grid_diff is not None else config.grid_diff
        self.grid_min = -self.grid_diff
        self.grid_max = self.grid_diff
        self.num_grids = num_grids if num_grids is not None else config.num_grids


    @classmethod
    @override
    def config_type(cls) -> ConfigType: return ConfigType.FastKAN


    @classmethod
    @override
    def ffn_model_supplier(cls,
            layers: List[int],
            grid_diff: int = None,
            num_grids: int = None
    ) -> nn.Module:
        config: FastKANConfig = cast(FastKANConfig, cls.get_config())
        return FastKAN.__KAN_impl(
            layers_hidden = layers,
            grid_min  = -grid_diff if grid_diff is not None else config.grid_diff,
            grid_max  =  grid_diff if grid_diff is not None else config.grid_diff,
            num_grids =  num_grids if num_grids is not None else config.num_grids,
        )


    # @override
    # def layers_seq(self) -> Sequence[nn.Module]:
    #     """Returns the internal sequence of layers from the FastKAN object for checkpointing."""
    #     return cast(FastKAN.__KAN_impl, self.ffn_layer).layers



class MLP(AbstractCheckpointFFN):  # No Checkpoints because backpropagation is slow...?
    """
    A standard Multi-Layer Perceptron (MLP) with ReLU activations,
    supporting gradient checkpointing used for comparison.
    """

    def __init__(self,
            in_channels: int,
            in_seq_len: int = None,
            hidden_layers: HiddenLayers = None,
            out_channels: int = num_classes):
        config = self.get_config()

        super().__init__(
            in_channels = in_channels,
            in_seq_len  = in_seq_len if in_seq_len is not None else config.in_seq_len,
            hidden_layers = hidden_layers if hidden_layers is not None else config.hidden_layers,
            out_channels = out_channels
        )


    @classmethod
    @override
    def config_type(cls) -> ConfigType: return ConfigType.MLP


    @staticmethod
    @override
    def ffn_model_supplier(layers: List[int]) -> nn.Module:
        """Builds an `nn.Sequential` model of Linear layers and ReLU activations."""
        modules = []
        for in_dim, out_dim in zip(layers[:-1], layers[1:]):
            modules.append(nn.Linear(in_dim, out_dim))
            # Add activation for all but the last layer
            if out_dim != layers[-1]:
                modules.append(nn.ReLU())
        return nn.Sequential(*modules)


    @override
    def layers_seq(self) -> Sequence[nn.Module]: return list(self.ffn_layer.children())



# Baseline models

class MLPpp(MLP):  # Per Protein
    """
    A specialized "Per-Protein" MLP that uses a custom data collation strategy.

    This model is designed for scenarios where each protein sequence is treated as a
    single feature vector, rather than a sequence of token embeddings.
    """

    def __init__(self,
            in_seq_len: int = None,
            hidden_layers: HiddenLayers = None,
            out_channels: int = num_classes,
            **_):
        """
        Initializes the Per-Protein MLP.

        Note: `in_channels` is hardcoded to 1, as the per-protein collate
        function produces a single channel.
        """
        config = self.get_config()

        super().__init__(
            in_channels = 1,
            in_seq_len = in_seq_len if in_seq_len is not None else config.in_seq_len,
            hidden_layers = hidden_layers if hidden_layers is not None else config.hidden_layers,
            out_channels = out_channels
        )


    @classmethod
    @override
    def config_type(cls) -> ConfigType: return ConfigType.MLP_PP


    @override
    def collate_function(self, batch: List[Data_T]) -> Tuple[Encodings_Batch_T, Labels_Batch_T, AttentionMask_Batch_T]:
        """
        Overrides the default collation to use the `per_protein_collate_function`, which treats each sequence as a single sample.
        """
        return per_protein_collate_function(batch)