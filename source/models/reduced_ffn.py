"""
This module defines a framework for building sequence models that first reduce
the dimensionality of an input sequence and then process the result with a Feed-Forward Network (FFN).

The core component is the `AbstractReducedFFN` class,
which combines a pluggable`AbstractReductionLayer` with a pluggable `AbstractFFN`.
This abstract base class handles the orchestration of
passing data through the reduction and FFN stages.

The module also provides several concrete implementations of this framework.
These classes pair specific reduction techniques (like Max Pooling, Attention, or U-Net)
with specific FFNs (like FastKAN or a standard MLP), creating a catalog of ready-to-use models.
Each concrete model is configured via its associated `ConfigType`.
"""

from abc import ABC

from typing_extensions import Type, Optional, List, Dict, override, cast

from source.models.ffn import AbstractFFN, FastKAN, MLP
from source.models.abstract import AbstractSequenceModel
from source.training.utils.hidden_layers import HiddenLayers
from source.training.utils.collate_functions import per_residue_collate_function  # , per_protein_collate_function
from source.config import ConfigType, AbstractReducedFFNConfig, ReducedFastKANConfig
from source.models.reduction_layers import (
    AbstractReductionLayer,
    PositionalWeightedConvLayer,
    MaxPoolConvLayer,
    AvgPoolConvLayer,
    LinearReductionLayer,
    UNetReductionLayer,
    AttentionLayer
)



from source.custom_types import (
    Encodings_Batch_T,
    Encodings_Batch_PerResidue_T,
    AttentionMask_Batch_T,
    Data_T,
    Labels_Batch_T,
    Logits_Batch_T
)



class AbstractReducedFFN(AbstractSequenceModel, ABC):
    """
    An abstract base model that combines a reduction layer with a feed-forward network.

    This class serves as a generic framework for models that first reduce the
    dimensionality of a sequence (e.g., its length or channels)
    and then process the reduced representation through an FFN to produce logits.

    The specific reduction and FFN layers are injected as class types during
    initialization, allowing for flexible composition of different model architectures.
    """

    def __init__(self,
            in_channels: int,
            in_seq_len: int,
            reduced_seq_len: int,
            hidden_layers: HiddenLayers,
            ffn_layer_class: Type[AbstractFFN],
            reduction_layer_class: Type[AbstractReductionLayer],
            ffn_layer_kwargs: Optional[Dict] = None,
            reduction_layer_kwargs: Optional[Dict] = None,
            out_channels: Optional[int] = None,
            reduced_channels: Optional[int] = None):
        super().__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            in_seq_len = in_seq_len
        )
        
        self.reduced_seq_len: int = reduced_seq_len
        self.reduced_channels: int = reduced_channels if reduced_channels is not None else self.in_channels
        self.hidden_layers: HiddenLayers = hidden_layers
        self.ffn_model_class: Type[AbstractFFN] = ffn_layer_class
        self.ffn_layer_kwargs: Dict = ffn_layer_kwargs if ffn_layer_kwargs is not None else {}
        self.reduction_layer_class: Type[AbstractReductionLayer] = reduction_layer_class
        self.reduction_layer_kwargs: Dict = reduction_layer_kwargs if reduction_layer_kwargs is not None else {}

        self.reduction_layer: AbstractReductionLayer = \
            reduction_layer_class(
                in_channels = self.in_channels,
                in_seq_len = self.in_seq_len,
                out_seq_len = self.reduced_seq_len,
                out_channels = self.reduced_channels,
                device = self.device,
                **self.reduction_layer_kwargs
            )

        # More dynamically setting parameters with the reduction layer deciding them.
        self.ffn_layer: AbstractFFN = ffn_layer_class(
            in_channels = self.reduction_layer.out_channels,
            in_seq_len  = self.reduction_layer.out_seq_len,
            hidden_layers = self.hidden_layers,
            out_channels = self.out_channels,
            **self.ffn_layer_kwargs
        )


    @override
    def forward(self, x: Encodings_Batch_PerResidue_T, attention_mask: AttentionMask_Batch_T) -> Logits_Batch_T:
        """
        Defines the forward pass of the model.

        Args:
            x: The input batch of per-residue encodings.
            attention_mask: An attention mask.

        Returns:
            The output logits from the FFN.
        """
        x_reduced = self.reduction_layer(x, attention_mask)  # [batch_size, reduced_seq_len, reduced_channels]
        current_batch_size = x_reduced.size(0)
        x_flatten = x_reduced.reshape(current_batch_size, -1) # Flatten encodings and seq dimension
        logits = self.ffn_layer(x_flatten)
        return logits


    @override
    def collate_function(self, batch: List[Data_T]) -> tuple[Encodings_Batch_T, Labels_Batch_T, AttentionMask_Batch_T]:
        return per_residue_collate_function(batch, max_seq_len=self.in_seq_len, flatten=False)  # unflattened!


    @classmethod
    @override
    def get_config(cls, force_reload: bool = False) -> AbstractReducedFFNConfig:
        return cast(AbstractReducedFFNConfig, super().get_config(force_reload=force_reload))



class AbstractReducedFastKAN(AbstractReducedFFN, ABC):

    def __init__(self,
            in_channels: int,
            in_seq_len: int,
            reduced_seq_len: int,
            hidden_layers: HiddenLayers,
            reduction_layer_class: Type[AbstractReductionLayer],
            reduction_layer_kwargs: Optional[Dict] = None,
            out_channels: Optional[int] = None,
            reduced_channels: Optional[int] = None,
            grid_diff: Optional[int] = None,
            num_grids: Optional[int] = None):

        config = self.get_config()

        self.grid_diff = grid_diff if grid_diff is not None else config.grid_diff
        self.num_grids = num_grids if num_grids is not None else config.num_grids

        super().__init__(
            in_channels = in_channels,
            in_seq_len = in_seq_len,
            out_channels = out_channels,
            reduced_seq_len = reduced_seq_len,
            reduced_channels = reduced_channels,
            hidden_layers = hidden_layers,
            reduction_layer_class = reduction_layer_class,
            reduction_layer_kwargs = reduction_layer_kwargs,
            ffn_layer_class = FastKAN,
            ffn_layer_kwargs = dict(
                grid_diff = self.grid_diff,
                num_grids = self.num_grids,
            )
        )


    @classmethod
    @override
    def get_config(cls, force_reload: bool = False) -> ReducedFastKANConfig:
        return cast(ReducedFastKANConfig, super().get_config(force_reload=force_reload))



# noinspection DuplicatedCode
class MaxPoolFastKAN(AbstractReducedFastKAN):
    """A model combining a `MaxPoolConvLayer` for reduction and a `FastKAN` FFN."""

    def __init__(self,
            in_channels: int,
            in_seq_len: int = None,
            out_channels: Optional[int] = None,
            reduced_seq_len: int = None,
            hidden_layers: HiddenLayers = None,
            grid_diff: int = None,
            num_grids: int = None):
        
        config = self.get_config()
        
        super().__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            reduced_channels = in_channels,  # in_channels == out_channels
            in_seq_len = in_seq_len if in_seq_len is not None else config.in_seq_len,
            reduced_seq_len = reduced_seq_len if reduced_seq_len is not None else config.reduced_seq_len,
            hidden_layers = hidden_layers if hidden_layers is not None else config.hidden_layers,
            reduction_layer_class = MaxPoolConvLayer,
            grid_diff = grid_diff if grid_diff is not None else config.grid_diff,
            num_grids = num_grids if num_grids is not None else config.num_grids
        )


    @classmethod
    @override
    def config_type(cls) -> ConfigType: return ConfigType.MaxPoolFastKAN


# noinspection DuplicatedCode
class MaxPoolMLP(AbstractReducedFFN):
    """A model combining a `MaxPoolConvLayer` for reduction and a standard `MLP` FFN."""

    def __init__(self,
            in_channels: int,
            in_seq_len: int = None,
            out_channels: Optional[int] = None,
            reduced_seq_len: int = None,
            hidden_layers: HiddenLayers = None):

        config = self.get_config()

        super().__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            reduced_channels = in_channels,  # in_channels == out_channels
            in_seq_len = in_seq_len if in_seq_len is not None else config.in_seq_len,
            reduced_seq_len = reduced_seq_len if reduced_seq_len is not None else config.reduced_seq_len,
            hidden_layers = hidden_layers if hidden_layers is not None else config.hidden_layers,
            reduction_layer_class = MaxPoolConvLayer,
            ffn_layer_class = MLP,
        )

    @classmethod
    @override
    def config_type(cls) -> ConfigType: return ConfigType.MaxPoolMLP



# noinspection DuplicatedCode
class AvgPoolFastKAN(AbstractReducedFastKAN):
    """A model combining an `AvgPoolConvLayer` for reduction and a `FastKAN` FFN."""

    def __init__(self,
            in_channels: int,
            in_seq_len: int = None,
            out_channels: Optional[int] = None,
            reduced_seq_len: int = None,
            hidden_layers: HiddenLayers = None,
            grid_diff: int = None,
            num_grids: int = None):

        config = self.get_config()

        super().__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            reduced_channels = in_channels,  # in_channels == out_channels
            in_seq_len = in_seq_len if in_seq_len is not None else config.in_seq_len,
            reduced_seq_len = reduced_seq_len if reduced_seq_len is not None else config.reduced_seq_len,
            hidden_layers = hidden_layers if hidden_layers is not None else config.hidden_layers,
            reduction_layer_class = AvgPoolConvLayer,
            grid_diff = grid_diff if grid_diff is not None else config.grid_diff,
            num_grids = num_grids if num_grids is not None else config.num_grids
        )


    @classmethod
    @override
    def config_type(cls) -> ConfigType: return ConfigType.AvgPoolFastKAN



# noinspection DuplicatedCode
class AvgPoolMLP(AbstractReducedFFN):
    """A model combining an `AvgPoolConvLayer` for reduction and a standard `MLP` FFN."""

    def __init__(self,
            in_channels: int,
            in_seq_len: int = None,
            out_channels: Optional[int] = None,
            reduced_seq_len: int = None,
            hidden_layers: HiddenLayers = None):

        config = self.get_config()

        super().__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            reduced_channels = in_channels,  # in_channels == out_channels
            in_seq_len = in_seq_len if in_seq_len is not None else config.in_seq_len,
            reduced_seq_len = reduced_seq_len if reduced_seq_len is not None else config.reduced_seq_len,
            hidden_layers = hidden_layers if hidden_layers is not None else config.hidden_layers,
            reduction_layer_class = AvgPoolConvLayer,
            ffn_layer_class = MLP,
        )


    @classmethod
    @override
    def config_type(cls) -> ConfigType: return ConfigType.AvgPoolMLP



# noinspection DuplicatedCode
class LinearFastKAN(AbstractReducedFastKAN):
    """A model combining a `LinearReductionLayer` and a `FastKAN` FFN."""

    def __init__(self,
            in_channels: int,
            in_seq_len: int = None,
            out_channels: Optional[int] = None,
            reduced_seq_len: int = None,
            hidden_layers: HiddenLayers = None,
            grid_diff: int = None,
            num_grids: int = None):

        config = self.get_config()
        
        super().__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            reduced_channels = in_channels,  # in_channels == out_channels
            in_seq_len = in_seq_len if in_seq_len is not None else config.in_seq_len,
            reduced_seq_len = reduced_seq_len if reduced_seq_len is not None else config.reduced_seq_len,
            hidden_layers = hidden_layers if hidden_layers is not None else config.hidden_layers,
            reduction_layer_class = LinearReductionLayer,
            grid_diff = grid_diff if grid_diff is not None else config.grid_diff,
            num_grids = num_grids if num_grids is not None else config.num_grids
        )


    @classmethod
    @override
    def config_type(cls) -> ConfigType: return ConfigType.LinearFastKAN



# noinspection DuplicatedCode
class LinearMLP(AbstractReducedFFN):
    """A model combining a `LinearReductionLayer` for reduction and a standard `MLP` FFN."""

    def __init__(self,
            in_channels: int,
            in_seq_len: int = None,
            out_channels: Optional[int] = None,
            reduced_seq_len: int = None,
            hidden_layers: HiddenLayers = None):

        config = self.get_config()

        super().__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            reduced_channels = in_channels,  # in_channels == out_channels
            in_seq_len = in_seq_len if in_seq_len is not None else config.in_seq_len,
            reduced_seq_len = reduced_seq_len if reduced_seq_len is not None else config.reduced_seq_len,
            hidden_layers = hidden_layers if hidden_layers is not None else config.hidden_layers,
            reduction_layer_class = LinearReductionLayer,
            ffn_layer_class = MLP,
        )


    @classmethod
    @override
    def config_type(cls) -> ConfigType: return ConfigType.LinearMLP



# noinspection DuplicatedCode
class AttentionFastKAN(AbstractReducedFastKAN):
    """A model combining an `AttentionLayer` for reduction and a `FastKAN` FFN."""

    def __init__(self,
            in_channels: int,
            in_seq_len: int = None,
            out_channels: Optional[int] = None,
            reduced_seq_len: int = None,
            hidden_layers: HiddenLayers = None,
            num_heads: int = None,
            grid_diff: int = None,
            num_grids: int = None):

        from source.config import AttentionFastKANConfig
        config: AttentionFastKANConfig = cast(AttentionFastKANConfig, self.get_config())

        self.num_heads: int = num_heads if num_heads is not None else config.num_heads

        super().__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            reduced_channels = in_channels,  # in_channels == out_channels
            in_seq_len = in_seq_len if in_seq_len is not None else config.in_seq_len,
            reduced_seq_len = reduced_seq_len if reduced_seq_len is not None else config.reduced_seq_len,
            hidden_layers = hidden_layers if hidden_layers is not None else config.hidden_layers,
            reduction_layer_class = AttentionLayer,
            reduction_layer_kwargs = dict(
                num_heads = self.num_heads
            ),
            grid_diff = grid_diff if grid_diff is not None else config.grid_diff,
            num_grids = num_grids if num_grids is not None else config.num_grids
            )


    @classmethod
    @override
    def config_type(cls) -> ConfigType: return ConfigType.AttentionFastKAN



# noinspection DuplicatedCode
class AttentionMLP(AbstractReducedFFN):
    """A model combining an `AttentionLayer` for reduction and a standard `MLP` FFN."""

    def __init__(self,
            in_channels: int,
            in_seq_len: int = None,
            out_channels: Optional[int] = None,
            reduced_seq_len: int = None,
            hidden_layers: HiddenLayers = None,
            num_heads: int = None):

        from source.config import AttentionMLPConfig
        config: AttentionMLPConfig = cast(AttentionMLPConfig, self.get_config())

        self.num_heads: int = num_heads if num_heads is not None else config.num_heads

        super().__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            reduced_channels = in_channels,  # in_channels == out_channels
            in_seq_len = in_seq_len if in_seq_len is not None else config.in_seq_len,
            reduced_seq_len = reduced_seq_len if reduced_seq_len is not None else config.reduced_seq_len,
            hidden_layers = hidden_layers if hidden_layers is not None else config.hidden_layers,
            reduction_layer_class = AttentionLayer,
            reduction_layer_kwargs = dict(
                num_heads = self.num_heads
            ),
            ffn_layer_class = MLP,
        )


    @classmethod
    @override
    def config_type(cls) -> ConfigType: return ConfigType.AttentionMLP



# noinspection DuplicatedCode
class PositionalFastKAN(AbstractReducedFastKAN):  # Adding functionality to FastKAN
    """A model combining a `PositionalWeightedConvLayer` and a `FastKAN` FFN."""

    def __init__(self,
            in_channels: int,
            in_seq_len: int = None,
            out_channels: Optional[int] = None,
            reduced_seq_len: int = None,
            reduced_channels: int = None,
            hidden_layers: HiddenLayers = None,
            kernel_size: int = None,
            weights_scalar: float = None,
            grid_diff: int = None,
            num_grids: int = None):

        from source.config import PositionalFastKANConfig
        config: PositionalFastKANConfig = cast(PositionalFastKANConfig, self.get_config())

        self.kernel_size: int = kernel_size if kernel_size is not None else config.kernel_size
        self.weights_scalar: float = weights_scalar if weights_scalar is not None else config.kernel_size

        super().__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            in_seq_len = in_seq_len if in_seq_len is not None else config.in_seq_len,
            reduced_seq_len = reduced_seq_len if reduced_seq_len is not None else config.reduced_seq_len,
            reduced_channels = reduced_channels if reduced_channels is not None else config.reduced_channels,
            hidden_layers = hidden_layers if hidden_layers is not None else config.hidden_layers,
            reduction_layer_class = PositionalWeightedConvLayer,
            reduction_layer_kwargs = dict(
                kernel_size = self.kernel_size,
                weights_scalar = self.weights_scalar
            ),
            grid_diff = grid_diff if grid_diff is not None else config.grid_diff,
            num_grids = num_grids if num_grids is not None else config.num_grids
        )


    @classmethod
    @override
    def config_type(cls) -> ConfigType: return ConfigType.PositionalFastKAN


    # @override
    # def add_figures_collection(self, figures) -> List:
    #     """Adds a figure visualizing the learned positional weights to the collection."""
    #     return [figures.positional_weights(model=self)]



# noinspection DuplicatedCode
class PositionalMLP(AbstractReducedFFN):
    """A model combining a `PositionalWeightedConvLayer` for reduction and a standard `MLP` FFN."""

    def __init__(self,
            in_channels: int,
            in_seq_len: int = None,
            out_channels: Optional[int] = None,
            reduced_seq_len: int = None,
            reduced_channels: int = None,
            hidden_layers: HiddenLayers = None,
            kernel_size: int = None,
            weights_scalar: float = None):

        from source.config import PositionalMLPConfig
        config: PositionalMLPConfig = cast(PositionalMLPConfig, self.get_config())

        self.kernel_size: int = kernel_size if kernel_size is not None else config.kernel_size
        self.weights_scalar: float = weights_scalar if weights_scalar is not None else config.kernel_size

        super().__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            in_seq_len = in_seq_len if in_seq_len is not None else config.in_seq_len,
            reduced_seq_len = reduced_seq_len if reduced_seq_len is not None else config.reduced_seq_len,
            reduced_channels = reduced_channels if reduced_channels is not None else config.reduced_channels,
            hidden_layers = hidden_layers if hidden_layers is not None else config.hidden_layers,
            reduction_layer_class = PositionalWeightedConvLayer,
            reduction_layer_kwargs = dict(
                kernel_size = self.kernel_size,
                weights_scalar = self.weights_scalar
            ),
            ffn_layer_class = MLP,
        )



    @classmethod
    @override
    def config_type(cls) -> ConfigType: return ConfigType.PositionalMLP



# noinspection DuplicatedCode
class UNetFastKAN(AbstractReducedFastKAN):
    """A model combining a `UNetReductionLayer` and a `FastKAN` FFN."""

    def __init__(self,
            in_channels: int,
            in_seq_len: int = None,
            out_channels: Optional[int] = None,
            reduced_seq_len: int = None,
            reduced_channels: int = None,
            hidden_layers: HiddenLayers = None,
            kernel_size: int = None,
            grid_diff: int = None,
            num_grids: int = None):

        from source.config import UNetFastKANConfig
        config: UNetFastKANConfig = cast(UNetFastKANConfig, self.get_config())

        self.kernel_size: int = kernel_size if kernel_size is not None else config.kernel_size

        super().__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            in_seq_len = in_seq_len if in_seq_len is not None else config.in_seq_len,
            reduced_seq_len = reduced_seq_len if reduced_seq_len is not None else config.reduced_seq_len,
            reduced_channels = reduced_channels if reduced_channels is not None else config.reduced_channels,
            hidden_layers = hidden_layers if hidden_layers is not None else config.hidden_layers,
            reduction_layer_class = UNetReductionLayer,
            reduction_layer_kwargs = dict(
                kernel_size = self.kernel_size
            ),
            grid_diff = grid_diff if grid_diff is not None else config.grid_diff,
            num_grids = num_grids if num_grids is not None else config.num_grids
        )


    @classmethod
    @override
    def config_type(cls) -> ConfigType: return ConfigType.UNetFastKAN



# noinspection DuplicatedCode
class UNetMLP(AbstractReducedFFN):
    """A model combining a `UNetReductionLayer` for reduction and a standard `MLP` FFN."""

    def __init__(self,
            in_channels: int,
            in_seq_len: int = None,
            out_channels: Optional[int] = None,
            reduced_seq_len: int = None,
            reduced_channels: int = None,
            hidden_layers: HiddenLayers = None,
            kernel_size: int = None):

        from source.config import UNetMLPConfig
        config: UNetMLPConfig = cast(UNetMLPConfig, self.get_config())
        
        self.kernel_size: int = kernel_size if kernel_size is not None else config.kernel_size
        
        super().__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            in_seq_len = in_seq_len if in_seq_len is not None else config.in_seq_len,
            reduced_seq_len = reduced_seq_len if reduced_seq_len is not None else config.reduced_seq_len,
            reduced_channels = reduced_channels if reduced_channels is not None else config.reduced_channels,
            hidden_layers = hidden_layers if hidden_layers is not None else config.hidden_layers,
            reduction_layer_class = UNetReductionLayer,
            reduction_layer_kwargs = dict(
                kernel_size = self.kernel_size
            ),
            ffn_layer_class = MLP,
        )


    @classmethod
    @override
    def config_type(cls) -> ConfigType: return ConfigType.UNetMLP