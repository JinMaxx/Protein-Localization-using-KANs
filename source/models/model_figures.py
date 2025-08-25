"""
Provides a framework for creating visualizations of internal model states,
focusing on interpretability and explainability.

This module is designed to be extensible, allowing for different figures to be
created for various model architectures.
"""

from abc import ABC

import plotly.graph_objects as go
from plotly.graph_objs import Figure

from typing_extensions import Optional, cast, override

from source.models.abstract import AbstractModel
from source.models.reduced_ffn import PositionalFastKAN
from source.abstract_figures import AbstractFiguresCollection, _AbstractFigure



class ModelFiguresCollection(AbstractFiguresCollection):
    """Manages a collection of figures for visualizing model-specific properties."""

    def __init__(self, save_dir: Optional[str] = None):
        super().__init__(save_dir=save_dir)


    def positional_weights(self, model: PositionalFastKAN) -> 'PositionalWeightsFigure':
        """
        Creates and adds a figure for visualizing the positional weights of a PositionalFastKAN model.

        Args:
            model (PositionalFastKAN): The model instance from which to extract weights.

        Returns:
            PositionalWeightsFigure: The created figure instance.
        """
        return cast(PositionalWeightsFigure, self._add(PositionalWeightsFigure(model=model, collection=self)))


    @override
    def update(self, clear: bool = True, **kwargs) -> None:
        """
        Updates all managed figures in the collection.

        Args:
            clear (bool): If True, clears the output before displaying updates.
        """
        super().update(clear=clear, **kwargs)



class _AbstractModelFigure(_AbstractFigure, ABC):
    """Abstract base class for figures that visualize internal aspects of a model."""

    def __init__(self, model: AbstractModel, collection: AbstractFiguresCollection, figure: Optional[Figure] = None):
        """
        Initializes the model-based figure.

        Args:
            model (AbstractModel): The model to be visualized. A reference is stored to access its internal state.
            collection (AbstractFiguresCollection): The collection this figure belongs to.
            figure (Optional[Figure]): An optional pre-existing Plotly Figure object.
        """
        super().__init__(collection=collection, figure=figure, identifier=model.id())
        self._model: AbstractModel = model  # ModelFigures need the reference as it is using internal variables.



# For one-hot encoding I could maybe color it according to amino-acid (most likely chemistry similarity)
# This does not look very good for high-dimensional data.
# It was designed with the goal in mind to visualize the reduction and with KANs to explain certain features in the data.
class PositionalWeightsFigure(_AbstractModelFigure):
    """
    Creates a bar chart to visualize aggregated positional weights from a PositionalFastKAN model.

    The height of each bar represents the cumulative "importance" of a residue position,
    while its color indicates which input feature dimension had the highest weight,
    providing insight into both *where* the model focuses and on *what* features.
    """

    def __init__(self, model: PositionalFastKAN, collection: AbstractFiguresCollection):
        """
        Initializes the positional weights figure.

        Args:
            model (PositionalFastKAN): The model instance to visualize.
            collection (AbstractFiguresCollection): The collection this figure belongs to.
        """
        super().__init__(model=model, collection=collection, figure=go.Figure())

        self._fig.update_layout(
            title = f"Positional Weights Visualization ({model.name()})",
            xaxis_title = "Residue Position Index",
            yaxis_title = "Aggregated Weight (Sum over Encoding Dim.)",  # Or "Average Weight" if using mean
        )


    @override
    def update(self, **kwargs) -> None:
        """
        Fetches the latest positional weights from the model and updates the bar chart.

        This method recalculates the aggregated weights and determines the color for each bar
        based on the index of the maximum weight in the encoding dimension.
        """
        self._fig.data = []

        # Access positional weights and detach/convert to numpy
        weights = self._model.reduction_layer.positional_weights.detach().cpu().numpy().squeeze()  # [in_seq_len, in_channels]

        # Aggregate across encoding dimensions (Maybe use mean?)
        aggregated_weights = weights.sum(axis=1)  # Shape: [in_seq_len]

        # Get indices of maximal weights to know which contributed the most.
        max_weight_indices = weights.argmax(axis=1)  # Shape: [in_seq_len], max index per position

        # Normalize the max_weight_indices between [0, 1] for gradient mapping
        normalized_indices = max_weight_indices / (self._model.in_channels - 1)  # Range: [0, 1024] -> [0, 1]

        self._fig.add_trace(go.Bar(
            x = list(range(0, len(aggregated_weights))),  # Residue positions (1-indexed)
            y = aggregated_weights,
            name = "Residue Importance",
            marker = dict( # Apply the gradient colors
                color = normalized_indices,
                colorscale = [
                    [0.0, "blue"],
                    [0.5, "purple"],
                    [1.0, "red"]
                ],
                colorbar = dict(
                    title = "Max Weight Index (Dim.)",
                    tickvals = [0.01, 0.5, 0.99],
                    ticktext = ["0", str(self._model.in_channels // 2), str(self._model.in_channels)],
                )
            )
        ))