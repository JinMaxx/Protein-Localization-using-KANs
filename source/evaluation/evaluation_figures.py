"""
This module provides a structured framework for generating, managing, and displaying evaluation figures.

Key components include:
- `EvaluationFiguresCollection`: A central manager that holds and updates multiple
  evaluation figures. It acts as a factory for creating specific plot types.
- `AccuracyComparisonFigure`: A concrete implementation that generates a bar chart
  to compare model accuracies, featuring error bars for confidence intervals and
  highlighting for new results.
- `_AbstractEvaluationFigure`: An abstract base class that defines the common
  interface for all evaluation figures, ensuring consistent integration with the
  collection.
"""

import plotly.graph_objects as go

from abc import ABC
from typing_extensions import Optional, List, cast, override, Tuple

from source.metrics.metrics import Metrics
from source.metrics.metrics_sampled import MetricsSampled
from source.abstract_figures import AbstractFiguresCollection, _AbstractFigure



class EvaluationFiguresCollection(AbstractFiguresCollection):
    """Manages a collection of figures for evaluating and comparing model performance."""


    def __init__(self, save_dir: Optional[str] = None):
        super().__init__(save_dir=save_dir)


    def accuracy_comparison(self,
            identifier: Optional[str] = None,
            model_names: Optional[List[str]] = None,
            accuracies: Optional[List[float]] = None,
            errors: Optional[List[float]] = None) -> 'AccuracyComparisonFigure':
        """
        Creates and adds a bar chart figure for comparing model accuracies.

        Args:
            identifier (Optional[str]): An identifier for the figure.
            model_names (Optional[List[str]]): Pre-existing model names to plot.
            accuracies (Optional[List[float]]): Pre-existing accuracy values.
            errors (Optional[List[float]]): Pre-existing error margins for the accuracies.

        Returns:
            AccuracyComparisonFigure: The created figure instance.
        """
        return AccuracyComparisonFigure(
            identifier = identifier,
            model_names = model_names,
            accuracies = accuracies,
            errors = errors,
            collection = self
        )


    @override
    def update(self,
            model_name: str,
            metrics: Metrics,
            metrics_sampled: Optional[MetricsSampled] = None,
            clear: bool = True,
            **kwargs
    ) -> None:
        """
        Updates all managed figures with new evaluation data.

        Args:
            model_name (str): The name of the model being evaluated.
            metrics (Metrics): The metrics object for the current model.
            metrics_sampled (Optional[MetricsSampled]): Metrics from sampled data, used for error bars.
            clear (bool): If True, clears the output before displaying updates.
        """
        super().update(
            model_name = model_name,
            metrics = metrics,
            metrics_sampled = metrics_sampled,
            clear = clear,
            **kwargs
        )



class _AbstractEvaluationFigure(_AbstractFigure, ABC):
    """Abstract base class for figures that are used for model evaluation."""

    def __init__(self, collection: AbstractFiguresCollection, identifier: Optional[str] = None):
        super().__init__(collection=collection, figure=go.Figure(), identifier=identifier)



class AccuracyComparisonFigure(_AbstractEvaluationFigure):
    """
    Creates a bar chart to compare the accuracy of multiple models.

    This figure displays each model's accuracy as a bar, making it easy to see
    performance differences. It can also show confidence intervals as error bars if
    bootstrapped metrics are provided. Newly added models are highlighted in a
    different color to draw attention to the latest results.
    """

    def __init__(self,
            collection: EvaluationFiguresCollection,
            identifier: Optional[str] = None,
            model_names: Optional[List[str]] = None,
            accuracies: Optional[List[float]] = None,
            errors: Optional[List[float]] = None):
        """
        Initializes the accuracy comparison figure.

        The figure can be pre-populated with existing model results.

        Args:
            collection (EvaluationFiguresCollection): The collection this figure belongs to.
            identifier (Optional[str]): An identifier for the figure.
            model_names (Optional[List[str]]): A list of names for pre-existing models.
            accuracies (Optional[List[float]]): A list of accuracies corresponding to the models.
            errors (Optional[List[float]]): A list of error margins for the accuracies.
        """
        super().__init__(collection=collection, identifier=identifier)

        self._model_names: List[str] = model_names or []
        self._accuracies: List[float] = accuracies or []
        self._errors: List[float] = errors or [float('nan')] * len(self._accuracies)
        self._colors: List[str] = ['grey'] * len(self._accuracies)

        assert len(self._model_names) == len(self._accuracies) == len(self._errors), "model_names, accuracies, and errors must be the same length"
        assert len(self._model_names) == len(set(self._model_names)), "model_names must be unique"
        self._num_pre_init = len(self._model_names)

        self._fig.update_layout(
            title = "Model Accuracy Comparison (Q10)",
            xaxis = dict(
                title = "Model",
                type = "category",  # for names on x-axis
                tickangle = -45
            ),
            yaxis = dict(
                title = "Accuracy",
                range = [0, 1],
                dtick = 0.1
            ),
            bargap = 0.35,
            showlegend = False
        )


    @override
    def update(self,
            model_name: str,
            metrics: Metrics,
            metrics_sampled: Optional[MetricsSampled] = None,
            **kwargs) -> None:
        """
        Adds a new model's accuracy to the bar chart or updates an existing one.

        This method appends a new bar for the specified model.
        If `metrics_sampled` is provided, it calculates and displays a confidence interval as an error bar.

        Args:
            model_name (str): The name of the model to add or update.
            metrics (Metrics): The standard metrics object providing the accuracy score.
            metrics_sampled (Optional[MetricsSampled]): The sampled metrics object used
                to calculate confidence intervals for the error bars.
        """
        self._fig.data = []

        self._model_names.append(model_name)
        assert len(self._model_names) == len(set(self._model_names)), "model_names must be unique"

        accuracy: float = metrics.accuracy()
        self._accuracies.append(accuracy)

        self._colors.append('orange')

        if metrics_sampled is not None:
            confidence_interval: Tuple[float, float] = metrics_sampled.accuracies_confidence_interval()
            error = abs(confidence_interval[1] - confidence_interval[0]) / 2  # symmetric error
            self._errors.append(error)  # (np.sqrt(accuracy * (1 - accuracy) / total) if total > 0 else 0.0) # standard error (multiply by 1.96 for 95% confidence interval)
        else:
            self._errors.append(float('nan'))  # hiding error bars

        # # keeping preset data in original order
        # pre_init_names = self._model_names[:self._num_pre_init]
        # pre_init_accuracies = self._accuracies[:self._num_pre_init]
        # pre_init_errors = self._errors[:self._num_pre_init]
        # pre_init_colors = self._colors[:self._num_pre_init]
        #
        # # separating new data from preset data
        # new_names = self._model_names[self._num_pre_init:]
        # new_accuracies = self._accuracies[self._num_pre_init:]
        # new_errors = self._errors[self._num_pre_init:]
        # new_colors = self._colors[self._num_pre_init:]
        #
        # # sort new data by accuracy
        # if new_names:
        #     zipped_new_data = sorted(
        #         zip(new_accuracies, new_names, new_errors, new_colors),
        #         key=lambda item: item[0]  # Sort by accuracy (first element)
        #     )
        #     sorted_new_accuracies, sorted_new_names, sorted_new_errors, sorted_new_colors = zip(*zipped_new_data)
        # else: # no new models
        #     sorted_new_accuracies, sorted_new_names, sorted_new_errors, sorted_new_colors = [], [], [], []
        #
        # # combine data
        # final_names = pre_init_names + list(sorted_new_names)
        # final_accuracies = pre_init_accuracies + list(sorted_new_accuracies)
        # final_errors = pre_init_errors + list(sorted_new_errors)
        # final_colors = pre_init_colors + list(sorted_new_colors)

        bar = go.Bar(
            x = self._model_names,
            y = self._accuracies,
            error_y = dict(
                type = 'data',
                array = self._errors,
                visible = True,
                color = 'black',
                thickness = 1.5
            ),
            text = [
                f"{acc * 100:.0f}%" if color == 'grey' else f"{acc * 100:.1f}%"
                for acc, color in zip(self._accuracies, self._colors)
            ],  # Display value over bar (lower precision for preset data)
            textposition = "auto",
            marker = dict(color = self._colors)
        )
        self._fig.add_trace(bar)