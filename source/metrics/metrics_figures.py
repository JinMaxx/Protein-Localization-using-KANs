"""
This module provides a structured and extensible framework for generating, managing,
and displaying a variety of model performance metrics visualizations using Plotly.

At its core, the module is built around the `MetricsFiguresCollection`, a central
manager that acts as a factory for creating specific plot types.
This collection simplifies the visualization workflow by allowing a user to instantiate multiple
figures and update them all simultaneously with a single `Metrics` object.

Key components include:
- `MetricsFiguresCollection`: The main factory and manager for all metric plots.
- `_AbstractMetricsFigure`: An abstract base class defining the common interface
  for all figures that are generated from a `Metrics` object, ensuring
  consistent updates and integration.
- Concrete Figure Implementations:
  - `MetricsHeatmap`: For visualizing a confusion matrix.
  - `ROCCurve` and `PRCurve`: For plotting standard classification performance curves.
  - `ProbabilityCalibrationCurve`: For assessing model probability calibration.
  - `PredictionConfidenceViolin`: For analyzing the distribution of prediction scores.
"""

import numpy as np
import plotly.graph_objects as go

from abc import abstractmethod, ABC
from typing_extensions import Optional, cast, override

from source.metrics.metrics import Metrics
from source.data_scripts.read_data import Label
from source.abstract_figures import AbstractFiguresCollection, _AbstractFigure, MultiFigure, get_label_color



class MetricsFiguresCollection(AbstractFiguresCollection):
    """
    Manages a collection of figures for visualizing model performance metrics.

    This class provides methods to create and update various plots like confusion matrices,
    ROC curves, and calibration curves based on a `Metrics` object.
    """

    def __init__(self, save_dir: Optional[str] = None):
        super().__init__(save_dir=save_dir)


    def metrics_heatmap(self, identifier: Optional[str] = None) -> 'MetricsHeatmap':
        """Creates and adds a confusion matrix heatmap figure."""
        return cast(MetricsHeatmap, self._add(MetricsHeatmap(identifier=identifier, collection=self)))


    def probability_calibration_curve(self, identifier: Optional[str] = None) -> 'ProbabilityCalibrationCurve':
        """Creates and adds a probability calibration curve figure."""
        return cast(ProbabilityCalibrationCurve, self._add(ProbabilityCalibrationCurve(identifier=identifier, collection=self)))


    def prediction_confidence_violin(self, identifier: Optional[str] = None) -> 'PredictionConfidenceViolin':
        """Creates and adds a prediction confidence violin plot."""
        return cast(PredictionConfidenceViolin, self._add(PredictionConfidenceViolin(identifier=identifier, collection=self)))


    def roc_curve(self, identifier: Optional[str] = None) -> 'ROCCurve':
        """Creates and adds a Receiver Operating Characteristic (ROC) curve figure."""
        return cast(ROCCurve, self._add(ROCCurve(identifier=identifier, collection=self)))


    def pr_curve(self, identifier: Optional[str] = None) -> 'PRCurve':
        """Creates and adds a Precision-Recall (PR) curve figure."""
        return cast(PRCurve, self._add(PRCurve(identifier=identifier, collection=self)))


    def duo_curves(self, identifier: Optional[str] = None) -> MultiFigure:
        """Creates and adds a combined figure containing both ROC and PR curves."""
        roc_curve_figure = ROCCurve(collection=self)
        pr_curve_figure = PRCurve(collection=self)
        return cast(MultiFigure, self._add(MultiFigure(figures=[roc_curve_figure, pr_curve_figure], identifier=identifier, collection=self)))


    @override
    def update(self, model_name: str, metrics: Metrics, clear: bool = True, identifier: Optional[str] = None, **kwargs) -> None:
        """
        Updates all managed figures with new metrics data.

        Args:
            model_name (str): The name of the model being evaluated.
            metrics (Metrics): The metrics object containing the performance data.
            clear (bool): If True, clears the output before displaying updates.
            identifier (Optional[str]): Optional identifier for filenames.
        """
        super().update(model_name=model_name, metrics=metrics, clear=clear, identifier=identifier, **kwargs)



class _AbstractMetricsFigure(_AbstractFigure, ABC):
    """Abstract base class for figures that are generated from a `Metrics` object."""

    def __init__(self, collection: AbstractFiguresCollection, identifier: Optional[str] = None):
        super().__init__(collection=collection, figure=go.Figure(), identifier=identifier)



class MetricsHeatmap(_AbstractMetricsFigure):
    """
    Displays a normalized confusion matrix as a heatmap.

    This plot helps visualize the performance of a classification model.
    Each cell (y, x) shows the proportion of samples with true label 'y' that were predicted as label 'x'.
    The diagonal represents correct classifications. Hovering over a cell reveals the raw count of predictions.
    """

    def __init__(self, collection: MetricsFiguresCollection, identifier: Optional[str] = None):
        super().__init__(identifier=identifier, collection=collection)

        self._fig.update_layout(
            xaxis = dict(title="Predicted Label"),
            yaxis = dict(title="True Label")
        )


    @override
    def update(self,  model_name: str, metrics: Metrics, identifier: Optional[str] = None, **kwargs) -> None:
        if identifier is not None: self._identifier = identifier  # overwriting identifier (new model)
        self._fig.data = []

        class_names: list[str] =  [label.name for label in metrics.labels()]
        matrix = metrics.copy_of_confusion_matrix()
        matrix_sum_rows = matrix.sum(axis=1)[:, np.newaxis]  # true_label counts.

        # Normalize the confusion matrix (handle division by zero)
        matrix_normalized = np.divide(
            matrix,
            matrix_sum_rows,
            out = np.zeros_like(matrix, dtype=float),  # Default to 0
            where = matrix_sum_rows != 0  # Perform division only where the sum is non-zero
        )

        # Format for cleaner display (e.g., 2 decimals)
        text_values = np.array([["{:.2f}".format(val) for val in row] for row in matrix_normalized])
        text_colors = "red"

        heatmap = go.Heatmap(
            z = matrix_normalized,  # Confusion matrix (row-wise counts)
            x = class_names,  # Predicted labels
            y = class_names,  # True labels
            colorscale = "Viridis",
            text = text_values,  # Show relative frequency on cells
            texttemplate = "%{text}",
            # Use the dynamic color array here
            textfont = {"color": text_colors, "size": 12},
            customdata = matrix,
            hovertemplate = (  # show raw counts when hovering
                "True: %{y}<br>Predicted: %{x}<br>"
                "Norm: %{text}<br>"
                "Count: %{customdata}<extra></extra>"
            ),
            xgap = 1,
            ygap = 1,
        )
        self._fig.add_trace(heatmap)
        self._fig.update_layout(
            title = f"Confusion Matrix ({model_name})"
        )



class ProbabilityCalibrationCurve(_AbstractMetricsFigure):
    """
    Plots a curve to assess the calibration of a model's predicted probabilities.

    A perfectly calibrated model has a curve that lies along the diagonal line,
    where the predicted probability matches the actual fraction of positives.
    This plot bins predictions by confidence and plots the mean predicted probability
    against the observed fraction of positives for each bin,
    helping to identify if the model is overconfident or underconfident.
    """

    def __init__(self, collection: MetricsFiguresCollection, identifier: Optional[str] = None):
        super().__init__(identifier=identifier, collection=collection)

        self._fig.update_layout(
            xaxis = dict(
                title = "Mean Predicted Probability",
                range = [0, 1],
                dtick = 0.1
            ),
            yaxis = dict(
                title = "Fraction of Positives",
                range = [0, 1],
                dtick = 0.1
            )
        )


    @override
    def update(self, model_name: str, metrics: Metrics, identifier: Optional[str] = None, n_bins: int = 10, **kwargs) -> None:
        if identifier is not None: self._identifier = identifier
        self._fig.data = []

        predicted_probs = np.array(metrics.predicted_probabilities(), dtype=np.float32)  # shape (num_samples, num_classes)
        true_labels: np.ndarray = np.array(metrics.true_labels_values)

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1) # Bin statistics

        for label in metrics.labels():
            # Extract true binary labels for this class (one-vs.-all)
            true_binary: np.ndarray = true_labels.__eq__(label.value).astype(int)
            predicted_class_probs = predicted_probs[:, label.value]

            # Bin data for calibration curve
            bin_true_positives = np.zeros(n_bins)
            bin_predicted_means = np.zeros(n_bins)
            for i in range(n_bins):
                bin_mask = (bin_edges[i] <= predicted_class_probs) & (predicted_class_probs < bin_edges[i + 1])
                bin_true_positives[i] = true_binary[bin_mask].mean() if bin_mask.any() else np.nan
                bin_predicted_means[i] = predicted_class_probs[bin_mask].mean() if bin_mask.any() else np.nan

            # Add calibration scatter trace for this class
            self._fig.add_trace(
                go.Scatter(
                    x = bin_predicted_means,
                    y = bin_true_positives,
                    mode = "lines+markers",
                    name = label.name,
                    line = dict(
                        color = get_label_color(label),
                        shape = "linear"
                    ),
                    connectgaps = True
                )
            )

        # Add a perfect calibration reference line
        self._fig.add_trace(
            go.Scatter(
                x = [0, 1],
                y = [0, 1],
                mode = "lines",
                name = "Perfect Calibration",
                line = dict(color="black", dash="dash"),  # Dashed black line
                showlegend = True
            )
        )

        self._fig.update_layout(
            title = f"Probability Calibration Curve ({model_name})"
        )



class PredictionConfidenceViolin(_AbstractMetricsFigure):
    """
    Visualizes the distribution of prediction confidence scores for each class.

    This violin plot shows the density of confidence scores for all predictions assigned to a particular class.
    It helps to understand if a model is consistently confident or hesitant in its predictions for different classes.
    """

    def __init__(self, collection: MetricsFiguresCollection, identifier: Optional[str] = None):
        super().__init__(identifier=identifier, collection=collection)

        self._fig.update_layout(
            xaxis_title = "Class Labels",
            yaxis = dict(
                title = "Confidence Scores",
                range = [0, 1],
                dtick = 0.1
            )
        )


    @override
    def update(self, model_name: str, metrics: Metrics, identifier: Optional[str] = None, **kwargs) -> None:
        if identifier is not None: self._identifier = identifier
        self._fig.data = []

        confidence_distribution = metrics.prediction_confidence_distribution()

        for label, confidences in confidence_distribution.items():
            self._fig.add_trace( # Add trace for each label as a separate violin plot
                go.Violin(
                    y = confidences,
                    name = label.name,
                    box = dict(visible=False),  # Show box inside the violin
                    meanline = dict(visible=True),
                    line = dict(color=get_label_color(label))  # Set the color correctly here
                )
            )

        self._fig.update_layout(
            title = f"Prediction Confidence Distribution ({model_name})"
        )



class _AbstractCurve(_AbstractMetricsFigure, ABC):
    """
    Abstract base class for performance curves like ROC and PR.

    Handles the common logic for plotting data derived from classification thresholds.
    """

    def __init__(self, collection: MetricsFiguresCollection, identifier: Optional[str] = None):
        super().__init__(identifier=identifier, collection=collection)

        self._fig.update_layout(
            xaxis = dict(
                range = [0, 1],
                dtick = 0.2
            ),
            yaxis = dict(
                range = [0, 1],
                dtick = 0.2
            )
        )


    @staticmethod
    @abstractmethod
    def _curve_metrics_function(metrics: Metrics, label: Label) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Abstract method to retrieve the specific curve data (e.g., TPR, FPR) from the metrics."""
        raise NotImplementedError


    @override
    def update(self, model_name: str, metrics: Metrics, identifier: Optional[str] = None, **kwargs) -> None:
        if identifier is not None: self._identifier = identifier
        self._fig.data = []

        for label in metrics.labels():
            false_positive_rate, true_positive_rate, _ = self._curve_metrics_function(metrics, label)
            self._fig.add_trace(
                go.Scatter(
                    x = false_positive_rate,
                    y = true_positive_rate,
                    mode = "lines",
                    name = label.name,
                    legendgroup = label.name,  # Group all traces for this label
                    line = dict(
                        color = get_label_color(label),
                        shape = "hv" # "linear"
                    ),
                )
            )

        self._fig.update_layout(
            title = f"{self.__class__.__name__} ({model_name})"
        )



class ROCCurve(_AbstractCurve):
    """
    Plots the Receiver Operating Characteristic (ROC) curve.

    The ROC curve illustrates a classifier's diagnostic ability by plotting the
    True Positive Rate (TPR) against the False Positive Rate (FPR) at various thresholds.
    A curve that bows towards the top-left corner indicates a better-performing model.
    The diagonal line represents a random-chance classifier.
    """

    def __init__(self, collection: MetricsFiguresCollection, identifier: Optional[str] = None):
        super().__init__(identifier=identifier, collection=collection)

        self._fig.update_layout(
            xaxis_title = "Recall",
            yaxis_title = "Precision"
        )

        # Diagonal reference line (random baseline)
        self._fig.add_trace(
            go.Scatter(
                x = [0, 1],
                y = [0, 1],
                mode = "lines",
                name = "Baseline",
                line = dict(color="black", dash="dash"),
                showlegend = True
            )
        )


    @staticmethod
    @override
    def _curve_metrics_function(metrics: Metrics, label: Label) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return metrics.roc_curve(label)



class PRCurve(_AbstractCurve):
    """
    Plots the Precision-Recall (PR) curve.

    The PR curve shows the trade-off between Precision and Recall
    for different classification thresholds.
    It is particularly informative for imbalanced datasets,
    where a high Area Under the PR Curve indicates strong performance.
    """

    def __init__(self, collection: MetricsFiguresCollection, identifier: Optional[str] = None):
        super().__init__(identifier=identifier, collection=collection)

        self._fig.update_layout(
            xaxis_title = "False Positive Rate (FPR)",
            yaxis_title = "True Positive Rate (TPR)"
        )


    @staticmethod
    @override
    def _curve_metrics_function(metrics: Metrics, label: Label) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return metrics.pr_curve(label)