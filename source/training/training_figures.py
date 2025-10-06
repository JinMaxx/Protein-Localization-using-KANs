"""
This module provides a powerful and extensible framework for creating and managing visualizations
for machine learning experiments using Plotly.

It is designed to generate dynamic, interactive figures that track key metrics
throughout the model training and hyperparameter tuning processes.

Key features include:
- Collections (`TrainingFiguresCollection`, `EpochFiguresCollection`) to manage groups of figures.
- Specific figure classes for common plots like epoch-wise training/validation loss (`TrainingLoss`)
  and performance metrics (`TrainingPerformance`).
- A composite `EpochDualAxisFigure` to combine two plots on a shared x-axis, ideal for
  comparing metrics with different scales (e.g., loss vs. accuracy).
- Support for visualizing hyperparameter tuning trials.
- An object-oriented design that allows figures to be dynamically updated as new data
  from training epochs becomes available.
"""
import numpy as np
import plotly.graph_objects as go

from copy import deepcopy
from math import ceil, isclose
from abc import abstractmethod, ABC

from optuna import Study
from optuna.visualization import plot_optimization_history
from typing_extensions import Any, Optional, List, Dict, Set, Union, Tuple, cast, override, overload

from source.config import ParamInfo
from source.metrics.metrics import PerformanceMetric
from source.abstract_figures import _AbstractFigure, AbstractFiguresCollection



METRIC_COLORS: Dict[PerformanceMetric, str] = {
    PerformanceMetric.ACCURACY: "sienna",
    PerformanceMetric.SENSITIVITY: "cyan",
    PerformanceMetric.SPECIFICITY: "gold4",
    PerformanceMetric.PRECISION: "violet",
    PerformanceMetric.F1_SCORE: "green",
    PerformanceMetric.AUC_ROC: "pink",
    PerformanceMetric.AUC_PR: "purple",
    PerformanceMetric.PERFORMANCE: "darkslategray",
}



# -------------------------------- Hyper Param --------------------------------

class HyperParamFiguresCollection(AbstractFiguresCollection):
    """Manages figures related to hyperparameter optimization trials."""

    def __init__(self, save_dir: Optional[str] = None):
        super().__init__(save_dir=save_dir)


    def hyper_param_performance(self, identifier: Optional[str] = None) -> 'HyperParamPerformance':
        """Creates and adds a figure for plotting hyperparameter performance."""
        return HyperParamPerformance(identifier=identifier, collection=self)


    def hyper_param_parameter(self, study: Study, param_name: str, param_info: ParamInfo, identifier: Optional[str] = None) -> 'HyperParamParameter':
        """Creates and adds a figure for plotting hyperparameter optimization history."""
        return HyperParamParameter(
            study = study,
            param_name = param_name,
            param_info = param_info,
            identifier = identifier,
            collection = self
        )


    @override
    def update(self, performances: List[float], clear: bool = True, **kwargs) -> None:
        """
        Updates all managed figures with new hyperparameter performance data.

        Args:
            performances (List[float]): A list of performance scores from trials.
            clear (bool): If True, clears the output before displaying updates.
        """
        super().update(performances=performances, clear=clear, **kwargs)



class _AbstractHyperParamFigure(_AbstractFigure, ABC):
    """An abstract base class for figures related to hyperparameter optimization trials."""



class HyperParamPerformance(_AbstractHyperParamFigure):
    """A figure that plots the performance metric across successive hyperparameter trials."""

    def __init__(self, collection: HyperParamFiguresCollection, identifier: Optional[str] = None):
        super().__init__(collection=collection, figure=go.Figure(), identifier=identifier)

        # self.__performances: list[float] = []
        self._fig.add_trace(go.Scatter(x=[], y=[], mode="lines+markers", name="Performance", line=dict(color="blue")))

        self._fig.update_layout(
            title = f"Performance History",
            xaxis = dict(
                title = "Trial",
                range = [0, None],
                dtick = 2
        ),
            yaxis = dict(
                title = "Performance",
                range = [0, 1],
                dtick = 0.2
            ),
        )


    @override
    def update(self, performances: List[float], **kwargs) -> None:
        # self.__performances.append(performance)
        self._fig.data[0].x = list(range(len(performances)))
        self._fig.data[0].y = performances



class HyperParamParameter(_AbstractHyperParamFigure):
    """
    A figure that plots the optimization history for a single hyperparameter,
    showing its value across successive trials.
    """

    def __init__(self,
            study: Study,
            param_name: str,
            param_info: ParamInfo,
            collection: HyperParamFiguresCollection,
            identifier: Optional[str] = None):
        """
        Initializes the hyperparameter history plot.

        Args:
            study: The Optuna study object.
            param_name: The name of the hyperparameter to plot.
            param_info: Metadata about the parameter (e.g., range, type).
            collection: The collection this figure belongs to.
            identifier: An optional unique identifier.
        """
        self.__study: Study = study
        self.__param_name: str = param_name
        self.__y_axis: Dict[str, Any]

        existing_values = [
            trial.params[self.__param_name] for trial in self.__study.trials if self.__param_name in trial.params
        ]

        if param_info.type == 'linear':

            _range: Optional[Tuple[Union[int, float], Union[int, float]]] = None
            _step: Optional[Union[int, float]] = None

            if existing_values:
                _range = (
                    min(min(existing_values), param_info.range[0]),
                    max(max(existing_values), param_info.range[1])
                )
                if isclose(_range[0], param_info.range[0], abs_tol=1e-6) and isclose(_range[1], param_info.range[1], abs_tol=1e-6):
                    _step = param_info.step
            else:
                _range = param_info.range
                _step = param_info.step

            if _step is None: _step = 1 if _range is None else (_range[1] - _range[0]) / 10

            self.__y_axis = dict(
                range = _range,
                dtick = _step
            )

        elif param_info.type == 'category':
            self.__y_axis = dict(
                type = 'category',
                categoryorder = 'array',
                categoryarray = sorted(list(set(param_info.categories + existing_values)))
            )

        super().__init__(collection=collection, identifier=identifier, figure=self.__create_figure())


    def __create_figure(self) -> go.Figure:
        return plot_optimization_history(
            study = self.__study,
            target_name = self.__param_name.replace("_", " ").capitalize(),
            target = lambda trial: trial.params[self.__param_name]
        ).update_layout(
            xaxis = dict(
                range = [0, len(self.__study.trials)],
                dtick = 2
            ),
            yaxis = self.__y_axis
        )


    @override
    def update(self, **kwargs) -> None:
        """Re-generates the optimization history plot with the latest trial data."""
        self._fig = self.__create_figure()


    @override
    def name(self) -> str:
        return self.__param_name


# -------------------------------- Epoch Figures --------------------------------

class EpochFiguresCollection(AbstractFiguresCollection):
    """Manages a collection of figures that track metrics over training epochs."""

    def __init__(self, save_dir: Optional[str] = None):
        super().__init__(save_dir=save_dir)


    def epoch_dual_axis_figure(self,
            model_name: str,
            figure_left: '_AbstractEpochFigure',
            figure_right: '_AbstractEpochFigure',
            identifier: Optional[str] = None
    ) -> 'EpochDualAxisFigure':
        """
        Creates a figure that combines two epoch-based plots on a dual y-axis.

        Args:
            model_name (str): The name of the model being trained.
            figure_left (_AbstractEpochFigure): The figure for the left y-axis.
            figure_right (_AbstractEpochFigure): The figure for the right y-axis.
            identifier (Optional[str]): A unique identifier for the figure.

        Returns:
            EpochDualAxisFigure: The combined dual-axis figure instance.
        """
        return EpochDualAxisFigure(
            model_name = model_name,
            fig_left = figure_left,
            fig_right = figure_right,
            identifier = identifier,
            collection = self
        )

    
    def training_performance(self,
        model_name: str,
        epochs: int,
        performance_metrics: Optional[Set[PerformanceMetric]] = None,
        performance_data: Optional[Dict[PerformanceMetric, List[float | None]]] = None,
        identifier: Optional[str] = None
    ) -> 'TrainingPerformance':
        """
        Creates a figure for plotting model performance over epochs.

        Args:
            model_name (str): The name of the model being trained.
            epochs (int): The total number of training epochs.
            performance_metrics: A set of PerformanceMetric members to plot. Defaults to all metrics if None.
            performance_data (Optional[Dict[PerformanceMetric, List[float | None]]]): Initial dictionary of performance scores.
            identifier (Optional[str]): A unique identifier for the figure.

        Returns:
            TrainingPerformance: The created figure instance.
        """
        return TrainingPerformance(
            model_name = model_name,
            epochs = epochs,
            performance_metrics = performance_metrics,
            performance_data = performance_data,
            identifier = identifier,
            collection = self
        )


    def training_loss(self,
            model_name: str,
            epochs: int,
            train_losses: list[float] = None,
            val_losses: list[float] = None,
            identifier: Optional[str] = None
    ) -> 'TrainingLoss':
        """
        Creates a figure for plotting training and validation loss over epochs.

        Args:
            model_name (str): The name of the model being trained.
            epochs (int): The total number of training epochs.
            train_losses (list[float], optional): An initial list of training losses.
            val_losses (list[float], optional): An initial list of validation losses.
            identifier (Optional[str]): A unique identifier for the figure.

        Returns:
            TrainingLoss: The created figure instance.
        """
        return TrainingLoss(
            model_name = model_name,
            epochs = epochs,
            train_losses = train_losses,
            val_losses = val_losses,
            identifier = identifier,
            collection = self
        )


    @override
    def update(self,
            epoch: Optional[int] = None,
            epoch_cut: Optional[int] = None,
            epoch_mark: Optional[Tuple[int, Optional[str]]] = None,
            performance_values: Optional[Dict[PerformanceMetric, float]] = None,
            train_loss: Optional[float] = None,
            val_loss: Optional[float] = None,
            clear: bool = True,
            **kwargs) -> None:
        """
        Updates all managed epoch-based figures with new data.

        Args:
            epoch (Optional[int]): The current epoch number.
            epoch_cut (Optional[int]): An epoch at which data should be truncated.
            epoch_mark (Optional[Tuple[int, Optional[str]]]): An epoch that should be marked with a vertical line. 
                                                              If no color is given, the line will be green by default.
            performance_values (Optional[Dict[PerformanceMetric, float]]): The performance scores for the current epoch.
            train_loss (Optional[float]): The training loss for the current epoch.
            val_loss (Optional[float]): The validation loss for the current epoch.
            clear (bool): If True, clears the output before displaying updates.
        """
        super().update(
            epoch = epoch,
            epoch_cut = epoch_cut,
            epoch_mark = epoch_mark,
            performance_values = performance_values,
            train_loss = train_loss,
            val_loss = val_loss,
            clear = clear,
            **kwargs
        )



class _AbstractEpochFigure(_AbstractFigure, ABC):
    """An abstract base class for figures that are plotted against epochs."""

    def __init__(self, model_name: str, epochs: int, collection: EpochFiguresCollection, identifier: Optional[str] = None):
        super().__init__(
            collection = collection,
            figure = go.Figure(),
            identifier = identifier,
        )
        self.epochs: int = epochs
        self.model_name: str = model_name

        self._fig.update_layout(
            xaxis = dict(
                range = [0, self.epochs - 1],
                dtick = self.__calculate_dtick(self.epochs - 1)
            ),
        )


    @abstractmethod
    def _current_epoch(self) -> int:
        """
        Returns the most recent epoch number for which data is available.

        Returns:
            int: The current epoch number.
        """
        raise NotImplementedError


    @staticmethod
    def __calculate_dtick(epoch: int) -> int:
        dtick: int = ceil(epoch / 10)  # ~ 10 gridlines
        return 1 if dtick < 5 else (dtick // 5) * 5  # in steps of at least 5


    @override
    def save(self, save_dir: str, **kwargs: Any) -> Optional[str]:
        """
        Saves the figure, adjusting the x-axis to fit only the plotted data.

        Args:
            save_dir (str): The directory where the figure will be saved.

        Returns:
            Optional[str]: The file path of the saved image, or None if not saved.
        """
        self._fig.update_layout(  # focusing only on the plotted area
            xaxis = dict(
                range = [0, self._current_epoch()+1],
                dtick = self.__calculate_dtick(self._current_epoch()+1)
            ),
        )
        filepath = super().save(save_dir, **kwargs)
        self._fig.update_layout(  # return to normal
            xaxis = dict(
                range = [0, self.epochs - 1],
                dtick = self.__calculate_dtick(self.epochs - 1)
            ),
        )
        return filepath



class EpochDualAxisFigure(_AbstractEpochFigure):
    """Combines two epoch-based figures into one plot with a shared x-axis and dual y-axes."""
    # Left y-axis: fig_left, right y-axis: fig_right.

    def __init__(self,
            model_name: str,
            fig_left: _AbstractEpochFigure,
            fig_right: _AbstractEpochFigure,
            collection: EpochFiguresCollection, 
            identifier: Optional[str] = None
    ):
        self.fig_left = fig_left
        self.fig_right = fig_right

        # Both figures must span the same epoch range
        assert self.fig_left.epochs == self.fig_right.epochs
        epochs = fig_left.epochs
        super().__init__(
            model_name = model_name,
            epochs = epochs,
            collection = collection,
            identifier = identifier  # if identifier is not None else f"{self.fig_left.name()}__{self.fig_right.name()}"
        )

        # unregister from figures
        self.fig_left.remove()
        self.fig_right.remove()

        self.__refresh_traces()
        self.__update_layout()


    def __refresh_traces(self) -> None:
        # Remove all current traces
        self._fig.data = []

        # Copy left figure traces
        for trace in self.fig_left._fig.data:
            self._fig.add_trace(trace)

        # get max min value here
        # and calculate dtick

        # Copy right figure traces but assign to secondary (right) y-axis2
        for trace in self.fig_right._fig.data:
            self._fig.add_trace(trace.update(yaxis='y2'))
            # self._fig.add_trace(trace.update(yaxis='y2') if hasattr(trace, 'update') else trace)


    def __update_layout(self) -> None:
        yaxis2_layout = deepcopy(self.fig_right._fig.layout.yaxis)  # Using deepcopy for robustness
        yaxis2_layout.update(overlaying='y', side='right')  # Add dual-axis specific properties

        # Layout with dual y-axis
        self._fig.update_layout(
            title = f"{self.fig_left.__class__.__name__} & {self.fig_right.__class__.__name__} (epoch: {self._current_epoch()}) | {self.model_name}",
            yaxis = self.fig_left._fig.layout.yaxis,
            yaxis2 = yaxis2_layout,
            legend = dict(  # Move legend further to the right by setting x > 1
                x = 1.05,  # so it does not clash with the values of yaxis2
                xanchor = "left",  # anchor the left edge of legend
                y = 1,  # top aligned (default)
                yanchor = "top"
            )
        )


    @override
    def update(self, **kwargs) -> None:
        epoch_cut: Optional[int] = kwargs.get("epoch_cut")
        if epoch_cut is not None:
            self._fig.add_vline(
                x = epoch_cut,
                line = dict(
                    color = "red",
                    width = 1,
                    dash = "dash"
                ),
                annotation_text = str(epoch_cut),
                annotation_position = "top"
            )
        epoch_mark: Optional[Tuple[int, Optional[str]]] = kwargs.get("epoch_mark")
        if epoch_mark is not None:
            self._fig.add_vline(
                x = epoch_mark[0],
                line = dict(
                    color = epoch_mark[1] if epoch_mark[1] is not None else "green",
                    width = 1,
                    dash = "dash"
                ),
                annotation_text = str(epoch_mark[0]),
                annotation_position = "top"
            )
        self.fig_left.update(**kwargs)
        self.fig_right.update(**kwargs)
        self.__refresh_traces()
        self.__update_layout()


    @override
    def _current_epoch(self) -> int:
        assert self.fig_left._current_epoch() == self.fig_right._current_epoch()
        return self.fig_left._current_epoch()   # ^ Ensuring both sub-figures are synced.



class TrainingPerformance(_AbstractEpochFigure):
    """
    A figure for plotting multiple model performance metrics over training epochs,
    driven by the PerformanceMetric enum.
    """

    def __init__(self,
            model_name: str,
            epochs: int,
            collection: EpochFiguresCollection,
            performance_metrics: Optional[Set[PerformanceMetric]] = None,
            performance_data: Optional[Dict[PerformanceMetric, List[float | None]]] = None,
            identifier: Optional[str] = None):
        super().__init__(
            model_name = model_name,
            epochs = epochs,
            collection = collection,
            identifier = identifier
        )
        
        # Default to all metrics if none are specified
        self.performance_metrics = performance_metrics or set(PerformanceMetric)
        # Create a sorted list to ensure a consistent trace order
        self._ordered_metrics = sorted(list(self.performance_metrics), key=lambda m: m.name)

        self.__performance_data: Dict[PerformanceMetric, List[float]] = performance_data or {
            metric: [] for metric in self._ordered_metrics
        }

        for metric in self._ordered_metrics:
            self._fig.add_trace(go.Scatter(
                x = list(range(len(self.__performance_data.get(metric, [])))),
                y = self.__performance_data.get(metric, []),
                mode = "lines+markers",
                name = metric.value.replace("_", " ").title(),  # e.g., "f1_score" -> "F1 Score"
                line = dict(color=METRIC_COLORS.get(metric, "black")) # Default to black
            ))

        self._fig.update_layout(
            title = f"Training Performance {self.model_name} (Epoch: 0)",
            xaxis = dict(title="Epoch"),
            yaxis = dict(title="Metrics", range=[0.0, 1.0], dtick=0.1),
            # legend_title_text = 'Metrics'
        )


    @override
    def _current_epoch(self) -> int:
        if not self._ordered_metrics: return -1
        return len(self.__performance_data[self._ordered_metrics[0]]) - 1


    @overload
    def update(self, epoch: int, performance_values: Dict[PerformanceMetric, float], **kwargs) -> None: ...


    @overload
    def update(self, epoch_cut: int, **kwargs) -> None: ...
    
    
    @overload
    def update(self, epoch_mark: Tuple[int, Optional[str]], **kwargs) -> None: ...


    @override
    def update(self,
        epoch: Optional[int] = None,
        epoch_cut: Optional[int] = None,
        epoch_mark: Optional[Tuple[int, Optional[str]]] = None,
        performance_values: Optional[Dict[PerformanceMetric, float]] = None,
        **kwargs
    ) -> None:
        
        if epoch_cut is not None:
            self._fig.add_vline(
                x = epoch_cut,
                line = dict(
                    color = "red", 
                    width = 1, 
                    dash = "dash"
                ),
                annotation_text = str(epoch_cut),
                annotation_position = "top"
            )
            # Truncate performances up to (including) epoch_cut
            for i, metric in enumerate(self._ordered_metrics):
                self.__performance_data[metric] = self.__performance_data[metric][:epoch_cut + 1]
                self._fig.data[i].x = list(range(len(self.__performance_data[metric])))
                self._fig.data[i].y = self.__performance_data[metric]
        
        if epoch_mark is not None:
            self._fig.add_vline(
                x = epoch_mark[0],
                line = dict(
                    color = epoch_mark[1] if epoch_mark[1] is not None else "green", 
                    width = 1, 
                    dash = "dash"
                ),
                annotation_text = str(epoch_mark[0]),
                annotation_position = "top"
            )

        elif epoch is not None and performance_values is not None:
            # Update each plotted metric with new data for the given epoch
            for i, metric in enumerate(self._ordered_metrics):
                value = performance_values.get(metric)  # Returns None if metric not in dict
                self.__performance_data[metric].append(value)
                # Update the figure's data directly
                self._fig.data[i].x = list(range(len(self.__performance_data[metric])))
                self._fig.data[i].y = self.__performance_data[metric]

            self._fig.layout.update(title=f"Training Performance {self.model_name} (Epoch: {epoch})")

        else:
            raise ValueError("Either 'epoch_cut' or both 'epoch' and 'performance_values' must be provided.")



class TrainingLoss(_AbstractEpochFigure):
    """A figure for plotting training and validation loss over epochs."""

    def __init__(self, 
            model_name: str, 
            epochs: int, 
            collection: EpochFiguresCollection, 
            train_losses: list[float] = None, 
            val_losses: list[float] = None, 
            identifier: Optional[str] = None
    ):
        super().__init__(
            model_name = model_name,
            epochs = epochs,
            collection = collection,
            identifier = identifier
        )

        self.__train_losses: list[float] = train_losses or []
        self.__val_losses: list[float] = val_losses or []
        assert len(self.__train_losses) == len(self.__val_losses)

        self._fig.add_trace(go.Scatter(x=[], y=[], mode="lines+markers", name="Train Loss", line=dict(color="blue")))
        self._fig.add_trace(go.Scatter(x=[], y=[], mode="lines+markers", name="Val Loss", line=dict(color="orange")))
        self._fig.update_layout(
            title = f"Training & Validation Loss {self.model_name} (Epoch: 0)",
            xaxis = dict(title="Epoch"),
            yaxis = dict(
                title = "Loss",
                range = [0.0, None],
                tickformat = ".2f"
            ),
        )


    def __set_y_ticks(self):
        # Compute the y-axis ticks so that there are always 11
        losses = self.__train_losses + self.__val_losses
        if losses:
            y_min = 0.0
            y_max = max(losses)
        else:
            y_min = 0.0
            y_max = 1.0  # fallback default

        tick_vals = np.linspace(y_min, y_max, 11)
        self._fig.update_layout(
            yaxis = dict(
                tickvals = tick_vals.tolist(),
                range = [y_min, y_max],
                tickformat = ".3f" if y_max < 2.0 else ".2f" if y_max < 5.0 else ".1f" if y_max < 10.0 else ".0f"
            )
        )


    @override
    def _current_epoch(self) -> int:
        assert len(self.__train_losses) == len(self.__val_losses)
        return len(self.__train_losses) - 1


    @overload
    def update(self, epoch: int, train_loss: float, val_loss: float, **kwargs) -> None: ...

    @overload
    def update(self, epoch_cut: int, **kwargs) -> None: ...
    
    @overload
    def update(self, epoch_mark: Tuple[int, Optional[str]], **kwargs) -> None: ...

    @override
    def update(self,
            epoch: Optional[int] = None,
            epoch_cut: Optional[int] = None,
            epoch_mark: Optional[Tuple[int, Optional[str]]] = None,
            train_loss: Optional[float] = None,
            val_loss: Optional[float] = None,
            **kwargs) -> None:
        
        if epoch_cut is not None:
            self._fig.add_vline(
                x = epoch_cut,
                line = dict(
                    color = "red", 
                    width = 1,
                    dash = "dash"
                ),
                annotation_text = str(epoch_cut),
                annotation_position = "top"
            )

            # Truncate losses lists up to (including) epoch_cut
            self.__train_losses = self.__train_losses[:epoch_cut + 1]
            self.__val_losses = self.__val_losses[:epoch_cut + 1]

            # noinspection DuplicatedCode
            self._fig.data[0].x = list(range(len(self.__train_losses)))
            self._fig.data[0].y = self.__train_losses
            self._fig.data[1].x = list(range(len(self.__val_losses)))
            self._fig.data[1].y = self.__val_losses
            
        if epoch_mark is not None:
            self._fig.add_vline(
                x = epoch_mark[0],
                line = dict(
                    color = epoch_mark[1] if epoch_mark[1] is not None else "green", 
                    width = 1, 
                    dash = "dash"
                ),
                annotation_text = str(epoch_mark[0]),
                annotation_position = "top"
            )

        elif (epoch is not None and
              train_loss is not None and
              val_loss is not None):
            self.__train_losses.append(train_loss)
            self.__val_losses.append(val_loss)

            # Update plot with the latest training and validation losses
            # noinspection DuplicatedCode
            self._fig.data[0].x = list(range(len(self.__train_losses)))
            self._fig.data[0].y = self.__train_losses
            self._fig.data[1].x = list(range(len(self.__val_losses)))
            self._fig.data[1].y = self.__val_losses

            self._fig.layout.update(title=f"Training & Validation Loss {self.model_name} (Epoch: {epoch})")

            # for debug purposes
            # print(f"train_losses: {self.__train_losses}")
            # print(f"val_losses: {self.__val_losses}")

        else: raise ValueError("Either epoch_cut or (epoch, train_loss, val_loss) must be provided.")

        self.__set_y_ticks()



# -------------------------------- Training --------------------------------

# Multiple inheritance. Combining metrics figures and epoch figures
# class TrainingFiguresCollection(MetricsFiguresCollection, EpochFiguresCollection):
#     """A unified collection for managing all figures related to model training, including epoch-wise metrics and performance evaluation."""
#
#     def __init__(self, save_dir: Optional[str] = None):
#         super().__init__(save_dir=save_dir)
#
#
#     @override
#     def update(self,
#             model_name: Optional[str] = None,
#             metrics: Optional[Metrics] = None,
#             epoch: Optional[int] = None,
#             epoch_cut: Optional[int] = None,
#             performance_values: Optional[Dict[PerformanceMetric, float]] = None,
#             train_loss: Optional[float] = None,
#             val_loss: Optional[float] = None,
#             identifier: Optional[str] = None,
#             clear: bool = True,
#             **kwargs
#     ) -> None:
#         """
#         Updates all managed figures, dispatching arguments to the appropriate figure types.
#
#         Args:
#             model_name (Optional[str]): The name of the model.
#             metrics (Optional[Metrics]): Metrics object for evaluation plots.
#             epoch (Optional[int]): The current epoch number.
#             epoch_cut (Optional[int]): An epoch for truncating data.
#             performance_values (Optional[Dict[PerformanceMetric, float]]): The performance scores for the current epoch.
#             train_loss (Optional[float]): The current training loss.
#             val_loss (Optional[float]): The current validation loss.
#             identifier (Optional[str]): A unique identifier.
#             clear (bool): If True, clears the output before displaying updates.
#         """
#         AbstractFiguresCollection.update(self,
#             model_name = model_name,
#             metrics = metrics,
#             epoch = epoch,
#             epoch_cut = epoch_cut,
#             performance_values = performance_values,
#             train_loss = train_loss,
#             val_loss = val_loss,
#             identifier = identifier,
#             clear = clear,
#             **kwargs
#         )