"""
This module defines the `SaveState` class,
a container for preserving the state of a model's training process.

The class is designed to track training and validation metrics across epochs,
such as loss and accuracy, f1_score, ...
It supports a mechanism for temporary storage of recent epoch results,
which can then be committed to the permanent history.
This is particularly useful for implementing early stopping,
where training might continue for several epochs past the best-performing one before stopping,
but only the state up to the best epoch needs to be saved.
"""

import copy

from typing_extensions import List, Tuple, Optional, Dict

from source.metrics.metrics import PerformanceMetric
from source.models.abstract import AbstractModel

from source.config import TrainingConfig, ConfigType, parse_config
training_config: TrainingConfig = parse_config(ConfigType.Training)



class SaveState:
    """
    Holds relevant information to continue or analyze a training process.

    This class stores the history of training loss, validation loss,
    and a standardized set of performance metrics (defined by PerformanceMetric enum) for each epoch.
    It uses temporary lists to accumulate results during a training run,
    which can then be selectively committed to the main history.
    """

    def __init__(self):
        self.training_loss_history: List[float] = []
        self.validation_loss_history: List[float] = []
        self.performance_history: Dict[PerformanceMetric, List[Optional[float]]] = {
            metric: [] for metric in PerformanceMetric
        }

        self.__training_loss_history_temp: List[float] = []
        self.__validation_loss_history_temp: List[float] = []
        self.__performance_history_temp: Dict[PerformanceMetric, List[Optional[float]]] = {
            metric: [] for metric in PerformanceMetric
        }


    def increment(self, training_loss: float, validation_loss: float, performance_values: Dict[PerformanceMetric, float]) -> None:
        """
        Appends the metrics from a single completed epoch to the temporary history.

        Args:
            training_loss: The training loss for the epoch.
            validation_loss: The validation loss for the epoch.
            performance_values: A dictionary mapping PerformanceMetric members to their scores.
                                If a metric is not provided, 'None' will be recorded for that epoch.
        """
        unknown_keys = set(performance_values.keys()) - set(PerformanceMetric)
        if unknown_keys:
            raise ValueError(f"Unknown performance metric keys provided: {unknown_keys}")

        self.__training_loss_history_temp.append(training_loss)
        self.__validation_loss_history_temp.append(validation_loss)

        for metric in PerformanceMetric:
            value = performance_values.get(metric)
            self.__performance_history_temp[metric].append(value)


    def current_epoch(self) -> int:
        """
        Calculates the last epoch number based on the length of the permanent history.

        Returns:
            The index of the last completed and saved epoch.
        """
        len_train_loss = len(self.training_loss_history)
        assert len(self.validation_loss_history) == len_train_loss, "Mismatch in history lengths."
        for metric, history in self.performance_history.items():
            assert len(history) == len_train_loss, f"Mismatch in length for performance metric '{metric.name}'."
        return len_train_loss - 1


    def update(self, best_epoch: Optional[int]) -> None:
        """
        Updates the permanent history from the temporary history.

        Args:
            best_epoch: The epoch number to update the history to (inclusive).
                        If None, all data from the temporary history is transferred.
        """
        if best_epoch is None:
            epoch_diff = None
        else:
            current_epoch = self.current_epoch()
            assert best_epoch >= current_epoch, f"best_epoch {best_epoch} must be greater than current epoch {self.current_epoch()}!"
            epoch_diff = best_epoch - current_epoch + 1

        self.training_loss_history.extend(self.__training_loss_history_temp[:epoch_diff])
        self.validation_loss_history.extend(self.__validation_loss_history_temp[:epoch_diff])
        for metric in PerformanceMetric:
            self.performance_history[metric].extend(self.__performance_history_temp[metric][:epoch_diff])

        if epoch_diff is None:
            self.clean()
        else:
            self.__training_loss_history_temp = self.__training_loss_history_temp[epoch_diff:]
            self.__validation_loss_history_temp = self.__validation_loss_history_temp[epoch_diff:]
            for metric in PerformanceMetric:
                self.__performance_history_temp[metric] = self.__performance_history_temp[metric][epoch_diff:]


    def clean(self) -> None:
        """Clears all temporary history lists."""
        self.__training_loss_history_temp = []
        self.__validation_loss_history_temp = []
        self.__performance_history_temp = {metric: [] for metric in PerformanceMetric}


    def clone(self) -> 'SaveState':
        """Creates a deep copy of the current SaveState object."""
        return copy.deepcopy(self)


    @staticmethod
    def load(model_file_path: str) -> Tuple[AbstractModel, 'SaveState']:
        """Loads a model and its associated SaveState from a file."""
        model, kwargs = AbstractModel.load(file_path=model_file_path)
        save_state: SaveState = kwargs.get("save_state")
        assert save_state is not None, "save_state not found in kwargs!"
        return model, save_state