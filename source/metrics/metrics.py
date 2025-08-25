#!/usr/bin/python3
"""
This module provides a comprehensive framework for evaluating the performance of multi-class classification models.
It defines the `Metrics` class, which serves as the central tool for calculating, summarizing,
and exporting a wide range of evaluation metrics.

The script is designed to:
- Process predictions (true labels, logits, and loss) from a model.
- Compute a confusion matrix as the basis for most metrics.
- Calculate standard metrics like accuracy, precision, recall (sensitivity),
  specificity, F1-score, ROC AUC, and PR AUC.
- Support per-class, macro-averaged (unweighted), and weighted-averaged
  metric calculations.
- Generate human-readable summaries, detailed performance tables (as pandas
  DataFrames), and save results to YAML or TSV files.
- Define a configurable `performance` score, which is a weighted sum of
  several key metrics.

The main entry point allows for running evaluations on saved models directly.
"""

import os
import json
import yaml
import numpy as np
import pandas as pd

from enum import Enum
from torch import tensor
from torch.nn import functional
from collections.abc import Sequence
from cachetools.func import lru_cache
from typing_extensions import List, Dict, Any, cast, Optional

from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    f1_score as sk_f1_score,
    accuracy_score as sk_accuracy_score,
    recall_score as sk_sensitivity_score,
    precision_score as sk_precision_score,
    confusion_matrix as sk_confusion_matrix,
    average_precision_score as sk_average_pr_score,
    precision_recall_curve as sk_pr_curve,
    roc_auc_score as sk_roc_auc_score,
    roc_curve as sk_roc_curve,
)

from source.data_scripts.read_data import Label, all_labels, num_classes

from source.custom_types import (
    Loss_T,
    Prob_T,
    Label_T,
    Prediction_Generator_T
)

from source.config import MetricsConfig, ConfigType, parse_config
config: MetricsConfig = parse_config(ConfigType.Metrics)



class PerformanceMetric(Enum):
    ACCURACY = "accuracy"
    SENSITIVITY = "sensitivity"
    SPECIFICITY = "specificity"
    PRECISION = "precision"
    F1_SCORE = "f1_score"
    AUC_ROC = "auc_roc"
    AUC_PR = "auc_pr"
    PERFORMANCE = "performance"


class Metrics:
    """
    A comprehensive container for calculating and managing evaluation metrics
    for a multi-class classification task.

    Upon initialization, it consumes a generator of predictions and computes
    all necessary statistics, including a confusion matrix. It provides methods
    to query individual metrics (e.g., accuracy, F1-score) on a per-class,
    macro-average, or weighted-average basis. It can also generate summaries,
    data tables, and save results to files.
    """

    def __init__(self,
            true_labels: List[Label],
            predicted_labels: List[Label],
            predicted_probs: List[Prob_T],
            losses: List[Loss_T],
            class_weights: Optional[Dict[Label, float]] = None):
        """
        Initializes the Metrics object with its essential state data.

        Args:
            true_labels: A list of the ground truth Label enums.
            predicted_labels: A list of the predicted Label enums.
            predicted_probs: A list of the predicted probability tensors.
            losses: A list of the loss values for each sample.
            class_weights: An optional dictionary mapping each class label to a weight.
        """
        self.__true_labels: List[Label] = true_labels
        self.__predicted_labels: List[Label] = predicted_labels
        self.__predicted_probs: List[Prob_T] = predicted_probs
        self.__losses: List[Loss_T] = losses
        self.__class_weights: Optional[Dict[Label, float]] = class_weights


    @classmethod
    def from_generator(cls, predictions_generator: Prediction_Generator_T,
                       class_weights: Optional[Dict[Label, float]] = None) -> "Metrics":
        """
        Factory method to create and compute a Metrics object from a prediction generator.
        """
        losses: List[Loss_T] = []
        true_labels: List[Label] = []
        predicted_labels: List[Label] = []
        predicted_probs: List[Prob_T] = []

        for true_label_tensor, logit, loss in predictions_generator:
            true_label_tensor: Label_T
            logit: Label_T
            loss: Loss_T

            predicted_prob: Prob_T = functional.softmax(logit, dim=0)
            predicted_label: Label = Label(logit.argmax(dim=0).item())
            true_label: Label = Label(true_label_tensor.item())

            losses.append(loss)
            true_labels.append(true_label)
            predicted_labels.append(predicted_label)
            predicted_probs.append(predicted_prob)

        if not true_labels or not predicted_labels or not predicted_probs:
            raise ValueError("`predictions_generator` did not yield any data. Metrics cannot be calculated.")

        return cls(
            true_labels = true_labels,
            predicted_labels = predicted_labels,
            predicted_probs = predicted_probs,
            losses = losses,
            class_weights = class_weights
        )


    # Serialization
    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the essential state of the Metrics object to a JSON-compatible dictionary.
        """
        # Convert non-serializable objects (tensors, enums) to basic Python types
        return {
            "true_labels": [label.value for label in self.__true_labels],
            "predicted_labels": [label.value for label in self.__predicted_labels],
            "predicted_probs": [p.tolist() for p in self.__predicted_probs],
            "losses": self.__losses,
            "class_weights": {k.value: v for k, v in self.__class_weights.items()} if self.__class_weights else None,
        }


    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Metrics":
        """
        Factory method to reconstruct a Metrics object from a serialized dictionary.
        """
        return cls(
            true_labels = [Label(v) for v in data["true_labels"]],
            predicted_labels = [Label(v) for v in data["predicted_labels"]],
            predicted_probs = [tensor(p) for p in data["predicted_probs"]],
            losses = data["losses"],
            class_weights = {Label(k): v for k, v in data["class_weights"].items()} if data.get("class_weights") else None
        )


    # non-relevant but why not
    @staticmethod
    def labels() -> List[Label]:
        """Returns the list of all possible labels in the dataset."""
        return all_labels


    @property
    @lru_cache(maxsize=1)
    def true_labels_values(self) -> List[int]:
        """Returns the integer values of the true labels."""
        return [label.value for label in self.__true_labels]


    @property
    @lru_cache(maxsize=1)
    def predicted_labels_values(self) -> List[int]:
        """Returns the integer values of the predicted labels."""
        return [label.value for label in self.__predicted_labels]


    @property
    @lru_cache(maxsize=1)
    def __true_binarized(self) -> np.ndarray:
        """Computes and caches the binarized true labels."""
        return label_binarize(self.true_labels_values, classes=[label.value for label in self.labels()])



    @property
    @lru_cache(maxsize=1)
    def __matrix(self) -> np.ndarray:
        """Computes and caches the confusion matrix."""
        return sk_confusion_matrix(self.true_labels_values, self.predicted_labels_values, labels=list(range(num_classes)))


    @property
    @lru_cache(maxsize=1)
    def __class_weights_array(self) -> Optional[List[float]]:
        """Computes and caches the array of sample weights."""
        if not self.__class_weights: return None
        else: return [self.__class_weights[label] for label in self.__true_labels]


    @property
    @lru_cache(maxsize=1)
    def total(self) -> int:
        """Returns the total number of samples evaluated."""
        return cast(int, sum(self.__matrix.flatten()))
        # Alternative: len(self.__true_labels)


    @property
    @lru_cache(maxsize=1)
    def loss_per_sample(self) -> float:
        """Calculates the average loss across all samples."""
        assert len(self.__losses) == self.total
        return float(sum(self.__losses)) / len(self.__losses)


    def predicted_probabilities(self) -> List[Prob_T]:
        """Returns a copied list of predicted probability tensors, one for each sample."""
        return [prob.clone() for prob in self.__predicted_probs]


    def copy_of_confusion_matrix(self) -> np.ndarray:
        """Returns a copy of the internal confusion matrix."""
        return self.__matrix.copy()


    # Maybe incorporate this into performance calculation?
    # High values approach 0 quickly
    # def normalized_loss_exp(self, scale=10.0):
    #     # scale controls how rapidly the score drops with increasing loss
    #     loss = self.loss_per_sample()
    #     return np.exp(-loss / scale)


    # # High values approach decay slowly to 0
    # def normalized_loss_log(self):
    #     loss = self.loss_per_sample()
    #     return 1.0 / (1.0 + np.log1p(loss))  # log1p(l) = log(1+l)


    @lru_cache(maxsize=num_classes)
    def __actual_sum(self, label: Label) -> int:
        """Calculates the total number of actual instances for a given class."""
        return cast(int, sum(self.__matrix[label.value, :]))


    @lru_cache(maxsize=num_classes)
    def __predicted_sum(self, label: Label) -> int:
        """Calculates the total number of predicted instances for a given class."""
        return cast(int, sum(self.__matrix[:, label.value]))


    # if label is None: then the metric is calculated overall
    # if label is not None: metric is calculated per class

    @lru_cache(maxsize=num_classes+1)
    def true_positives(self, label: Label|None = None) -> int:
        """Calculates True Positives."""
        if label is None: # Sum of True Positives (diagonal sum)
            return cast(int, np.trace(self.__matrix))
        return cast(int, self.__matrix[label.value, label.value])


    @lru_cache(maxsize=num_classes+1)
    def false_positives(self, label: Label|None = None) -> int:
        """Calculates False Positives."""
        if label is None: # False Positives (sum of all columns minus the diagonals)
            return cast(int, sum(self.__matrix.sum(axis=0) - np.diag(self.__matrix)))
        return self.__predicted_sum(label) - self.true_positives(label)


    @lru_cache(maxsize=num_classes+1)
    def false_negatives(self, label: Label|None = None) -> int:
        """Calculates False Negatives."""
        if label is None: # Sum of False Negatives (sum of all rows minus the diagonals)
            return cast(int, sum(self.__matrix.sum(axis=1) - np.diag(self.__matrix)))
        return self.__actual_sum(label) - self.true_positives(label)


    @lru_cache(maxsize=num_classes)
    def true_negatives(self, label: Label) -> int:
        """
        Calculates True Negatives for a specific class.
        This metric is not meaningful for an overall (micro) average.
        """
        if label is None: return -1  # Micro-averaged across all classes would count double
        total = self.total
        actual_sum = self.__actual_sum(label)
        predicted_sum = self.__predicted_sum(label)
        true_positives = self.true_positives(label)
        return total - (actual_sum + predicted_sum - true_positives)


    # Scikit scores dont support per-label scores so I make my custom implementation.

    @lru_cache(maxsize=num_classes+1)
    def sensitivity(self, label: Label|None = None) -> float:
        """
        Calculates sensitivity (Recall or True Positive Rate).

        This metric measures the ability of the model to correctly identify
        positive instances for a class (TP / (TP + FN)).

        Behavior:
        - `label` = a class: Calculates sensitivity for that specific class.
        - `label` = None, with weights: Calculates the weighted average, where
          each class's sensitivity is weighted by its support (number of true
          instances).
        - `label` = None, no weights: Calculates the macro average (the unweighted
          mean of the sensitivities for all classes).
        """
        if label is None and self.__class_weights_array is not None:
            return sk_sensitivity_score(
                self.true_labels_values,
                self.predicted_labels_values,
                sample_weight=self.__class_weights_array,
                average="weighted",
                zero_division=0
            )
        elif label is None: # Macro average (unweighted)
            sensitivity_values = [self.sensitivity(label) for label in self.labels()]
            if len(sensitivity_values) == 0: return 0.0
            return sum(sensitivity_values) / len(sensitivity_values)
        else:
            true_positives = self.true_positives(label)
            false_negatives = self.false_negatives(label)
            if true_positives + false_negatives == 0: return 0.0
            return true_positives / (true_positives + false_negatives)


    @lru_cache(maxsize=num_classes+1)
    def specificity(self, label: Label|None = None) -> float:
        """
        Calculates specificity (True Negative Rate).

        This metric measures the ability of the model to correctly identify
        negative instances for a class (TN / (TN + FP)).

        Behavior:
        - `label` = a class: Calculates specificity for that specific class.
        - `label` = None, with weights: Calculates a weighted average of the
          per-class specificities.
        - `label` = None, no weights: Calculates the macro average (the unweighted
          mean of the specificities for all classes).
        """
        if label is None and self.__class_weights is not None:  # scikit learn does not have a specificity_score
            sum_of_weighted_specificities: float = sum(
                [weight * self.specificity(label) for label, weight in self.__class_weights.items()])
            sum_of_weights = sum(self.__class_weights.values())
            return sum_of_weighted_specificities / sum_of_weights
        elif label is None: # Macro average (unweighted)
            specificity_values = [self.specificity(label) for label in self.labels()]
            if len(specificity_values) == 0: return 0.0
            return sum(specificity_values) / len(specificity_values)
        else:
            true_negatives = self.true_negatives(label)
            false_positives = self.false_positives(label)
            if true_negatives + false_positives == 0: return 0.0
            return true_negatives / (true_negatives + false_positives)


    @lru_cache(maxsize=num_classes+1)
    def precision(self, label: Label|None = None) -> float:
        """
        Calculates precision (Positive Predictive Value).

        This metric measures the accuracy of positive predictions for a class
        (TP / (TP + FP)).

        Behavior:
        - `label` = a class: Calculates precision for that specific class.
        - `label` = None, with weights: Calculates the weighted average, where
          each class's precision is weighted by its support.
        - `label` = None, no weights: Calculates the macro average (the unweighted
          mean of the precisions for all classes).
        """
        if label is None and self.__class_weights_array is not None:
            return sk_precision_score(
                self.true_labels_values,
                self.predicted_labels_values,
                sample_weight=self.__class_weights_array,
                average="weighted",
                zero_division=0 # if certain true labels don't have predicted samples.
            )
        elif label is None: # Macro average (unweighted)
            precision_values = [self.precision(label) for label in self.labels()]
            if len(precision_values) == 0: return 0.0
            return sum(precision_values) / len(precision_values)
        else:
            true_positives = self.true_positives(label)
            false_positives = self.false_positives(label)
            if true_positives + false_positives == 0: return 0.0
            return true_positives / (true_positives + false_positives)


    @lru_cache(maxsize=num_classes+1)
    def accuracy(self, label: Label|None = None) -> float:
        """
        Calculates accuracy.

        Behavior:
        - `label` = None: Calculates the overall accuracy (also known as micro-
          average accuracy), which is the ratio of all correct predictions to the
          total number of samples. It can be weighted if class weights are provided.
        - `label` = a class: Calculates per-class accuracy, which is the ratio of
          (TP + TN) to the total samples for that specific class. This is less
          commonly used in multi-class settings than overall accuracy.
        """
        if label is None and self.__class_weights_array is not None:
            return sk_accuracy_score(
                self.true_labels_values,
                self.predicted_labels_values,
                sample_weight=self.__class_weights_array,
            )
        elif label is None:  # simple overall accuracy
            if self.total == 0: return 0.0
            return self.true_positives() / self.total
        else:
            # per-class accuracy is not very meaningful in multiclass prediction.
            true_positives = self.true_positives(label)
            true_negatives = self.true_negatives(label)
            false_positives = self.false_positives(label)
            false_negatives = self.false_negatives(label)
            total_samples = true_positives + true_negatives + false_positives + false_negatives
            if total_samples == 0: return 0.0
            return (true_positives + true_negatives) / total_samples


    @lru_cache(maxsize=num_classes+1)
    def f1_score(self, label: Label|None = None) -> float:
        """
        Calculates the F1-score (the harmonic mean of precision and sensitivity).

        The F1-score provides a single metric that balances both precision and
        recall (2 * (Precision * Recall) / (Precision + Recall)).

        Behavior:
        - `label` = a class: Calculates the F1-score for that specific class.
        - `label` = None, with weights: Calculates the weighted average F1-score.
        - `label` = None, no weights: Calculates the macro average F1-score.
        """
        if label is None and self.__class_weights_array is not None:
            return sk_f1_score(
                self.true_labels_values,
                self.predicted_labels_values,
                sample_weight=self.__class_weights_array,
                average="weighted",
                zero_division=0
            )
        elif label is None: # Macro average (unweighted)
            f1_score_values = [self.f1_score(label) for label in self.labels()]
            if len(f1_score_values) == 0: return 0.0
            return sum(f1_score_values) / len(f1_score_values)
        else:
            precision = self.precision(label)
            recall = self.sensitivity(label)
            if precision + recall == 0: return 0.0
            return 2 * (precision * recall) / (precision + recall)


    @lru_cache(maxsize=num_classes+1)
    def roc_auc(self, label: Label | None = None) -> float:
        """
        Calculates the Area Under the Receiver Operating Characteristic Curve (ROC AUC).

        This metric evaluates the model's ability to distinguish between classes.
        Returns `NaN` for a class if it has only one true label value present.

        Behavior:
        - `label` = a class: Calculates ROC AUC for that one-vs-rest case.
        - `label` = None, with weights: Calculates the weighted average ROC AUC.
        - `label` = None, no weights: Calculates the macro average ROC AUC.
        """
        if label is None and self.__class_weights is not None:  # Weighted calculation for all classes
            auc_weights = [(auc, weight) for label, weight in self.__class_weights.items()
                           if not np.isnan(auc := self.roc_auc(label))]
            sum_of_weighted_auc = sum(auc * weight for auc, weight in auc_weights)
            sum_of_weights = sum(weight for _, weight in auc_weights)
            if sum_of_weights == 0: return float("nan") # all classes invalid
            return sum_of_weighted_auc / sum_of_weights
        elif label is None:  # Macro averaging (default)
            auc_values = [auc for label in self.labels() if not np.isnan(auc := self.roc_auc(label))]
            if len(auc_values) == 0: return float("nan")
            return sum(auc_values) / len(auc_values)
        else:  # Single class ROC AUC
            # Not enough diversity (only one class present, or the label is effectively missing)
            if len(np.unique(self.__true_binarized[:, label.value])) < 2: return float("nan")
            else: return sk_roc_auc_score(
                y_true=self.__true_binarized[:, label.value],
                y_score=[p[label.value] for p in self.__predicted_probs]
            )


    @lru_cache(maxsize=num_classes+1)
    def pr_auc(self, label: Label | None = None) -> float:
        """
        Calculates the Area Under the Precision-Recall Curve (PR AUC).

        This metric is particularly useful for imbalanced datasets.
        Returns `NaN` for a class if it has only one true label value present.

        Behavior:
        - `label` = a class: Calculates PR AUC for that one-vs-rest case.
        - `label` = None, with weights: Calculates the weighted average PR AUC.
        - `label` = None, no weights: Calculates the macro average PR AUC.
        """
        if label is None and self.__class_weights is not None:  # Weighted calculation for all classes
            auc_weights = [(auc, weight) for label, weight in self.__class_weights.items()
                           if not np.isnan(auc := self.pr_auc(label))]
            sum_of_weighted_auc = sum(auc * weight for auc, weight in auc_weights)
            sum_of_weights = sum(weight for _, weight in auc_weights)
            if sum_of_weights == 0: return float("nan") # all classes invalid
            return sum_of_weighted_auc / sum_of_weights
        elif label is None:  # Macro averaging (default)
            auc_values = [auc for label in self.labels() if not np.isnan(auc := self.pr_auc(label))]
            if len(auc_values) == 0: return float("nan")
            return sum(auc_values) / len(auc_values)
        else:  # Single class PR AUC
            if len(np.unique(self.__true_binarized[:, label.value])) < 2: return float("nan")
            else: return sk_average_pr_score(
                y_true=self.__true_binarized[:, label.value],
                y_score=[p[label.value] for p in self.__predicted_probs]
            )


    def roc_curve(self, label: Label) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the ROC curve for a specific class.

        Returns:
            A tuple of (false_positive_rates, true_positive_rates, thresholds).
        """
        y_true = self.__true_binarized[:, label.value]
        y_scores = [probs[label.value] for probs in self.__predicted_probs]
        return sk_roc_curve(y_true, y_scores) # false_positive_rates, true_positive_rates, thresholds


    def pr_curve(self, label: Label) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the Precision-Recall curve for a specific class.

        Returns:
            A tuple of (precision, recall, thresholds).
        """
        y_true = self.__true_binarized[:, label.value]
        y_scores = [probs[label.value] for probs in self.__predicted_probs]
        return sk_pr_curve(y_true, y_scores) # precision, recall, thresholds


    # Needed for figure PredictionConfidenceViolin.
    def prediction_confidence_distribution(self) -> Dict[Label, List[float]]:
        """
        Calculates the distribution of prediction confidences for each true class.

        The confidence is the probability assigned to the correct class.

        Returns:
            A dictionary mapping each true label to a list of confidences.
        """
        confidence_distribution: Dict[Label, List[float]] = {label: [] for label in self.labels()}

        for true_label, probs in zip(self.__true_labels, self.__predicted_probs):
            true_label: Label
            probs: Prob_T
            confidence_distribution[true_label].append(probs[true_label.value].item())

        return confidence_distribution


    @lru_cache(maxsize=1)
    def performance(self) -> float:
        """
        Calculates the overall performance score as a weighted sum of key metrics.
        The weights are defined in the `MetricsConfig`.
        The result is cached after the first call.
        """
        return (
            config.accuracy_weight    * self.accuracy() +
            config.sensitivity_weight * self.sensitivity() +
            config.specificity_weight * self.specificity() +
            config.precision_weight   * self.precision() +
            config.f1_score_weight    * self.f1_score() +
            config.auc_roc_weight     * float(np.nan_to_num(self.roc_auc(), nan=0.0)) +
            config.auc_pr_weight      * float(np.nan_to_num(self.pr_auc(), nan=0.0))
        )


    def performance_summary(self) -> str:
        """Returns a detailed string showing the calculation of the performance score."""
        return (  # Nice to have for logging.
            f"---------- weight * metric -------\n"
            f"accuracy:    {config.accuracy_weight:.3f} * {self.accuracy():.3f} = {config.accuracy_weight * self.accuracy():.3f}\n"
            f"sensitivity: {config.sensitivity_weight:.3f} * {self.sensitivity():.3f} = {config.sensitivity_weight *  self.sensitivity():.3f}\n"
            f"specificity: {config.specificity_weight:.3f} * {self.specificity():.3f} = {config.specificity_weight * self.specificity():.3f}\n"
            f"precision:   {config.precision_weight:.3f} * {self.precision():.3f} = {config.precision_weight * self.precision():.3f}\n"
            f"f1_score:    {config.f1_score_weight:.3f} * {self.f1_score():.3f} = {config.f1_score_weight * self.f1_score():.3f}\n"
            f"auc_roc:     {config.auc_roc_weight:.3f} * {np.nan_to_num(self.roc_auc(), nan=0.0):.3f} = {config.auc_roc_weight * self.roc_auc():.3f}\n"
            f"auc_pr:      {config.auc_pr_weight:.3f} * {np.nan_to_num(self.pr_auc(), nan=0.0):.3f} = {config.auc_pr_weight * self.pr_auc():.3f}\n"
            f"------------ sum -----------------\n"
            f"performance: {self.performance():.4f}\n"
        )


    def get_all_metrics(self) -> Dict[PerformanceMetric, float]:
        """
        Gathers all primary performance metrics into a dictionary.

        This method maps each member of the PerformanceMetric enum to its
        corresponding calculated value (overall/averaged). It handles potential
        NaN values from AUC calculations by converting them to 0.0.

        Returns:
            A dictionary where keys are PerformanceMetric enum members and
            values are the float values of the corresponding metrics.
        """
        return {
            PerformanceMetric.ACCURACY: self.accuracy(),
            PerformanceMetric.SENSITIVITY: self.sensitivity(),
            PerformanceMetric.SPECIFICITY: self.specificity(),
            PerformanceMetric.PRECISION: self.precision(),
            PerformanceMetric.F1_SCORE: self.f1_score(),
            PerformanceMetric.AUC_ROC: float(np.nan_to_num(self.roc_auc(), nan=0.0)),
            PerformanceMetric.AUC_PR: float(np.nan_to_num(self.pr_auc(), nan=0.0)),
            PerformanceMetric.PERFORMANCE: self.performance(),
        }


    def generate_table(self) -> pd.DataFrame:
        """
        Generates a pandas DataFrame containing all metrics for each class,
        plus a final row for the overall (averaged) metrics.
        """
        data = {}
        for label in all_labels + [None]:
            data[label.name if label is not None else "Average"] = {
                "true_positives": self.true_positives(label),
                "false_positives": self.false_positives(label),
                "false_negatives": self.false_negatives(label),
                "true_negatives": self.true_negatives(label),
                "sensitivity": self.sensitivity(label),
                "specificity": self.specificity(label),
                "precision": self.precision(label),
                "accuracy": self.accuracy(label),
                "f1_score": self.f1_score(label),
                "roc_auc": self.roc_auc(label),
                "pr_auc": self.pr_auc(label),
                "class_weight": self.__class_weights[label] if label is not None and self.__class_weights is not None else None
            }
        return pd.DataFrame.from_dict(data, orient="index")


    def save_metrics_to_yaml(self, file_path: str, **kwargs: Any) -> None:
        """
        Saves the metrics table to a YAML file.

        Args:
            file_path: The path to the output YAML file.
            **kwargs: Additional key-value pairs to add as columns to the data.
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        df = self.generate_table()
        for key, value in kwargs.items(): df[key] = value
        df["performance"] = self.performance()

        with open(file_path, "a") as file:
            yaml.dump(df.to_dict(orient="index"), file, default_flow_style=False)


    def save_metrics_to_tsv(self, file_path: str, **kwargs: Any) -> None:
        """
        Appends the metrics table to a TSV file. Creates the file if it doesn't exist.

        Args:
            file_path: The path to the output TSV file.
            **kwargs: Additional key-value pairs to add as columns to the data.
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        df = self.generate_table()
        for key, value in kwargs.items():
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                df[key] = json.dumps(value)
            else:
                df[key] = value
        df["performance"] = self.performance()
        df.index.name = "class_name"

        file_exists = os.path.exists(file_path)
        if not file_exists:
            with open(file_path, 'w') as _: pass
            print(f"File created: '{file_path}'")

        with open(file_path, "a") as file:
            df.to_csv(file, index=True, header=not file_exists, sep='\t')


    def summary(self) -> str:
        """Returns a concise, human-readable summary of the key overall metrics."""
        return (
            f"Summary of Metrics:\n"
            f"- Samples: {self.total}\n"
            f"- Loss: {self.loss_per_sample:.3f}\n"
            f"- Accuracy: {self.accuracy()*100:.3f}%\n"
            f"- Precision (PPV): {self.precision()*100:.3f}%\n"
            f"- Recall (Sensitivity): {self.sensitivity()*100:.3f}%\n"
            f"- Specificity (TNR): {self.specificity()*100:.3f}%\n"
            f"- F1-Score: {self.f1_score()*100:.3f}%\n"
            f"- ROC AUC: {self.roc_auc() * 100:.3f}%\n"
            f"- PR AUC: {self.pr_auc() * 100:.3f}%\n"
            f"- Performance: {self.performance() * 100:.3f}%\n"
            f"- Weighted: {self.__class_weights is not None}\n"
        )


    def __str__(self) -> str:
        """Returns the summary string when the object is printed."""
        return self.summary()



from source.models.abstract import AbstractModel
from source.training.data_handling import DataHandler
from source.training.prediction_loop import Validator

def main(model: AbstractModel,
         encodings_file_path: str,
         batch_size: int = 28) -> Metrics:

    data_handler: DataHandler = DataHandler(
        encodings_file_path= encodings_file_path,
        batch_size = batch_size
    )

    validator: Validator = Validator(
        model = model,
        data_handler = data_handler
    )

    try:
        print(f"Beginning calculating Metrics.\nModel: {model}")
        metrics: Metrics = Metrics.from_generator(
            predictions_generator = validator.prediction_generator(),
            class_weights = data_handler.class_weights
        )
        print(f"Metrics calculated successfully.")
        return metrics

    except Exception as e:
        print(f"Error during calculation of metrics: {str(e)}")
        raise e



if __name__ == "__main__":
    from dotenv import load_dotenv

    # noinspection DuplicatedCode
    load_dotenv()

    __encodings_output_path = os.getenv('ENCODINGS_OUTPUT_DIR_LOCAL')
    __model_save_dir = os.getenv('MODEL_SAVE_DIR_LOCAL')
    __test_encodings_file_path = f"{__encodings_output_path}/setHARD.h5"

    if os.path.exists(__model_save_dir):
        __model_file_paths: List[str] = [
            os.path.join(__model_save_dir, __model_file_name)
            for __model_file_name in os.listdir(__model_save_dir)
        ]
    else:
        raise ValueError(f"The folder '{__model_save_dir}' does not exist.")
    # For testing at least one model should be saved locally

    for __model_file_path in __model_file_paths:
        print("Calculating metrics: ", __model_file_path)
        __model, _ = AbstractModel.load(__model_file_path)

        __metrics: Metrics = main(
            model = __model,
            encodings_file_path = __test_encodings_file_path
        )

        print(__metrics.summary())