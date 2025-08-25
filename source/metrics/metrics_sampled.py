#!/usr/bin/python3
"""
This script defines the MetricsSampled class, which is designed for robust
evaluation of a model's performance by repeatedly sampling a test dataset.
It calculates key performance metrics over multiple iterations and then provides
statistical analysis, including means and confidence intervals.
"""

import numpy as np

from typing_extensions import List, Tuple

from source.metrics.metrics import Metrics
from source.models.abstract import AbstractModel
from source.training.prediction_loop import Validator
from source.training.data_handling import DataHandlerRandom
from source.metrics.metrics_figures import MetricsFiguresCollection



class MetricsSampled:
    """
    Holds and analyzes performance metrics from multiple validation runs.

    This class is designed to aggregate metrics (accuracy, sensitivity, etc.)
    collected over numerous random samples of a dataset. It then calculates
    statistical properties like the mean, standard deviation, and confidence
    intervals for these metrics, providing a more robust assessment of model
    performance than a single validation run.

    The primary way to create an instance of this class is via the `calculate`
    static method, which performs the iterative evaluation.

    Attributes:
        accuracies (List[float]): A list of accuracy scores from each run.
        sensitivities (List[float]): A list of sensitivity scores from each run.
        specificities (List[float]): A list of specificity scores from each run.
        f1_scores (List[float]): A list of F1-scores from each run.
    """

    def __init__(self,
            accuracies: List[float],
            sensitivities: List[float],
            specificities: List[float],
            f1_scores: List[float]):
        """Initializes the MetricsSampled object with lists of metrics."""
        self.accuracies = accuracies
        self.sensitivities = sensitivities
        self.specificities = specificities
        self.f1_scores = f1_scores


    @staticmethod
    def __mean(numbers: List[float]) -> float:
        """Calculates the mean for a list of numbers."""
        return float(np.mean(numbers))


    @staticmethod
    def __standard_deviation(numbers: List[float]) -> float:
        """Calculates the standard deviation for a list of numbers."""
        return float(np.std(numbers))


    @staticmethod
    def __standard_error_of_mean(numbers: List[float]) -> float:
        """Calculates the standard error of the mean for a list of numbers."""
        return float(np.std(numbers) / np.sqrt(len(numbers)))


    @staticmethod
    def __confidence_interval(numbers: List[float], ci: int = 95) -> Tuple[float, float]:
        """
        Calculates the confidence interval for a list of numbers.

        Args:
            numbers: The list of numbers to analyze.
            ci: The desired confidence interval percentage (e.g., 95 for 95%).

        Returns:
            A tuple containing the lower and upper bounds of the confidence interval.
        """
        assert 0 <= ci <= 100, "ci must be between 0 and 100!"
        mean = MetricsSampled.__mean(numbers)
        sem = MetricsSampled.__standard_error_of_mean(numbers)
        return mean - sem * float(ci / 100) / 2, mean + sem * float(ci / 100) / 2


    def accuracies_mean_and_confidence_interval(self) -> Tuple[float, Tuple[float, float]]:
        """Returns the mean and 95% confidence interval for the accuracies."""
        return self.__mean(self.accuracies), self.__confidence_interval(self.accuracies)


    def sensitivities_mean_and_confidence_interval(self) -> Tuple[float, Tuple[float, float]]:
        """Returns the mean and 95% confidence interval for the sensitivities."""
        return self.__mean(self.sensitivities), self.__confidence_interval(self.sensitivities)


    def specificities_mean_and_confidence_interval(self) -> Tuple[float, Tuple[float, float]]:
        """Returns the mean and 95% confidence interval for the specificities."""
        return self.__mean(self.specificities), self.__confidence_interval(self.specificities)


    def f1_scores_mean_and_confidence_interval(self) -> Tuple[float, Tuple[float, float]]:
        """Returns the mean and 95% confidence interval for the F1-scores."""
        return self.__mean(self.f1_scores), self.__confidence_interval(self.f1_scores)


    @staticmethod
    def __format_tuple(_tuple: Tuple[float, float]) -> str:
        """Formats a tuple of floats into a '[lower - upper]' percentage string."""
        return f"[{_tuple[0]*100:.2f} - {_tuple[1]*100:.2f}]"


    def summary(self) -> str:
        """
        Generates a formatted, human-readable summary of the metrics.

        Returns:
            A string table showing the mean and 95% confidence interval for
            each key metric.
        """
        return (
            "\nMetricsSampled Summary:\n"
            "mean metric           95% confidence interval\n"
            f"Accuracy:     {self.__mean(self.accuracies)*100:.2f}   {self.__format_tuple(self.__confidence_interval(self.accuracies, 95))}\n"
            f"Sensitivity:  {self.__mean(self.sensitivities)*100:.2f}   {self.__format_tuple(self.__confidence_interval(self.sensitivities, 95))}\n"
            f"Specificity:  {self.__mean(self.specificities)*100:.2f}   {self.__format_tuple(self.__confidence_interval(self.specificities, 95))}\n"
            f"F1-Score:     {self.__mean(self.f1_scores)*100:.2f}   {self.__format_tuple(self.__confidence_interval(self.f1_scores, 95))}\n"
        )


    @staticmethod
    def calculate(
            model: AbstractModel,
            test_encodings_file_path: str,
            batch_size: int = 28,
            iterations: int = 100,
            use_class_weights: bool = False
    ) -> 'MetricsSampled':
        """
        Performs repeated random subsampling validation to assess performance.

        This method repeatedly runs the model on random batches of test data,
        calculating performance metrics for each run. This process, often called
        bootstrapping or repeated random subsampling validation, provides a
        distribution of metric scores, allowing for a more robust evaluation
        than a single test run.

        Args:
            model: The trained model to be evaluated.
            test_encodings_file_path: Path to the file containing test data.
            batch_size: The number of samples to use in each validation iteration.
            iterations: The number of validation iterations to perform.
            use_class_weights: Whether to use class weights when calculating metrics.

        Returns:
            A `MetricsSampled` object containing the aggregated results from all
            validation runs.
        """
        print("Calculating random sampled metrics...")
        print("Iterations: ", iterations)

        data_handler: DataHandlerRandom = DataHandlerRandom(
            encodings_file_path = test_encodings_file_path,
            batch_size = batch_size
        )
        assert model.in_channels == data_handler.encoding_dim, "Model and data encoding dimensions do not match!"

        model_figures: MetricsFiguresCollection = MetricsFiguresCollection()
        model_figures.prediction_confidence_violin(identifier=model.id())
        model_figures.metrics_heatmap(identifier=model.id())
        model_figures.duo_curves(identifier=model.id())

        validator: Validator = Validator(
            model = model,
            data_handler = data_handler
        )

        accuracies: List[float] = []
        specificities: List[float] = []
        sensitivities: List[float] = []
        f1_scores: List[float] = []

        for i in range(iterations):

            print("Iteration: ", i)

            # Calculate metrics for one random sample of the data
            metrics: Metrics = Metrics.from_generator(
                predictions_generator = validator.prediction_generator(),
                class_weights = None if not use_class_weights else data_handler.class_weights
            )

            accuracies.append(metrics.accuracy())
            specificities.append(metrics.specificity())
            sensitivities.append(metrics.sensitivity())
            f1_scores.append(metrics.f1_score())

            model_figures.update(
                model_name = model.name(),
                metrics = metrics
            )

            print(metrics.summary())

        del model_figures

        metrics_sampled: MetricsSampled = MetricsSampled(
            accuracies = accuracies,
            sensitivities = sensitivities,
            specificities = specificities,
            f1_scores = f1_scores
        )

        print(metrics_sampled.summary())

        return metrics_sampled



if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # Get paths from environment variables
    __model_save_dir = os.getenv('MODEL_SAVE_DIR_LOCAL')
    __encodings_output_path = os.getenv('ENCODINGS_OUTPUT_DIR_LOCAL')

    if not __model_save_dir or not os.path.exists(__model_save_dir):
        raise ValueError(f"The model directory '{__model_save_dir}' does not exist or is not set in .env.")

    __test_encodings_file_path = f"{__encodings_output_path}/setHARD.h5"
    if not os.path.exists(__test_encodings_file_path):
        raise ValueError(f"The test data file '{__test_encodings_file_path}' does not exist.")

    # Find all saved models
    __model_files = [f for f in os.listdir(__model_save_dir) if os.path.isfile(os.path.join(__model_save_dir, f))]

    if not __model_files:
        print(f"No models found in '{__model_save_dir}'. Skipping test run.")
    else:
        # Run test on the first model found
        __first_model_path = os.path.join(__model_save_dir, __model_files[0])
        print(f"--- Running MetricsSampled Test on: {os.path.basename(__first_model_path)} ---")

        try:
            # Load the model
            __model, _ = AbstractModel.load(__first_model_path)

            # Run the sampled metrics calculation with a low iteration count for a quick test
            __metrics_sampled = MetricsSampled.calculate(
                model = __model,
                test_encodings_file_path = __test_encodings_file_path,
                iterations = 5
            )

            # Print the final summary from the returned object
            print(__metrics_sampled.summary())

            print("\n--- Test run finished successfully ---")

        except Exception as e:
            print(f"An error occurred during the test run: {e}")
            raise