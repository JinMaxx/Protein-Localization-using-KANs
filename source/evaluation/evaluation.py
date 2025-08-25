#!/usr/bin/python3
"""
This script provides a complete framework for evaluating and comparing the
performance of one or more trained models against a given test dataset.

It performs two main types of evaluation:
1.  Standard Metrics: Calculates metrics (accuracy, F1, etc.) across the entire test set.
2.  Sampled Metrics: Performs robust statistical analysis by repeatedly
    evaluating the model on random subsets of the data to derive
    confidence intervals for the metrics.
"""

import uuid

from typing_extensions import List, Optional

from source.metrics.metrics import Metrics
from source.models.abstract import AbstractModel
from source.training.data_handling import DataHandler
from source.training.prediction_loop import Validator
from source.metrics.metrics_sampled import MetricsSampled
from source.metrics.metrics_figures import MetricsFiguresCollection
from source.evaluation.evaluation_figures import EvaluationFiguresCollection

from source.config import parse_config,ConfigType, EvaluationConfig
config: EvaluationConfig = parse_config(ConfigType.Evaluation)



def evaluate_models(
        model_file_paths: List[str],
        test_encodings_file_path: str,
        figures_save_dir: Optional[str] = None,
        iterations: int = 100,
        batch_size: int = 28,
        use_class_weights: bool = False
) -> None:
    """
    Evaluates a list of models, calculates performance metrics, and generates comparison figures.

    This function iterates through each provided model file,
    performs a full evaluation using the entire test set,
    and also runs a sampled evaluation to assess statistical robustness.
    It generates and saves a collection of visualizations for analysis.

    Args:
        model_file_paths: A list of file paths to the trained models to be evaluated.
        test_encodings_file_path: The file path to the H5 file containing the test data.
        figures_save_dir: Optional. The directory where generated figures will be saved.
                          If None, figures are displayed but not saved.
        iterations: The number of sampling iterations to perform for the robust `MetricsSampled` calculation.
        batch_size: The batch size to use when loading data for evaluation.
        use_class_weights: Whether to use class weights when calculating metrics.
    """

    # Generate a unique identifier for this evaluation run to group figures
    identifier = f"hyper_param_{str(uuid.uuid4())}"
    print("Evaluation id:", identifier)

    data_handler: DataHandler = DataHandler(
        encodings_file_path = test_encodings_file_path,
        batch_size = batch_size
    )

    eval_figures: EvaluationFiguresCollection = EvaluationFiguresCollection(save_dir=figures_save_dir)
    eval_figures.accuracy_comparison(
        identifier = identifier,
        model_names = config.model_names,
        accuracies = config.accuracies,
        errors = config.accuracies_errors
    )

    metrics_figures: MetricsFiguresCollection = MetricsFiguresCollection(save_dir=figures_save_dir)
    # No need for identifier here because model.id() changes for each model
    metrics_figures.prediction_confidence_violin()
    metrics_figures.metrics_heatmap()
    metrics_figures.duo_curves()

    # Loop through each model for evaluation
    for model_file_path in model_file_paths:
        model, _ = AbstractModel.load(model_file_path)
        assert model.in_channels == data_handler.encoding_dim, "Model and data encoding dimensions do not match!"

        # Standard Metrics Calculation
        validator: Validator = Validator(
            model = model,
            data_handler = data_handler
        )

        metrics: Metrics = Metrics.from_generator(
            predictions_generator = validator.prediction_generator(),
            class_weights = None if not use_class_weights else data_handler.class_weights
        )

        # Update and save figures specific to this model's standard performance
        metrics_figures.update(
            model_name = model.name(),
            metrics = metrics,
            identifier = f"{model.id()}_{identifier}"  # overwriting identifier (in case when saving figures)
        )

        # Sampled Metrics Calculation (Robust Statistical Evaluation)
        metrics_sampled: MetricsSampled = MetricsSampled.calculate(
            model = model,
            iterations = iterations,
            batch_size = batch_size,
            test_encodings_file_path = test_encodings_file_path,
        )

        # Update the overall evaluation figures with this model's results
        eval_figures.update(
            model_name = model.name(),
            metrics = metrics,
            metrics_sampled = metrics_sampled
        )

        print(f"Model: {model.id()}")
        print(metrics.summary())
        print(metrics_sampled.summary())

        metrics_figures.save(sub_dir=f"evaluation/{model.name()}/{model.id()}")


    eval_figures.save(sub_dir="evaluation")
    del eval_figures, metrics_figures



if __name__ == "__main__":
    import os
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
    else: raise ValueError(f"The folder '{__model_save_dir}' does not exist.")
    # For testing at least one model should be saved locally

    for __model_file_path in __model_file_paths: print(__model_file_path)

    evaluate_models(
        model_file_paths = __model_file_paths,
        test_encodings_file_path = __test_encodings_file_path,
        iterations = 5  # for testing purposes
    )