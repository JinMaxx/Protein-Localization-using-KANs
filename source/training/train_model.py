#!/usr/bin/python3
"""
Provides a comprehensive framework for orchestrating the training and evaluation of PyTorch models.
This module acts as the main driver for the training lifecycle, offering high-level functions
to either start training from scratch (`train_new`) or resume from a saved checkpoint (`train_continue`).

Key functionalities include:
- A core training loop (`__train`) with built-in support for early stopping
  based on validation performance, learning rate scheduling, and saving of the best model state.
- High-level wrappers that handle all setup, including data loading, model instantiation, and reporting results.
- Tight integration with `Metrics` and `TrainingFiguresCollection` for real-time performance tracking
  and the generation of training plots.
- Robust state management using a `SaveState` object, which preserves training history (e.g., loss, metrics)
  and enables seamless continuation.
- Graceful error handling for keyboard interrupts and context management for logging.
"""

import os
import torch

from typing_extensions import Type, Tuple, Optional, Dict, Set
from contextlib import AbstractContextManager, nullcontext

from source.training.utils.context_manager import TeeStdout, OptionalTempDir

from source.models.abstract import AbstractModel
from source.training.utils.save_state import SaveState
from source.metrics.metrics import Metrics, PerformanceMetric
from source.training.prediction_loop import Trainer, Validator
from source.training.training_figures import TrainingFiguresCollection
from source.training.data_handling import DataHandler, read_files as read_data_handlers

from source.config import TrainingConfig, ConfigType, parse_config

training_config: TrainingConfig = parse_config(ConfigType.Training)


__METRICS_SET: Set[PerformanceMetric] = {
    PerformanceMetric.ACCURACY,
    PerformanceMetric.F1_SCORE,
    PerformanceMetric.PERFORMANCE
}



class InterruptedTrainException(KeyboardInterrupt):
    """
    A custom exception raised when training is interrupted by the user.

    This exception extends KeyboardInterrupt and carries the model and its save state at the moment of interruption,
    allowing for graceful recovery and saving of the partial work.
    """

    def __init__(self, model: AbstractModel, save_state: SaveState, metrics: Optional[Metrics]):
        super().__init__()
        self.model: AbstractModel = model
        self.save_state: SaveState = save_state
        self.metrics: Optional[Metrics] = metrics



def __train(
        model: AbstractModel,
        train_data_handler: DataHandler,
        val_data_handler: DataHandler,
        save_state: SaveState,
        figures: TrainingFiguresCollection,
        epochs: int = training_config.epochs,
        patience: int = training_config.patience,
        use_weights: bool = training_config.use_weights,
        weight_decay: float = training_config.weight_decay,
        learning_rate: float = training_config.learning_rate,
        learning_rate_decay: float = training_config.learning_rate_decay,
        model_save_dir: Optional[str] = None,
        metrics_file_path: Optional[str] = None
) -> Tuple[AbstractModel, SaveState, Optional[Metrics]]:
    """
    Executes the core training and validation loop for a given model.

    This function iterates through epochs, performs training and validation, tracks metrics,
    implements early stopping, and saves the best-performing model state.

    :param model: The model instance to be trained.
    :param train_data_handler: Data handler for the training dataset.
    :param val_data_handler: Data handler for the validation dataset.
    :param save_state: Object managing the model's state across epochs.
    :param figures: Collection for generating and saving training plots.
    :param epochs: The total number of epochs to train.
    :param patience: Number of epochs to wait for improvement before early stopping.
    :param use_weights: Whether to use class weights in the loss function.
    :param weight_decay: The weight decay (L2 penalty) for the optimizer.
    :param learning_rate: The initial learning rate for the optimizer.
    :param learning_rate_decay: The decay factor for the learning rate scheduler.
    :param model_save_dir: Directory to save the best model. Uses a temp dir if None.
    :param metrics_file_path: Path to save epoch-wise metrics in TSV format.
    :return: A tuple containing the best model, its final save state, and the best metrics.
    """
    assert epochs > 0, "Epochs must be greater than 0!"
    epoch: int = save_state.current_epoch() + 1
    epochs += epoch  # to continue training from previous epochs.

    trainer: Trainer = Trainer(
        model = model,
        data_handler = train_data_handler,
        use_weights = use_weights,
        weight_decay = weight_decay,
        learning_rate = learning_rate,
        learning_rate_decay = learning_rate_decay,
    )
    validator: Validator = Validator(
        model = model,
        data_handler = val_data_handler
    )

    best_model_file_path: str | None = None
    best_metrics: Metrics | None = None
    best_performance: float = -float("inf")  # for this run of training, not accounting previous runs.
    patience_counter: int = 0

    with OptionalTempDir(supplied_dir=model_save_dir, prefix="tmp_model_save_") as model_save_dir:

        try:
            for epoch in range(epoch, epochs):

                # Training Loop
                total_loss: float = 0.0
                total_samples: int = 0
                for _, _, loss in trainer.prediction_generator():
                    total_loss += loss
                    total_samples += 1
                train_loss_per_sample: float = total_loss / total_samples

                # Validation Loop
                val_metrics: Metrics = Metrics.from_generator(
                    predictions_generator = validator.prediction_generator(),
                    class_weights = val_data_handler.class_weights
                )

                val_performances: Dict[PerformanceMetric, float] = val_metrics.get_all_metrics()

                if metrics_file_path is not None:
                    val_metrics.save_metrics_to_tsv(
                        metrics_file_path,
                        model_id = model.id(),
                        model_name = model.name(),
                        memory_size = model.memory_size(),
                        epoch = epoch,
                        train_loss_per_sample=train_loss_per_sample,
                        validation_loss_per_sample=val_metrics.loss_per_sample,
                    )

                save_state.increment(
                    training_loss = train_loss_per_sample,
                    validation_loss = val_metrics.loss_per_sample,
                    performance_values = val_performances
                )

                figures.update(
                    model_name = model.name(),
                    metrics = val_metrics,
                    epoch = epoch,
                    performance_values = val_performances,
                    train_loss = train_loss_per_sample,
                    val_loss = val_metrics.loss_per_sample
                )
                print(f"\nEpoch: {epoch}")
                print(val_metrics.summary())
                print(val_metrics.performance_summary())

                # model.save( # Saving the current model in case Google decides to kill the runtime ungracefully
                #     save_dir = model_save_dir,
                #     identifier = f"{model.id()}_safe",
                #     save_state = save_state.clone().update(best_epoch=epoch)
                # )

                if val_performances[PerformanceMetric.PERFORMANCE] > best_performance:
                    print(f"Validation Performance improved from {best_performance:.4f} to {val_performances[PerformanceMetric.PERFORMANCE]:.4f} at Epoch: {epoch}")
                    print(f"Validation Accuracy is {val_performances[PerformanceMetric.ACCURACY]:.4f}")
                    patience_counter = 0
                    best_metrics = val_metrics
                    best_performance = val_performances[PerformanceMetric.PERFORMANCE]
                    save_state.update(best_epoch=epoch)
                    figures.save(sub_dir=f"{model.name()}/{model.id()}", epoch=epoch)
                    best_model_file_path = model.save(
                        save_dir = model_save_dir,
                        identifier = model.id(),  # overwriting worse models (otherwise save as f"{model.id()}_epoch={epoch}")
                        save_state = save_state   # Also saving data needed to continue training from this point.
                    )  # Saving good models (overwriting worse models)
                else:
                    patience_counter += 1
                    if patience_counter >= patience: # Early stopping
                        print("Early stopping triggered!")
                        print(f"Best Performance: {best_performance:.4f} at Epoch: {epoch - patience_counter})")
                        break

        except KeyboardInterrupt:
            print(f"Training interrupted at {epoch}")
            raise InterruptedTrainException(model, save_state, best_metrics)

        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise e

        # End of training
        figures.save(sub_dir=f"{model.name()}/{model.id()}", epoch=epoch)

        # Loading the best model
        try: model, save_state = SaveState.load(model_file_path=best_model_file_path)
        except Exception as e:
            print(f"Failed to load best model from {best_model_file_path}: {e}")
            raise e


    save_state.clean()

    # noinspection PyUnboundLocalVariable
    return model, save_state, best_metrics  # with epochs > 0 asserted that best_metrics is not None.



def train_wrap(
        model: AbstractModel,
        save_state: SaveState,
        train_data_handler: DataHandler,
        val_data_handler: DataHandler,
        figures: TrainingFiguresCollection,
        epochs: int = training_config.epochs,
        patience: int = training_config.patience,
        use_weights: bool = training_config.use_weights,
        weight_decay: float = training_config.weight_decay,
        learning_rate: float = training_config.learning_rate,
        learning_rate_decay: float = training_config.learning_rate_decay,
        model_save_dir: Optional[str] = None,
        metrics_file_path: Optional[str] = None
) -> Tuple[AbstractModel, SaveState, Metrics]:
    """
    A wrapper for the training process that handles setup, cleanup, and error reporting.

    :param model: The model instance to be trained.
    :param save_state: The initial state for training.
    :param train_data_handler: Data handler for training data.
    :param val_data_handler: Data handler for validation data.
    :param figures: Collection for managing training plots.
    :param epochs: Total number of epochs for this training run.
    :param patience: Patience for early stopping.
    :param use_weights: Whether to use class weights.
    :param weight_decay: L2 penalty for the optimizer.
    :param learning_rate: Initial learning rate.
    :param learning_rate_decay: Learning rate decay factor.
    :param model_save_dir: Directory to save the final model.
    :param metrics_file_path: Path to save metrics.
    :return: A tuple of the best model, its final state, and metrics.
    """
    print(f"Training Model:\n{model.id()}\n{model}")
    print(f"Model Configuration:\n{model.get_config()}\n")

    try:
        model, save_state, metrics = __train(
            model = model,
            train_data_handler = train_data_handler,
            val_data_handler = val_data_handler,
            save_state = save_state,
            figures = figures,
            epochs = epochs,
            patience = patience,
            use_weights = use_weights,
            weight_decay = weight_decay,
            learning_rate = learning_rate,
            learning_rate_decay = learning_rate_decay,
            model_save_dir=model_save_dir,
            metrics_file_path=metrics_file_path
        )
        model: AbstractModel
        save_state: SaveState
        metrics: Metrics

        print(f"Training complete: {model.id()}")
        print(f"Best model at epoch: {save_state.current_epoch()}")
        if metrics is not None: print(metrics.summary())

        return model, save_state, metrics


    except Exception as e:
        print(f"Error during training: {str(e)}")
        if model_save_dir is not None:
            save_state.update(best_epoch=None)
            model.save(
                model_save_dir,
                identifier = f"{model.id()}_{e.__class__.__name__}",
                save_state = save_state
            )
        raise e

    finally:
        del figures
        torch.cuda.empty_cache()



def train_new(
        model_class: Type[AbstractModel],
        train_encodings_file_path: str,
        val_encodings_file_path: str,
        epochs: int = training_config.epochs,
        patience: int = training_config.patience,
        batch_size: int = training_config.batch_size,
        use_weights: bool = training_config.use_weights,
        weight_decay: float = training_config.weight_decay,
        learning_rate: float = training_config.learning_rate,
        learning_rate_decay: float = training_config.learning_rate_decay,
        model_save_dir: Optional[str] = None,
        figures_save_dir: Optional[str] = None,
        metrics_file_path: Optional[str] = None
) -> Tuple[AbstractModel, SaveState, Metrics]:
    """
    Initializes and trains a new model from scratch.

    :param model_class: The class of the model to be instantiated.
    :param train_encodings_file_path: Path to the training data HDF5 file.
    :param val_encodings_file_path: Path to the validation data HDF5 file.
    :param epochs: Number of epochs to train.
    :param patience: Patience for early stopping.
    :param batch_size: Batch size for training and validation.
    :param use_weights: Whether to use a weighted loss.
    :param weight_decay: L2 penalty.
    :param learning_rate: Initial learning rate.
    :param learning_rate_decay: Learning rate decay factor.
    :param model_save_dir: Directory to save the trained model.
    :param figures_save_dir: Directory to save training figures.
    :param metrics_file_path: Path to save metrics results.
    :return: A tuple of the trained model, its state, and metrics.
    """
    train_data_handler, val_data_handler, encoding_dim = read_data_handlers(
        train_encodings_file_path = train_encodings_file_path,
        val_encodings_file_path = val_encodings_file_path,
        batch_size = batch_size
    )

    model: AbstractModel = model_class(in_channels=encoding_dim)

    figures: TrainingFiguresCollection = TrainingFiguresCollection(save_dir=figures_save_dir)

    # specific figures for training
    figures.epoch_dual_axis_figure(
        model_name = model.name(),
        identifier = model.id(),
        figure_left = figures.training_loss(model_name=model.name(), epochs=epochs),
        figure_right = figures.training_performance(
            model_name = model.name(),
            epochs = epochs,
            performance_metrics = __METRICS_SET
        )
    )

    # specific figures for metrics
    figures.duo_curves(identifier=model.id())
    figures.metrics_heatmap(identifier=model.id())
    figures.prediction_confidence_violin(identifier=model.id())


    # specific figures for models (e.g. weight matrices, KAN representation, ...)
    model.add_figures_collection(figures=figures)

    save_state: SaveState = SaveState()

    # noinspection DuplicatedCode
    model, save_state, metrics = train_wrap(
        model = model,
        save_state = save_state,
        train_data_handler = train_data_handler,
        val_data_handler = val_data_handler,
        figures = figures,
        epochs = epochs,
        patience = patience,
        use_weights = use_weights,
        weight_decay = weight_decay,
        learning_rate = learning_rate,
        learning_rate_decay = learning_rate_decay,
        model_save_dir = model_save_dir,
        metrics_file_path = metrics_file_path
    )
    model: AbstractModel
    save_state: SaveState
    metrics: Metrics

    del figures

    return model, save_state, metrics



def train_continue(
        model_file_path: str,
        train_encodings_file_path: str,
        val_encodings_file_path: str,
        epochs: int = training_config.epochs,
        patience: int = training_config.patience,
        batch_size: int = training_config.batch_size,
        use_weights: bool = training_config.use_weights,
        weight_decay: float = training_config.weight_decay,
        learning_rate: float = training_config.learning_rate,
        learning_rate_decay: float = training_config.learning_rate_decay,
        model_save_dir: Optional[str] = None,
        figures_save_dir: Optional[str] = None,
        metrics_file_path: Optional[str] = None
) -> Tuple[AbstractModel, SaveState, Metrics]:
    """
    Continues training for a previously saved model.

    :param model_file_path: Path to the saved model file (.pt).
    :param train_encodings_file_path: Path to the training data.
    :param val_encodings_file_path: Path to the validation data.
    :param epochs: Number of additional epochs to train.
    :param patience: Patience for early stopping.
    :param batch_size: Batch size for data loaders.
    :param use_weights: Whether to use weighted loss.
    :param weight_decay: L2 penalty.
    :param learning_rate: Initial learning rate.
    :param learning_rate_decay: Learning rate decay factor.
    :param model_save_dir: Directory to save the updated model.
    :param figures_save_dir: Directory to save updated figures.
    :param metrics_file_path: Path to append new metrics.
    :return: A tuple of the trained model, its state, and metrics.
    """
    print("Continuing training")

    train_data_handler, val_data_handler, encoding_dim = read_data_handlers(
        train_encodings_file_path = train_encodings_file_path,
        val_encodings_file_path = val_encodings_file_path,
        batch_size = batch_size
    )

    model, save_state = SaveState.load(model_file_path=model_file_path)
    assert model.in_channels == encoding_dim, "Inconsistent encoding dimensions!"
    model.up_version()  # Give the model a new uuid to avoid overwriting the old model.

    print(f"Current Epoch: {save_state.current_epoch()}")

    figures: TrainingFiguresCollection = TrainingFiguresCollection(save_dir=figures_save_dir)

    # specific figures for training
    figure_left = figures.training_loss(
        model_name = model.name(),
        identifier = model.id(),
        epochs = epochs,
        train_losses = save_state.training_loss_history,
        val_losses = save_state.validation_loss_history
    )
    figure_right = figures.training_performance(
            model_name = model.name(),
            epochs = epochs,
            performance_metrics = __METRICS_SET,
            performance_data = save_state.performance_history
    )
    for figure in [figure_left, figure_right]: figure.update(epoch_cut=save_state.current_epoch())
    figures.epoch_dual_axis_figure(
        model_name = model.name(),
        identifier = model.id(),
        figure_left = figure_left,
        figure_right = figure_right
    )

    # specific figures for metrics
    figures.duo_curves(identifier=model.id())
    figures.metrics_heatmap(identifier=model.id())
    figures.prediction_confidence_violin(identifier=model.id())

    # specific figures for models (e.g. weight matrices, KAN representation, ...)
    model.add_figures_collection(figures=figures)

    # Visual marker for when training was continued.
    figures.update(epoch_cut=save_state.current_epoch())

    # noinspection DuplicatedCode
    model, save_state, metrics = train_wrap(
        model = model,
        save_state = save_state,
        train_data_handler = train_data_handler,
        val_data_handler = val_data_handler,
        figures = figures,
        epochs = epochs,
        patience = patience,
        use_weights = use_weights,
        weight_decay = weight_decay,
        learning_rate = learning_rate,
        learning_rate_decay = learning_rate_decay,
        model_save_dir = model_save_dir,
        metrics_file_path = metrics_file_path
    )
    model: AbstractModel
    save_state: SaveState
    metrics: Metrics

    del figures

    return model, save_state, metrics



# simply training a new initialized model for max epochs.
#@typechecked
def main(model: Type[AbstractModel] | str,
         train_encodings_file_path: str,
         val_encodings_file_path: str,
         epochs: int = training_config.epochs,
         patience: int = training_config.patience,
         batch_size: int = training_config.batch_size,
         use_weights: bool = training_config.use_weights,
         weight_decay: float = training_config.weight_decay,
         learning_rate: float = training_config.learning_rate,
         learning_rate_decay: float = training_config.learning_rate_decay,
         model_save_dir: Optional[str] = None,
         figures_save_dir: Optional[str] = None,
         metrics_file_path: Optional[str] = None,
         log_file_path: Optional[str] = None
) -> Tuple[AbstractModel, SaveState, Metrics]:
    """
    Main entry point to start or continue a training process.

    Dispatches to `train_new` if `model` is a class type, or `train_continue`
    if `model` is a string path to a saved model.

    :param model: The model class to train, or a path to a saved model file.
    :param train_encodings_file_path: Path to the training data.
    :param val_encodings_file_path: Path to the validation data.
    :param epochs: Number of epochs to train.
    :param patience: Patience for early stopping.
    :param batch_size: Data batch size.
    :param use_weights: Whether to use weighted loss.
    :param weight_decay: L2 penalty.
    :param learning_rate: Initial learning rate.
    :param learning_rate_decay: Learning rate decay factor.
    :param model_save_dir: Directory to save models.
    :param figures_save_dir: Directory to save figures.
    :param metrics_file_path: Path to save metrics.
    :param log_file_path: Optional path to redirect stdout to a log file.
    :return: A tuple of the trained model, its state, and performance metrics.
    """
    context: AbstractContextManager = TeeStdout(filename=log_file_path) if log_file_path is not None else nullcontext()
    with context:  # saving std out to log_file_path for tuning.

        print("Training.")
        print(f"epochs: {epochs}, patience: {patience}, batch_size: {batch_size}")
        print(f"weight_decay: {weight_decay}, learning_rate: {learning_rate}, learning_rate_decay: {learning_rate_decay}")
        print(f"train_data: {train_encodings_file_path}\nval_data: {val_encodings_file_path}")
        print(f"model_save_dir: {model_save_dir}\nfigures_save_dir: {figures_save_dir}")

        try:

            if isinstance(model, type):
                return train_new(
                    model_class = model,
                    train_encodings_file_path = train_encodings_file_path,
                    val_encodings_file_path = val_encodings_file_path,
                    epochs = epochs,
                    patience = patience,
                    batch_size = batch_size,
                    use_weights = use_weights,
                    weight_decay = weight_decay,
                    learning_rate = learning_rate,
                    learning_rate_decay = learning_rate_decay,
                    model_save_dir = model_save_dir,
                    figures_save_dir = figures_save_dir,
                    metrics_file_path = metrics_file_path
                )

            elif isinstance(model, str):
                return train_continue(
                    model_file_path = model,
                    train_encodings_file_path = train_encodings_file_path,
                    val_encodings_file_path = val_encodings_file_path,
                    epochs = epochs,
                    patience = patience,
                    batch_size = batch_size,
                    use_weights = use_weights,
                    weight_decay = weight_decay,
                    learning_rate = learning_rate,
                    learning_rate_decay = learning_rate_decay,
                    model_save_dir = model_save_dir,
                    figures_save_dir = figures_save_dir,
                    metrics_file_path = metrics_file_path
                )

            # else: raise TypeError(f"model must be either a type or a string, not {type(model)}.")

        except InterruptedTrainException as error:
            return error.model, error.save_state, error.metrics



if __name__ == "__main__":
    from dotenv import load_dotenv
    from source.models.ffn import FastKAN as __ModelClass

    load_dotenv()

    __encodings_output_path = os.getenv('ENCODINGS_OUTPUT_DIR_LOCAL')
    __model_save_dir = os.getenv('MODEL_SAVE_DIR_LOCAL')

    __train_encodings_file_path = f"{__encodings_output_path}/deeploc_our_train_set.h5"
    __val_encodings_file_path = f"{__encodings_output_path}/deeploc_our_val_set.h5"

    main(
        model = __ModelClass,
        train_encodings_file_path = __train_encodings_file_path,
        val_encodings_file_path = __val_encodings_file_path,
        model_save_dir = __model_save_dir,  # remove if saving not needed
        # for debugging
        epochs = 2,
        batch_size = 4
    )