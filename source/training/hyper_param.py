#!/usr/bin/python3
"""
This script provides functionality for hyperparameter optimization of tunable models using the Optuna framework.
It defines an objective function to train and evaluate a model with a given set of hyperparameters
and then uses an Optuna study to find the optimal combination of these parameters based on a performance metric.

The script is designed to be robust, handling common issues like out-of-memory errors
and allowing studies to be resumed.
It also generates visualizations of the tuning process.
"""

import os
import gc
import uuid

import torch
import optuna

from optuna import Trial, Study
from optuna.samplers import TPESampler as Sampler  # GPSampler, TPESampler. Which Sampler might be better...?

from typing_extensions import List, Type, Optional, Tuple
from contextlib import nullcontext, AbstractContextManager

from source.metrics.metrics import Metrics
from source.models.abstract import AbstractTunableModel
from source.training.train_model import train_new as train, SaveState
from source.training.utils.context_manager import TeeStdout, OptionalTempDir
from source.training.training_figures import HyperParamFiguresCollection, _AbstractHyperParamFigure

from source.config import TrainingConfig, HyperParamConfig, ConfigType, parse_config

training_config: TrainingConfig = parse_config(ConfigType.Training)
hyper_param_config: HyperParamConfig = parse_config(ConfigType.HyperParam)



def cleanup_zombie_trials(study: Study) -> bool:
    """
    Finds and fails any trials that are stuck in the 'RUNNING' state,
    which usually indicates a catastrophic crash in a previous session.

    Args:
        study: The Optuna study object.

    Returns:
        True if any zombie trials were found and failed, False otherwise.
    """
    zombie_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.RUNNING]
    if not zombie_trials: return False
    print(f"Found {len(zombie_trials)} zombie trial(s) from a previous crashed session.")
    for trial in zombie_trials:
        # noinspection PyProtectedMember
        study._storage.set_trial_state_values(trial.state, state=optuna.trial.TrialState.FAIL)
        # noinspection PyProtectedMember
        study._storage.set_trial_user_attr(trial._trial_id, "failure_reason", "System crash (likely OOM CPU RAM)")
        print(f"Marked Trial #{trial.number} as FAILED.")
    return True



def tune(
        model_class: Type[AbstractTunableModel],
        train_encodings_file_path: str,
        val_encodings_file_path: str,
        epochs: int = training_config.epochs,
        patience: int = training_config.patience,
        batch_size: int = training_config.batch_size,
        use_weights: bool = training_config.use_weights,
        weight_decay: float = training_config.weight_decay,
        learning_rate: float = training_config.learning_rate,
        learning_rate_decay: float = training_config.learning_rate_decay,
        n_trials: Optional[int] = hyper_param_config.n_trials,
        timeout: Optional[int] = hyper_param_config.n_trials,
        model_save_dir: Optional[str] = None,
        figures_save_dir: Optional[str] = None,
        study_name: Optional[str] = None,
        studies_save_dir: Optional[str] = None,
        metrics_file_path: Optional[str] = None,
) -> Optional[Tuple[AbstractTunableModel, SaveState, Metrics]]:  # best model according to performance metrics
    """
    Performs hyperparameter tuning for a given model class using Optuna.

    This function sets up and runs an Optuna study.
    For each trial, it creates a model with a new set of hyperparameters, trains it, and evaluates its performance.
    The best-performing model across all trials is saved and returned.

    :param model_class: The class of the model to be tuned.
    :param train_encodings_file_path: Path to the training data.
    :param val_encodings_file_path: Path to the validation data.
    :param epochs: The maximum number of training epochs per trial.
    :param patience: The patience for early stopping.
    :param batch_size: The batch size for training.
    :param use_weights: Whether to use class weights for the optimizer.
    :param weight_decay: The weight decay for the optimizer.
    :param learning_rate: The learning rate for the optimizer.
    :param learning_rate_decay: The learning rate decay factor.
    :param n_trials: The number of trials to run.
    :param timeout: The time limit for the study in seconds.
    :param model_save_dir: Directory to save the best model.
    :param figures_save_dir: Directory to save performance figures.
    :param study_name: The name for the Optuna study.
    :param studies_save_dir: Directory to save the Optuna study object.
    :param metrics_file_path: Path to save performance metrics for each trial.
    :return: A tuple containing the best model, its save state, and its metrics,
             or None if no trial completes successfully.
    """
    study_name: str = study_name or f"study_{model_class.name()}"
    print(f"Study Name: {study_name}")
    study_run: str = str(uuid.uuid4())
    print(f"Study run: {study_run}")

    with OptionalTempDir(supplied_dir=model_save_dir, prefix="tmp_model_save_") as model_save_dir:

        # --------------------------------

        def objective(trial: Trial):
            """
            The objective function for an Optuna trial.
            It trains a model with a set of hyperparameters and returns a performance score.
            """
            try:
                # Create a model with hyperparameters suggested by the trial
                test_model_class = model_class.create_test_model_class(trial)

                # Train the ffn_layer
                model, save_state, metrics = train(
                    model_class = test_model_class,
                    train_encodings_file_path = train_encodings_file_path,
                    val_encodings_file_path = val_encodings_file_path,
                    learning_rate_decay = learning_rate_decay,
                    learning_rate = learning_rate,
                    weight_decay = weight_decay,
                    use_weights = use_weights,
                    batch_size = batch_size,
                    patience = patience,
                    epochs = epochs,
                    figures_save_dir = figures_save_dir,
                    model_save_dir = None  # Not saving models because of storage constraints.

                )
                model: AbstractTunableModel
                save_state: SaveState
                metrics: Metrics

            except (ValueError, torch.cuda.OutOfMemoryError, MemoryError) as error:
                # Prune trials that fail due to common issues like invalid model architecture
                # or memory limitations, allowing Optuna to avoid similar hyperparameter regions.
                error_type = type(error).__name__
                print(f"{error_type} during training: {error}")
                if "CUDA" in error_type:
                    torch.cuda.empty_cache()
                raise optuna.TrialPruned(f"{error_type}: {error}")
            except KeyboardInterrupt:
                # Allow graceful stopping of the study
                trial.study.stop()
                raise optuna.TrialPruned("KeyboardInterrupt")
            except Exception as error:
                print(f"An unexpected error occurred during training: {error}")
                raise error

            performance = metrics.performance()

            if metrics_file_path is not None:
                metrics.save_metrics_to_tsv(
                    metrics_file_path,
                    model = model.id(),
                    # model_class = model.__class__.__name__,  # usually the same as study_name  TODO: Uncomment afterwards!
                    memory_size = model.memory_size(),
                    epoch = save_state.current_epoch(),
                    study_name = study_name,
                    trial_number = trial.number,
                    trial_params = list(trial.params.items())
                )

            # Check if the current trial is the best one so far and save its model
            successful_trials = [_trial for _trial in trial.study.trials if _trial.state == optuna.trial.TrialState.COMPLETE]
            if not successful_trials or performance > trial.study.best_value:
                trial.study.set_user_attr("best_model_id", model.id())  # Keep track of the best model of study.
                trial.study.set_user_attr("best_metrics", metrics.to_dict())
                trial.study.set_user_attr(
                    "best_model_file_path",
                    model.save(  # only saving the best model of all trials.
                        save_dir = model_save_dir,
                        identifier = f"{model.name()}_{study_name}_{study_run}",
                        save_state = save_state
                    )
                )
                print(f"Best model updated: {model.id()}\nSaved with identifier: {model.name()}_{study_name}_{study_run}")

            # Clean up to free memory for the next trial
            model.cpu()  # deleting like this can avoid gpu memory fragmentation.
            del test_model_class, model, metrics
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

            return performance


        # --------------------------------

        # Load an existing study or create a new one
        storage_url: str | None = None
        if studies_save_dir is not None:
            os.makedirs(studies_save_dir, exist_ok=True)
            db_file_path = os.path.join(studies_save_dir, f"{study_name}.db")
            storage_url = f"sqlite:///{db_file_path}"

        study: Study = optuna.create_study(
            direction = "maximize",
            sampler = Sampler(seed=42),
            study_name = study_name,
            storage = storage_url,
            load_if_exists = True
        )

        cleanup_zombie_trials(study)  # At the beginning of each study, remove any zombie trials.

        print(f"Successfully loaded study '{study.study_name}' using SQLite backend.")
        if study.trials:
            print("Existing trials:")
            print(study.trials_dataframe().tail())

        # Set up visualization for the tuning process
        figures: HyperParamFiguresCollection = HyperParamFiguresCollection(save_dir=figures_save_dir)
        grid_figures: List[_AbstractHyperParamFigure] = [
            figures.hyper_param_parameter(
                study = study,
                param_name = param_name,
                param_info = param_info,
                identifier = f"{param_name}_{study_name}"
            ) for param_name, param_info in model_class.get_config().test_parameters_info()
        ]
        grid_figures.append(figures.hyper_param_performance(identifier=study_name))
        figures.multifigure(figures=grid_figures, show_legend=False, identifier=study_name)

        # If possible (study is continued), display figures at the start.
        if any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
            print("Displaying existing study history...")
            figures.update(clear=False, performances=[t.value for t in study.trials if t.state.is_finished()])
            figures.display(clear=False)

        # Run the optimization
        study.optimize(
            objective,
            n_trials = n_trials,
            timeout = timeout,
            n_jobs = 1,
            callbacks = [                            # Makes the previous figures from training not disappear right after the trial is finished.
                (lambda _study, _trial: figures.update(clear=False, performances=[t.value for t in study.trials if t.state.is_finished]))
            ]   # Define callbacks for post-trial operations
        )

        figures.display(clear=False)
        figures.save(f"hyper_param/{study_name}")
        del figures


        # Output the best trial (hyperparameters and result)
        print("\n\n----------------------------------------------------------------\n")
        best_trial = study.best_trial
        print(f"Best trial: {best_trial.params}")
        print(f"Best value (validation loss): {best_trial.value}")
        print(study.trials_dataframe())

        if "best_model_id" not in study.user_attrs:
            print("No best model found!")
            print("All trials failed or no trials were completed!")
            return None
        else:
            best_model_id = study.user_attrs.get("best_model_id")
            print(f"################ Best Model: {best_model_id} ################")

            _metrics: Metrics = Metrics.from_dict(study.user_attrs.get("best_metrics"))
            best_model_file_path: str = study.user_attrs.get("best_model_file_path")
            assert _metrics is not None and best_model_file_path is not None

            _model, _save_state = SaveState.load(model_file_path=best_model_file_path)
            _model: AbstractTunableModel
            _save_state: SaveState

            return _model, _save_state, _metrics



def main(
        model_class: Type[AbstractTunableModel],
        train_encodings_file_path: str,
        val_encodings_file_path: str,
        epochs: int = training_config.epochs,
        patience: int = training_config.patience,
        batch_size: int = training_config.batch_size,
        use_weights: bool = training_config.use_weights,
        weight_decay: float = training_config.weight_decay,
        learning_rate: float = training_config.learning_rate,
        learning_rate_decay: float = training_config.learning_rate_decay,
        n_trials: Optional[int] = hyper_param_config.n_trials,
        timeout: Optional[int] = hyper_param_config.n_trials,
        model_save_dir: Optional[str] = None,
        figures_save_dir: Optional[str] = None,
        study_name: Optional[str] = None,
        studies_save_dir: Optional[str] = None,
        log_file_path: Optional[str] = None,
        metrics_file_path: Optional[str] = None
) -> Optional[Tuple[AbstractTunableModel, SaveState, Metrics]]:
    """
    A wrapper for the `tune` function that adds optional logging of stdout.

    This function serves as the main entry point,
    allowing console output to be redirected to a log file during the tuning process.

    :param model_class: The class of the model to be tuned.
    :param train_encodings_file_path: Path to the training data.
    :param val_encodings_file_path: Path to the validation data.
    :param epochs: The maximum number of training epochs per trial.
    :param patience: The patience for early stopping.
    :param batch_size: The batch size for training.
    :param use_weights: Whether to use class weights for the optimizer.
    :param weight_decay: The weight decay for the optimizer.
    :param learning_rate: The learning rate for the optimizer.
    :param learning_rate_decay: The learning rate decay factor.
    :param n_trials: The number of trials to run.
    :param timeout: The time limit for the study in seconds.
    :param model_save_dir: Directory to save the best model.
    :param figures_save_dir: Directory to save performance figures.
    :param study_name: The name for the Optuna study.
    :param studies_save_dir: Directory to save the Optuna study object.
    :param metrics_file_path: Path to save performance metrics for each trial.
    :param log_file_path: If provided, redirects stdout to this file.
    :param metrics_file_path: Path to save performance metrics for each trial.
    :return: A tuple containing the best model, its save state, and its metrics,
             or None if no trial completes successfully.
    """
    context: AbstractContextManager = TeeStdout(filename=log_file_path) if log_file_path is not None else nullcontext()
    with context:  # saving std out to log_file_path for tuning.

        return tune(
            model_class = model_class,
            train_encodings_file_path = train_encodings_file_path,
            val_encodings_file_path = val_encodings_file_path,
            epochs = epochs,
            patience = patience,
            batch_size = batch_size,
            use_weights = use_weights,
            weight_decay = weight_decay,
            learning_rate = learning_rate,
            learning_rate_decay = learning_rate_decay,
            n_trials = n_trials,
            timeout = timeout,
            model_save_dir = model_save_dir,
            figures_save_dir = figures_save_dir,
            study_name = study_name,
            studies_save_dir = studies_save_dir,
            metrics_file_path=metrics_file_path
        )



if __name__ == "__main__": # testing hyperparameter optimization
    from dotenv import load_dotenv
    from source.models.ffn import FastKAN as __ModelClass

    load_dotenv()

    # Get data paths from environment variables
    __encodings_output_path = os.getenv('ENCODINGS_OUTPUT_DIR_LOCAL')
    __train_encodings_file_path = f"{__encodings_output_path}/deeploc_our_train_set.h5"
    __val_encodings_file_path = f"{__encodings_output_path}/deeploc_our_val_set.h5"
    __metrics_file_path = os.getenv("HYPER_PARAM_METRICS_FILE_PATH_LOCAL")  # maybe use a different one for hyper_param training

    # Run the main tuning function with example parameters for debugging
    main(
        model_class = __ModelClass,
        train_encodings_file_path = __train_encodings_file_path,
        val_encodings_file_path = __val_encodings_file_path,
        metrics_file_path = __metrics_file_path,
        epochs = 2,  # For debugging
        n_trials = 2,
        timeout = 10 * 60,  # 10 Minutes
    )