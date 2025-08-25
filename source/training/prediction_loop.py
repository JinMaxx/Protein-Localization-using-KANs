"""
This module provides a comprehensive framework for training and evaluating PyTorch models.
It orchestrates the entire training lifecycle, from data loading to model saving,
and is designed to be both flexible and robust.

Key functionalities include:
- A core training loop (`__train`) with support for early stopping, learning rate
  scheduling, and class-weighted loss.
- High-level functions to either train a new model from scratch (`train_new`) or
  continue training from a saved checkpoint (`train_continue`).
- Integration with `Metrics` and `TrainingFiguresCollection` to track performance,
  log results, and generate visualizations of the training process.
- A main entry point (`main`) that intelligently dispatches to the correct training
  function based on the provided arguments.
- Error handling and context management for clean setup and teardown, including
  optional logging to a file.
"""

import torch
import warnings

from abc import ABC
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from typing_extensions import List, Dict, Any, override

from source.models.abstract import AbstractModel
from source.training.data_handling import _BaseDataHandler

from source.custom_types import (
    Labels_Batch_T,
    Logits_Batch_T,
    Encodings_Batch_T,
    AttentionMask_Batch_T,
    Prediction_Generator_T,
    Batch_Losses_T,
    Data_Loader_T
)



# template method pattern
class PredictionLoop(ABC):
    """
    An abstract base class that defines a template for prediction loops.

    This class provides a structured way to iterate over a dataset,
    make predictions with a model, and calculate losses.
    It is designed to be subclassed for specific tasks like training or validation.
    The core logic resides in the `prediction_generator` method,
    which can be customized by overriding hook methods (`on_start`, `process_loss`, `on_end`).
    """

    def __init__(self,
            model: AbstractModel,
            data_handler: _BaseDataHandler,
            use_weights: bool = False):
        """
        Initializes the PredictionLoop.

        :param model: The model to use for predictions.
        :param data_handler: The data handler that provides the dataset.
        :param use_weights: If True, applies class weights to the loss function to handle imbalanced datasets.
        """
        self.model: AbstractModel = model.to(model.device)

        self.data_handler: _BaseDataHandler = data_handler
        self.data_loader: Data_Loader_T = data_handler.create_dataloader(collate_function=model.collate_function)

        weights_tensor: torch.Tensor | None = None
        if use_weights:
            # Ensure weights are ordered correctly according to class labels
            weights: List[float] = [data_handler.class_weights[label] for label in sorted(data_handler.class_weights.keys(), key=lambda l: l.value)]
            weights_tensor = torch.tensor(weights, dtype=torch.float32).to(model.device)
        
        self.criterion = CrossEntropyLoss(weight=weights_tensor, reduction='none').to(model.device)
    

    def on_start(self) -> None:
        """Optional: Logic to execute before the loop starts."""
        pass


    def process_loss(self, losses: torch.Tensor) -> None:
        """Optional: Logic to execute after a batch is calculated."""
        pass


    def on_end(self) -> None:
        """Optional: Logic to execute after the loop ends."""
        pass


    def additional_progress_bar_info(self) -> Dict[str, Any] | None:
        """
        Optional: Adding additional information to the progress bar.

        :return: A dictionary of key-value pairs to display, or None.
        """
        pass


    def prediction_generator(self) -> Prediction_Generator_T:
        """
        A generator that yields predictions and losses for each sample in the dataset.

        It iterates through the data loader, performs a forward pass, calculates the loss,
        and handles potential NaN values. It also manages a progress bar with relevant metrics.

        :yields: A tuple containing the true label, model logits, and the calculated loss for a single sample.
        """
        self.on_start()
        with tqdm(self.data_loader, desc=self.__class__.__name__, total=self.data_handler.total_count, leave=False) as progress_bar:
            for encodings, labels, attention_masks in progress_bar:
                encodings: Encodings_Batch_T = encodings.to(self.model.device)
                labels: Labels_Batch_T = labels.to(self.model.device)
                attention_masks: AttentionMask_Batch_T = attention_masks.to(self.model.device)

                # Forward pass in mixed precision
                with torch.amp.autocast(device_type=str(self.model.device.type)):
                    logits: Logits_Batch_T = self.model(encodings, attention_mask=attention_masks)
                    losses: Batch_Losses_T = self.criterion(logits, labels)  # shape:[valid_batch_size]

                batch_size: int = len(encodings)

                # NaN Handling: Checks each sample's loss for NaN.
                valid_mask = ~torch.isnan(losses)  # A NaN loss usually comes from unstable logits (inf or NaN).
                valid_batch_size: int = valid_mask.sum().detach().cpu().item()

                if not valid_mask.all():  # If any NaNs are found
                    labels = labels[valid_mask]
                    logits = logits[valid_mask]
                    losses = losses[valid_mask]

                if valid_batch_size > 0:  # contains valid losses

                    batch_loss = losses.sum()

                    self.process_loss(batch_loss)

                    predicted: torch.Tensor = torch.argmax(logits, dim=1)
                    batch_correct: int = (predicted == labels).sum().detach().cpu().item()

                    progress_bar.update(batch_size)
                    progress_bar.set_postfix(
                        loss = batch_loss.detach().cpu().item() / valid_batch_size,
                        accuracy = batch_correct / valid_batch_size,
                        **(self.additional_progress_bar_info() or {})
                    )

                    # Yield detached tensors for metric calculation
                    for label, logit, loss in zip(labels.detach().cpu(), logits.detach().cpu().float(), losses.detach().cpu().float()):
                        yield label, logit, loss.item()

                else:
                    progress_bar.update(batch_size)
                    progress_bar.set_postfix(
                        loss = "NaN",
                        accuracy = "NaN",
                        **(self.additional_progress_bar_info() or {})
                    )

                if not valid_mask.all():  # This is here, so that the problematic progressbar string before gets frozen.
                    print(f"\nWarning: Filtering {batch_size - valid_batch_size} NaN losses found in a batch.")

        self.on_end()



class Trainer(PredictionLoop):
    """
    Implements the prediction loop for model training.

    This class extends `PredictionLoop` and adds training-specific logic, including optimizer steps,
    backpropagation with gradient scaling for mixed precision and learning rate scheduling.
    """

    def __init__(self,
            model: AbstractModel,
            data_handler: _BaseDataHandler,
            use_weights: bool,
            weight_decay: float,
            learning_rate: float,
            learning_rate_decay: float
    ):
        """
        Initializes the Trainer.

        :param model: The model to be trained.
        :param data_handler: The data handler for the training set.
        :param use_weights: If True, applies class weights to the loss.
        :param weight_decay: The weight decay (L2 penalty) for the AdamW optimizer.
        :param learning_rate: The initial learning rate for the optimizer.
        :param learning_rate_decay: The decay factor for the exponential learning rate scheduler.
        """
        super().__init__(
            model = model,
            data_handler = data_handler,
            use_weights = use_weights
        )

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # If learning_rate_decay is 1.0, it means no decay -> skip creating the scheduler
        self.scheduler = None
        if learning_rate_decay < 1.0:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=learning_rate_decay)

        self.scaler = torch.amp.GradScaler(device=str(self.model.device.type))


    @override
    def on_start(self):
        """Sets the model to training mode."""
        self.model.train()


    @override
    def process_loss(self, loss):
        """Performs backpropagation and updates model weights."""
        self.optimizer.zero_grad(set_to_none=True)  # reduces memory fragmentation
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()


    @override
    def on_end(self):
        """Steps the learning rate scheduler and clears the CUDA cache."""
        if self.scheduler is not None: self.scheduler.step()  # Update learning rate
        torch.cuda.empty_cache()


    @override
    def additional_progress_bar_info(self):
        """Returns the current learning rate to display on the progress bar."""
        return {"learning_rate": self.optimizer.param_groups[0]['lr']}



class Validator(PredictionLoop):
    """
    Implements the prediction loop for model validation.

    This class extends `PredictionLoop` for evaluating the model on a validation set.
    It ensures the model is in evaluation mode and that no gradients are computed.
    """

    def __init__(self,
            model: AbstractModel,
            data_handler: _BaseDataHandler):
        """
        Initializes the Validator.

        :param model: The model to be validated.
        :param data_handler: The data handler for the validation set.
        """
        super().__init__(model=model, data_handler=data_handler)


    @override
    def on_start(self):
        """Sets the model to evaluation mode."""
        self.model.eval()


    @override
    def prediction_generator(self) -> Prediction_Generator_T:
        """
        Wraps the base generator in `torch.no_grad()` to disable gradient calculation during validation,
        improving performance, and reducing memory usage.
        """
        with torch.no_grad():
            with warnings.catch_warnings():  # filter warnings
                warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True. Gradients will be None")
                yield from super().prediction_generator()