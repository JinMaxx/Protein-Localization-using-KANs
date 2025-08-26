"""
This module defines the abstract base classes for all models in this project,
establishing a common interface and providing shared functionality.

- `AbstractModel`:
    The foundational abstract class for any model,
    providing core functionality for saving, loading, identification, and introspection.

- `AbstractTunableModel`:
    Extends `AbstractModel` to add a framework for hyperparameter tuning using Optuna,
    linking models to their configuration files.

- `AbstractSequenceModel`:
    A specialization of `AbstractTunableModel` for models that operate on sequence data.

The `_create_test_model_class` factory function is a key component of the tuning
framework, allowing for the dynamic creation of model variants for each Optuna trial.
"""

import os
import uuid
import inspect
import importlib

import torch
import torch.nn as nn
from torch import device

from optuna import Trial
from abc import ABC, abstractmethod
from typing_extensions import cast, override, Type, Any, List, Tuple, Optional, Self

from source.data_scripts.read_data import num_classes
from source.config import (
    AbstractModelConfig,
    AbstractTunableModelConfig,
    AbstractSequenceModelConfig,
    parse_config, ConfigType
)

from source.custom_types import (
    Encodings_Batch_T,
    AttentionMask_Batch_T,
    Data_T,
    Labels_Batch_T,
    Logits_Batch_T
)


# "Any problem in computer science can be solved with another layer of abstraction.
#  But that usually will create another problem." - Wheeler, David J. (slightly paraphrased)



class AbstractModel(nn.Module, ABC):
    """
    An abstract base class for all models, providing a common interface and
    core functionalities like saving, loading, and versioning.
    """

    @abstractmethod
    def __init__(self, in_channels: int, **_kwargs):
        """
        Initializes the model.

        :param in_channels: The number of input features for the model.
        :param _kwargs: Catches unused parameters from subclasses.
        """
        super().__init__()
        self.in_channels: int = in_channels
        self.uuid: str = f"{str(uuid.uuid4())}"
        self.device: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    @classmethod
    def name(cls):
        """Returns the name of the model class. Can be overridden by subclasses."""
        return cls.__name__


    def id(self):
        """Returns a unique identifier for the model instance."""
        return f"{self.name()}_{self.uuid}"
    

    def up_version(self):
        """
        Updates the model's UUID to create a new version.

        This prepends a new UUID to the existing one, creating a traceable chain
        of model versions, which is useful when fine-tuning a pre-trained model.
        """
        print(f"Changing model id:\n{self.id()}")
        self.uuid = f"{str(uuid.uuid4())}_{self.uuid}" # this will keep a chain of uuids for all versions.
        print(f"to:\n{self.id()}")


    def __str__(self) -> str:
        return f"{self.name()} | encoding_dim={self.in_channels}, memory_size={self.memory_size():.2f}MB"


    @classmethod
    @abstractmethod
    def config_type(cls) -> ConfigType:
        """Specifies the `ConfigType` enum for this model to load its configuration."""
        raise NotImplementedError


    @classmethod
    def get_config(cls, force_reload: bool = False) -> AbstractModelConfig:
        """Retrieves and caches the parsed configuration for this model class."""
        if force_reload or not hasattr(cls, '_config_cache'):
            cls._config_cache: AbstractModelConfig = parse_config(cls.config_type())
        return cls._config_cache


    @abstractmethod
    def forward(self, x: Encodings_Batch_T, attention_mask: AttentionMask_Batch_T) -> Logits_Batch_T:
        """Defines the forward pass of the model."""
        raise NotImplementedError


    @abstractmethod
    def collate_function(self, batch: List[Data_T]) -> tuple[Encodings_Batch_T, Labels_Batch_T, AttentionMask_Batch_T]:
        """Processes a batch of data into tensors for the model."""
        raise NotImplementedError


    def _get_init_params(self) -> dict[str, Any]:
        """
        Introspects the `__init__` method to get its parameters and their current values.
        This is a key part of the robust save/load mechanism.
        Model Parameters should be saved publicly with the same name.

        :return: A dictionary mapping `__init__` parameter names to their values.
        """
        params = inspect.signature(self.__init__).parameters  # Get __init__ parameter info
        param_names = list(params.keys())
        # Return a dictionary of parameter names and their current values
        return {name: getattr(self, name) for name in param_names}


    def save(self, save_dir: str, identifier: Optional[str], **kwargs: Any) -> str:
        """
        Saves the model to a file.

        The saved file contains everything needed to reconstruct the model,
        including `__init__` parameters, the state dictionary, and class/module information.

        :param save_dir: The directory where the model will be saved.
        :param identifier: A unique name for the saved file (e.g., "best_model" or "epoch_10").
                           If None, the model's `id()` is used.
        :param kwargs: Additional keyword arguments to save alongside the model (e.g., a `SaveState` object).
        :return: The full path to the saved model file.
        """
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f"{identifier if identifier is not None else self.id()}.pth")
        torch.save({
            'init_params': self._get_init_params(),  # Infer the ffn_layer’s __init__ parameters and their current values
            'state_dict': self.state_dict(),  # Save the ffn_layer weights
            'uuid': self.uuid,
            'class_name': self.__class__.__name__,
            'module': self.__class__.__module__,
            'kwargs': kwargs
        }, file_path)
        print(f"Model saved: {file_path}")
        return file_path


    @staticmethod
    def load(file_path: str) -> Tuple['AbstractModel', dict[str, Any]]:
        """
        Loads a model from a file.

        Dynamically imports the correct model class,
        reconstructs it using the saved `__init__` parameters,
        and loads the saved weights.

        Unable to load models that were saved with a different version of the code.

        :param file_path: The path to the saved model `.pth` file.
        :return: A tuple containing the loaded model instance and any additional
                 keyword arguments saved with it.
        """
        checkpoint = torch.load(file_path, weights_only=False, map_location=None if torch.cuda.is_available() else torch.device('cpu'))
        module = importlib.import_module(checkpoint['module'])
        cls = getattr(module, checkpoint['class_name'])
        init_params = checkpoint['init_params']
        print(f"Init params: {init_params}")
        model = cls(**init_params)
        # print(f"model.device: {model.device}")
        model.load_state_dict(checkpoint['state_dict'])
        model.uuid = checkpoint['uuid']
        kwargs = checkpoint.get('kwargs', {})
        print(f"Loaded Model: {model}")
        return model, kwargs


    def memory_size(self) -> float:
        """Calculates and returns the model's memory footprint in megabytes."""
        if self.device.type == "cuda":
            params_size = sum(param.nelement() * param.element_size() for param in self.parameters()) if hasattr(self, 'parameters') else 0
            buffers_size = sum(buffer.nelement() * buffer.element_size() for buffer in self.buffers()) if hasattr(self, 'buffers') else 0
            size = params_size + buffers_size
        else:
            size = sum(
                tensor.nelement() * tensor.element_size()
                for tensor in self.state_dict().values()
            )
        # print(f"Total size: {size} bytes")
        size_mb = size / 1024 / 1024
        return size_mb


    # Overwrite: Add figures to this collection after that and return them.
    # Note: Cant use typing here or else this ends up in a dependency circle.
    # noinspection PyMethodMayBeStatic, PyUnusedLocal
    def add_figures_collection(self, figures) -> List:
        """A placeholder method for collecting model figures for analysis."""
        return []



class AbstractTunableModel(AbstractModel, ABC):
    """
    An abstract base class for models that support hyperparameter tuning.
    It links the model to its configuration and provides methods for generating
    trial-specific model variants.
    """

    def __init__(self, in_channels: int, **_kwargs):
        super().__init__(in_channels=in_channels, **_kwargs)


    @classmethod
    def create_test_model_class(cls, trial: Trial, **kwargs) -> Type[Self]:
        """
        Generates a new model class with hyperparameters suggested by an Optuna trial.

        This method leverages the `test_parameters` defined in the model's configuration
        to create a dictionary of trial-specific hyperparameters.

        :param trial: The Optuna trial object which is used to suggest hyperparameters.
        :param kwargs: Additional fixed parameters to be passed to the model's constructor.
        :return: A new, temporary class definition with the trial's hyperparameters "baked in".
        """
        trial_params = cls.get_config().trial_parameters(trial)

        # The _create_test_model_class factory bakes these parameters into a new class
        return _create_test_model_class(cls, **kwargs, **trial_params)


    @classmethod
    @override
    def get_config(cls, force_reload: bool = False) -> AbstractTunableModelConfig:
        return cast(AbstractTunableModelConfig, super().get_config())



# Generates a subclass dynamically during runtime.
def _create_test_model_class(model_class: Type[AbstractTunableModel], **kwargs) -> Type[AbstractTunableModel]:
    """
    A factory that dynamically generates a model subclass for an Optuna trial.

    This function creates a new class that inherits from `model_class`.
    This new class has the trial-specific hyperparameters (passed via `**kwargs`)
    baked into its `__init__` method.
    This approach allows Optuna to create different model variants
    without polluting the original class definition.

    :param model_class: The base tunable model class.
    :param kwargs: The set of hyperparameters for this specific trial. (Model parameters to be exact)
    :return: A new `TestModel` class definition.
    """
    assert isinstance(model_class, type), f"model_class: {model_class} must be a class!"
    assert issubclass(model_class, AbstractTunableModel), f"model_class: {model_class} is not a subclass of AbstractTunableModel!"
    print(f"TestModel: {model_class.name()} | kwargs={kwargs}")


    class TestModel(model_class):
        """A temporary model variant for a single hyperparameter trial."""

        def __init__(self, in_channels: int):
            # Pass both trial-specific kwargs and fixed params to the parent constructor.
            super().__init__(
                in_channels = in_channels,  # This is set in train_models when data is read.
                **kwargs  # Setting all specific class-related parameters here.
            )

        @classmethod
        @override
        def name(cls): return f"{cls.__name__}_{model_class.name()}"

        @override
        def _get_init_params(self) -> dict[str, Any]:
            # Introspect the parent's __init__ to get the correct parameters for saving.
            # Need to overwrite the class in inspect.signature(...)
            params = inspect.signature(model_class.__init__).parameters
            param_names = [name for name in params.keys() if name != "self"]
            # Combine params from signature with the baked-in kwargs
            return {name: getattr(self, name) for name in param_names}

        @override
        def save(self, save_dir: str, identifier: Optional[str], **_kwargs: Any):
            # When saving a TestModel, save it as an instance of its parent class.
            # This ensures that AbstractModel.load reconstructs the original,
            # permanent class, not the temporary TestModel.
            os.makedirs(save_dir, exist_ok=True)
            file_path = os.path.join(save_dir, f"{identifier if identifier is not None else super().id()}.pth")
            torch.save({
                'init_params': self._get_init_params(),
                'state_dict': self.state_dict(),
                'uuid': self.uuid,
                'class_name': model_class.__name__,  # use parents class
                'module': model_class.__module__,    # and module instead.
                'kwargs': _kwargs
            }, file_path)
            print(f"TestModel saved as parent: {file_path}")
            return file_path

        # Implement abstract methods by delegating to the parent implementation.

        @classmethod
        @override
        def config_type(cls) -> ConfigType:
            return super().config_type()

        @override
        def forward(self, x: Encodings_Batch_T, attention_mask: AttentionMask_Batch_T) -> Logits_Batch_T:
            return super().forward(x=x, attention_mask=attention_mask)

        @override
        def collate_function(self, batch: List[Data_T]) -> tuple[Encodings_Batch_T, Labels_Batch_T, AttentionMask_Batch_T]:
            return super().collate_function(batch)


    return cast(Type[model_class], TestModel)



class AbstractSequenceModel(AbstractTunableModel, ABC):
    """An abstract base class for sequence-based models."""

    def __init__(self,
                 in_channels: int,
                 in_seq_len: int,
                 out_channels: Optional[int] = None,
                 **_kwargs):
        """
        Initializes the sequence model.

        :param in_channels: The number of features per element in the sequence.
        :param in_seq_len: The fixed sequence length for padding/truncation.
        :param out_channels: The number of output classes. Defaults to `num_classes` from config.
        """
        super().__init__(in_channels=in_channels, **_kwargs)
        self.in_seq_len: int = in_seq_len  # The common property of sequence_models
        self.out_channels: int = out_channels if out_channels is not None else num_classes


    @classmethod
    @override
    def create_test_model_class(cls, trial: Trial, **kwargs) -> Type['AbstractSequenceModel']:
        """
        Creates a trial-specific sequence model class.

        It retrieves the required `in_seq_len` from the model's configuration
        and passes it as a fixed parameter to the constructor.
        """
        config: AbstractSequenceModelConfig = cast(AbstractSequenceModelConfig, cls.get_config())
        in_seq_len: int = config.in_seq_len  # This is not part of the test parameters but needs to be passed to init.
        return cast(type[AbstractSequenceModel], super().create_test_model_class(trial, in_seq_len=in_seq_len, **kwargs))