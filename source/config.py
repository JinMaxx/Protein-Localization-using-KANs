"""
This module provides a robust and extensible system for managing configurations
for machine learning experiments, loaded from a central YAML file.

It leverages Python's `dataclasses` to create strongly typed,
validated configuration objects for various parts of the application,
such as training, hyperparameter tuning, and specific model architectures.

Key features include:

- YAML-Driven Configuration:
    Defines all parameters in a YAML file,
    supporting environment-specific paths (e.g., local vs. Colab).

- Type-Safe Loading:
    Uses an Enum (`ConfigType`) and dataclasses to parse the YAML into validated Python objects.

- Built-in Validation:
    Each configuration class implements a `validate` method to ensure
    all parameters are coherent and within expected ranges.

- Hyperparameter Tuning Schema:
    Provides a structured way to define hyperparameter search spaces (e.g., ranges, choices)
    directly within the YAML, which seamlessly integrates with the Optuna framework.

- Extensible Design:
    Uses abstract base classes to allow for easy addition of new model configurations
    while enforcing a consistent structure.

- Custom YAML Tags:
    Implements a custom YAML loader to construct complex parameter types
    (e.g., `!Categorical`, `!RangeInt`) during parsing.
"""

import os
import math
import yaml

from enum import Enum
from optuna import Trial
from dotenv import load_dotenv
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, is_dataclass, fields

from typing_extensions import override, Any, Optional, Union, Tuple, List, Set, Literal

from source.training.utils.hidden_layers import HiddenLayers

load_dotenv()

__config_path = os.getenv("CONFIG_PATH_COLAB") if "COLAB_RELEASE_TAG" in os.environ else os.getenv("CONFIG_PATH_LOCAL")
    # else os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.getenv("CONFIG_PATH_LOCAL")))
#        Get the absolute path of a script to join with a relative path. After that normalize



class ConfigType(Enum):
    """
    Enumerates the available configuration sections in the YAML file.
    This allows for type-safe loading of specific configuration blocks.
    """

    Training = "Training"
    HyperParam = "HyperParam"
    Pruning = "Pruning"
    Encodings = "Encodings"
    Metrics = "Metrics"
    Evaluation = "Evaluation"

    MLP = "MLP"
    MLP_PP = "MLP_Per_Protein"
    FastKAN = "FastKAN"

    MaxPoolFastKAN = "MaxPoolFastKAN"
    MaxPoolMLP = "MaxPoolMLP"
    AvgPoolFastKAN = "AvgPoolFastKAN"
    AvgPoolMLP = "AvgPoolMLP"
    LinearFastKAN = "LinearFastKAN"
    LinearMLP = "LinearMLP"
    AttentionFastKAN = "AttentionFastKAN"
    AttentionMLP = "AttentionMLP"
    PositionalFastKAN = "PositionalFastKAN"
    PositionalMLP = "PositionalMLP"
    UNetFastKAN = "UNetFastKAN"
    UNetMLP = "UNetMLP"

    AttentionLstmHybrid = "AttentionLstmHybrid"
    LstmReductionHybrid = "LstmReductionHybrid"
    LightAttention = "LightAttention"
    LightAttentionFastKAN = "LightAttentionFastKAN"



class BaseConfig(ABC):
    """
    Abstract base class for all configuration objects.
    Provides common functionality like validation and a structured string representation.
    """

    @abstractmethod
    def validate(self):
        """
        Abstract method to validate the configuration values.
        Each concrete config class must implement this to ensure its values are sane.
        """
        pass


    def __str__(self) -> str:
        """
        Generates a structured, indented string representation of the config.

        This method recursively traverses the dataclass fields, including any
        nested BaseConfig instances, to build a human-readable summary of
        the entire configuration tree.

        Returns:
            A formatted string representing the configuration.
        """
        def _build_lines(obj, level: int) -> list[str]:
            """Recursive helper to build formatted lines for each config level."""
            lines = []
            indent = "    " * level  # 4 spaces per indentation level

            if not is_dataclass(obj):
                lines.append(f"{indent}{repr(obj)}")
                return lines

            for f in fields(obj):
                key = f.name
                value = getattr(obj, key)

                if isinstance(value, BaseConfig):
                    # For nested configs, show the class and recurse
                    lines.append(f"{indent}{key}: {value.__class__.__name__}")
                    nested_lines = _build_lines(value, level + 1)
                    lines.extend(nested_lines)
                else:
                    # For simple attributes, show the key and value
                    lines.append(f"{indent}{key}: {repr(value)}")
            return lines

        # Start the process with a header for the root config object
        header = f"<{self.__class__.__name__}>"
        config_lines = _build_lines(self, level=1)
        return "\n".join([header] + config_lines)


    def __repr__(self) -> str:
        return self.__str__()



# ----------------- Python Module Config ------------------

@dataclass
class TrainingConfig(BaseConfig):
    """Configuration settings for the model training loop."""

    epochs: int
    patience: int
    batch_size: int
    use_weights: bool
    weight_decay: float
    learning_rate: float
    learning_rate_decay: float

    @override
    def validate(self):
        """Ensures that training parameters are within logical bounds."""
        if self.epochs < 1: raise ValueError("Epochs must be at least 1.")
        if self.patience <= 0: raise ValueError("Patience must be positive.")
        if self.batch_size <= 0: raise ValueError("Batch size must be positive.")
        if not (0.0 <= self.weight_decay <= 1.0): raise ValueError("Weight decay must be between [0.0: 1.0].")
        if not (0.0 < self.learning_rate < 1.0): raise ValueError("Learning rate must be between ]0.0: 1.0[.")
        if not (0.0 < self.learning_rate_decay <= 1.0): raise ValueError("Learning rate decay must be between ]0.0: 1.0].")



@dataclass
class HyperParamConfig(BaseConfig):
    """Configuration for the hyperparameter tuning process."""

    n_trials: int
    timeout: int

    @override
    def validate(self):
        """Ensures that trial and timeout values are positive."""
        if self.n_trials < 1: raise ValueError("Number of trials must be at least 1.")
        if self.timeout < 1: raise ValueError("Timeout must be at least 1.")



@dataclass
class EncodingsConfig(BaseConfig):
    """Configuration for generating sequence encodings."""

    threads: int
    batch_size: int

    @override
    def validate(self):
        """Ensures threads and batch size are positive."""
        if self.threads < 1: raise ValueError("Number of threads must be at least 1.")
        if self.batch_size <= 0: raise ValueError("Batch size must be positive.")



@dataclass
class MetricsConfig(BaseConfig):
    """Configuration for defining weights for a composite evaluation metric."""

    accuracy_weight: int | float
    sensitivity_weight: int | float
    specificity_weight: int | float
    precision_weight: int | float
    f1_score_weight: int | float
    auc_roc_weight: int | float
    auc_pr_weight: int | float

    def __normalize_weights(self):
        """Normalizes all metric weights so that they sum to 1."""
        total_sum = (
            self.accuracy_weight +
            self.sensitivity_weight +
            self.specificity_weight +
            self.precision_weight +
            self.f1_score_weight +
            self.auc_roc_weight +
            self.auc_pr_weight
        )

        if total_sum <= 0:
            raise ValueError("Total weight sum cannot be zero when normalizing.")

        self.accuracy_weight /= total_sum
        self.sensitivity_weight /= total_sum
        self.specificity_weight /= total_sum
        self.precision_weight /= total_sum
        self.f1_score_weight /= total_sum
        self.auc_roc_weight /= total_sum
        self.auc_pr_weight /= total_sum

    @override
    def validate(self):
        """Ensures all weights are non-negative and then normalizes them."""
        if self.accuracy_weight < 0: raise ValueError("Accuracy weight must be non-negative.")
        if self.sensitivity_weight < 0: raise ValueError("Sensitivity weight must be non-negative.")
        if self.specificity_weight < 0: raise ValueError("Specificity weight must be non-negative.")
        if self.precision_weight < 0: raise ValueError("Precision weight must be non-negative.")
        if self.f1_score_weight < 0: raise ValueError("F1 score weight must be non-negative.")
        self.__normalize_weights()



@dataclass
class EvaluationConfig(BaseConfig):
    """Configuration for model evaluation and comparison."""

    model_names: List[str]
    accuracies: List[float]
    accuracies_errors: List[float]

    batch_size: int

    @override
    def validate(self):
        """Configuration for model evaluation and comparison."""
        length: int = len(self.model_names)
        if len(self.accuracies) != length:
            raise ValueError("Number of accuracies does not match number of model names.")
        for accuracy in self.accuracies:
            if not (0.0 <= accuracy <= 1.0): raise ValueError("Accuracy must be between [0.0: 1.0].")
        if len(self.accuracies_errors) != length:
            raise ValueError(f"Number of accuracies errors {len(self.accuracies_errors)} does not match number of model names.")
        for accuracy_error in self.accuracies_errors:
            if math.isnan(accuracy_error): continue
            if not (0.0 <= accuracy_error <= 1.0):
                raise ValueError(f"Accuracy error {accuracy_error} must be between [0.0: 1.0].")



# -------------------- Test Parameter --------------------

@dataclass
class ParamInfo:
    """Holds metadata for a hyperparameter, used for plotting."""
    type: Literal['linear', 'category']
    range: Optional[Tuple[Union[int, float], Union[int, float]]] = None
    step: Optional[Union[int, float]] = None
    categories: Optional[List[Union[str, int, float]]] = field(default_factory=list)



@dataclass
class AbstractTestParameter(BaseConfig, ABC):
    """Abstract base class for a hyperparameter that can be tuned."""
    pass

    @abstractmethod
    def trial_parameter(self, name: str, trial: Trial) -> Any:
        """Returns the value of the hyperparameter for a trial."""
        pass

    @abstractmethod
    def get_info(self, name: str) -> dict[str, ParamInfo]:
        """Returns a dictionary mapping parameter names to their plot axis configuration."""
        pass




@dataclass
class CategoricalParameter(AbstractTestParameter):
    """Represents a hyperparameter chosen from a predefined list of values."""

    choices: List[Union[str, int, float]]

    @override
    def validate(self):
        """Validates that choices is a non-empty list."""
        if not isinstance(self.choices, list) or not self.choices:
            raise ValueError("`choices` must be a non-empty list.")


    @override
    def trial_parameter(self, name: str, trial: Trial) -> Union[str, int, float]:
        """Suggests a categorical value to Optuna."""
        return trial.suggest_categorical(
            name = name,
            choices = self.choices
        )

    @override
    def get_info(self, name: str) -> dict[str, ParamInfo]:
        """For categorical parameters, the axis should treat choices as distinct categories."""
        return {name: ParamInfo(
            type = 'category',
            categories = self.choices
        )}


@dataclass
class RangeParameterInt(AbstractTestParameter):
    """Represents a hyperparameter sampled from a discrete integer range."""

    min: int
    max: int
    step: int = 1


    @override
    def validate(self):
        """Validates the integer range and step values."""
        if not all(isinstance(i, int) for i in [self.min, self.max, self.step]):
            raise TypeError("min, max, and step for RangeParameterInt must be integers.")
        if self.max < self.min:
            raise ValueError(f"max ({self.max}) must be greater than or equal to min ({self.min}).")
        if self.step <= 0:
            if not (self.step == 0 and (self.max - self.min) == 0):  # no interval just one value.
                raise ValueError(f"step ({self.step}) must be a positive integer.")
        elif (self.max - self.min) % self.step != 0:
            raise ValueError(f"The range (max - min) ({self.max - self.min}) must be divisible by step ({self.step}).")


    @override
    def trial_parameter(self, name: str, trial: Trial) -> int:
        """Suggests an integer value to Optuna."""
        return trial.suggest_int(
            name = name,
            low = self.min,
            high = self.max,
            step = self.step
        )

    @override
    def get_info(self, name: str) -> dict[str, ParamInfo]:
        """Returns info for a linear integer range."""
        return {name: ParamInfo(
            type = 'linear',
            range = (self.min, self.max),
            step = self.step
        )}



@dataclass
class RangeParameterFloat(AbstractTestParameter):
    """Represents a hyperparameter sampled from a continuous float range."""

    min: float | int
    max: float | int
    step: Optional[float | int] = None


    def __post_init__(self):
        """
        Ensures all numeric values are floats after initialization.
        Converts min, max, and step to percentages if they are integers.
        """
        if isinstance(self.min, int): self.min = self.min / 100.0
        if isinstance(self.max, int): self.max = self.max / 100.0
        if self.step is not None and isinstance(self.step, int):
            self.step = self.step / 100.0


    @override
    def validate(self):
        """Validates the float range and step values."""
        if not all(isinstance(variable, float) for variable in [self.min, self.max]):
            raise TypeError("min and max for RangeParameterFloat must be floats.")
        if self.max < self.min:
            raise ValueError(f"max ({self.max}) must be greater than or equal to min ({self.min}).")
        if self.step is not None:
            if self.step <= 0.0:
                if (math.isclose(self.step, 0.0, abs_tol=1e-6) and math.isclose((self.max - self.min), 0.0, abs_tol=1e-6)):
                    raise ValueError(f"step ({self.step}) must be positive.")
            else: # Check if the range is divisible by the step
                range_val = self.max - self.min
                remainder = range_val % self.step
                if not (math.isclose(remainder, 0.0, abs_tol=1e-6) or math.isclose(remainder, self.step, abs_tol=1e-6)):
                    raise ValueError(f"The range (max - min) ({range_val}) must be divisible by step ({self.step}). Remainder: {remainder}")


    @override
    def trial_parameter(self, name: str, trial: Trial) -> float:
        """Suggests a float value to Optuna."""
        return trial.suggest_float(
            name = name,
            low = self.min,
            high = self.max,
            step = self.step
        )


    @override
    def get_info(self, name: str) -> dict[str, ParamInfo]:
        """Returns info for a linear float range."""
        return {name: ParamInfo(
            type = 'linear',
            range = (self.min, self.max),
            step = self.step
        )}



@dataclass
class HiddenLayersParameter(AbstractTestParameter):
    """
    Represents the search space for creating relative hidden layer configurations.
    This parameter defines the ranges for the number of layers and their size percentage.
    """
    # Search space for the number of layers (e.g., 2 to 5 layers)
    num_layers: RangeParameterInt

    # Search space for the relative layer size percentage (e.g., 0.1 to 0.8)
    layer_size_percentage: RangeParameterFloat

    @override
    def validate(self):
        """Validates contained range parameters and their constraints."""
        self.num_layers.validate()
        self.layer_size_percentage.validate()

        # Add constraints specific to layer percentages
        if not (0.0 < self.layer_size_percentage.min and self.layer_size_percentage.max <= 1.0):
             raise ValueError("Layer size percentage search range must be within (0.0, 1.0].")


    @override
    def trial_parameter(self, name: str, trial: Trial) -> HiddenLayers:
        """Suggests values for layer count and size, returning a HiddenLayers object."""
        # Note: The 'name' parameter is ignored here because we need to suggest two separate sub-parameters for the trial.
        return HiddenLayers(kind=HiddenLayers.Type.RELATIVE, value=(
            self.layer_size_percentage.trial_parameter(name="layer_size_percentage", trial=trial),
            self.num_layers.trial_parameter(name="num_layers", trial=trial)
        ))


    @override
    def get_info(self, name: str) -> dict[str, ParamInfo]:
        """Aggregates plot info from sub-parameters."""
        info = {}
        info.update(self.num_layers.get_info(f"num_layers"))
        info.update(self.layer_size_percentage.get_info(f"layer_size_percentage"))
        return info


# -------------------- ML Model Config --------------------

@dataclass
class AbstractModelConfig(BaseConfig, ABC):
    """Abstract base class for model configurations."""


@dataclass
class AbstractTunableModelConfig(AbstractModelConfig, ABC):
    """Abstract base class for model configurations that have tunable hyperparameters."""

    test_parameters: dict[str, AbstractTestParameter]

    @property
    @abstractmethod
    def expected_test_parameters(self) -> Set[str]:
        """
        A set of strings representing the required keys in the 'test_parameters' dictionary.
        Each subclass must implement this to declare its specific hyperparameter schema.
        Returns an empty set by default for models that don't have testable parameters.
        """
        return set()


    @override
    def validate(self):
        """Validates that the provided test parameters match the expected schema."""
        # --- 1. Validate the schema of test_parameters ---
        expected_keys = self.expected_test_parameters
        actual_keys = set(self.test_parameters.keys())

        if expected_keys != actual_keys:
            missing = expected_keys - actual_keys
            extra = actual_keys - expected_keys
            error_msg = f"Mismatched test parameters for {self.__class__.__name__}."
            if missing:
                error_msg += f" Missing required parameters: {missing}."
            if extra:
                error_msg += f" Found unexpected parameters: {extra}."
            raise ValueError(error_msg)

        # --- 2. Validate the values of each provided test parameter ---
        for name, param in self.test_parameters.items():
            if not hasattr(param, 'validate'):
                raise TypeError(f"Parameter '{name}' is not a valid test parameter object.")
            param.validate()


    def trial_parameters(self, trial: Trial) -> dict[str, Any]:
        """Generates a dictionary of concrete hyperparameter values for an Optuna trial."""
        return {name: param.trial_parameter(name, trial) for name, param in self.test_parameters.items()}


    def test_parameters_info(self) -> List[Tuple[str, ParamInfo]]:
        """Merges ParamInfo from test-parameters flattened."""
        aggregated_info: List[Tuple[str, ParamInfo]] = []
        for param_name, param in self.test_parameters.items():
            for adjusted_param_name, param_info in param.get_info(f"{param_name}").items():
                aggregated_info.append((adjusted_param_name, param_info))
        return aggregated_info



@dataclass
class AbstractSequenceModelConfig(AbstractTunableModelConfig, ABC):
    """Base class for models that process sequence data."""

    in_seq_len: int


    @override
    def validate(self):
        super().validate()
        if self.in_seq_len < 1: raise ValueError("Max input sequence length must be bigger than 0.")



@dataclass
class AbstractFFNConfig(AbstractTunableModelConfig, ABC):
    """Base class for feed-forward network configurations."""

    in_seq_len: int

    hidden_layer_relative: Optional[int|float]
    num_hidden_layers:     Optional[int]
    hidden_layer_exact:    Optional[List[int]]


    @property
    @override
    def expected_test_parameters(self) -> Set[str]:
        """FFN models are expected to be able to tune their hidden layers."""
        return super().expected_test_parameters.union({'hidden_layers'})



    @override
    def validate(self):
        super().validate()

        if self.in_seq_len < 1: raise ValueError("Max input sequence length must be bigger than 0.")

        self.hidden_layers: HiddenLayers
        if self.hidden_layer_relative is not None:
            if isinstance(self.hidden_layer_relative, float):
                if not 0.0 <= self.hidden_layer_relative <= 1.0:
                    raise ValueError("Hidden layer reduction size must be between [0.0: 1.0].")
            elif isinstance(self.hidden_layer_relative, int):
                if not 1 <= self.hidden_layer_relative <= 100:
                    raise ValueError("Hidden layer reduction size must be between [1: 100].")
                self.hidden_layer_relative: float = self.hidden_layer_relative / 100.0
            else: raise ValueError("Hidden layer reduction size must be a float or an integer.")
            if self.num_hidden_layers is None: raise ValueError("Number of hidden layers must be specified.")
            if self.num_hidden_layers < 0: raise ValueError("Number of hidden layers must be positive")
            # noinspection PyAttributeOutsideInit
            self.hidden_layers = HiddenLayers(HiddenLayers.Type.RELATIVE, (self.hidden_layer_relative, self.num_hidden_layers))
        elif self.hidden_layer_exact is not None:
            for hidden_layer_exact in self.hidden_layer_exact:
                if  0 >= hidden_layer_exact: raise ValueError("Exact hidden layer size must be bigger than 0.")
            # noinspection PyAttributeOutsideInit
            self.hidden_layers = HiddenLayers(HiddenLayers.Type.EXACT, self.hidden_layer_exact)
        else: raise ValueError("Either hidden layer reduction size or hidden layer exact sizes must be specified.")



@dataclass
class AbstractReducedFFNConfig(AbstractFFNConfig, ABC):
    """Base class for FFNs that operate on a reduced-length sequence."""

    reduced_seq_len: int


    @property
    @override
    def expected_test_parameters(self) -> Set[str]:
        """Requires 'hidden_layers' and adds 'reduced_seq_len' as a tunable parameter."""
        return super().expected_test_parameters.union({'reduced_seq_len'})


    @override
    def validate(self):
        super().validate()
        if self.reduced_seq_len <= 0: raise ValueError("Reduced sequence length must be positive.")



@dataclass
class AbstractReducedWChannelsFFNConfig(AbstractReducedFFNConfig, ABC):
    """Base class for FFNs with reduced sequence length and channels."""

    reduced_channels: int


    @property
    @override
    def expected_test_parameters(self) -> Set[str]:
        """Requires parent parameters and adds 'reduced_channels' as tunable."""
        return super().expected_test_parameters.union({'reduced_channels'})


    @override
    def validate(self):
        super().validate()
        if self.reduced_channels <= 0: raise ValueError("Reduced channels must be positive.")



@dataclass
class AbstractKANConfig(AbstractTunableModelConfig, ABC):
    """Mixin class for KAN-specific configurations and validation."""
    grid_diff: float
    num_grids: int

    @property
    @override
    def expected_test_parameters(self) -> Set[str]:
        """Adds KAN-specific tunable parameters."""
        return super().expected_test_parameters.union({'grid_diff', 'num_grids'})

    @override
    def validate(self):
        """Validates KAN-specific parameters."""
        super().validate()
        if self.grid_diff < 0.1: raise ValueError("Grid diff must be at least 0.1.")
        if self.num_grids < 1: raise ValueError("Number of grids must be at least 1.")



@dataclass
class FastKANConfig(AbstractKANConfig, AbstractFFNConfig):
    """A KAN model based on a standard FFN input."""
    pass



@dataclass
class MLPConfig(AbstractFFNConfig):
    pass



@dataclass
class MLPPPConfig(AbstractFFNConfig):
    pass



@dataclass
class ReducedFastKANConfig(AbstractKANConfig, AbstractReducedFFNConfig, ABC):
    """A KAN model that operates on a reduced sequence length."""
    pass



@dataclass
class ReducedWChannelsFastKANConfig(AbstractKANConfig, AbstractReducedWChannelsFFNConfig, ABC):
    """A KAN model that operates on a reduced sequence length and channels."""
    pass



@dataclass
class MaxPoolFastKANConfig(ReducedFastKANConfig):
    pass



@dataclass
class MaxPoolMLPConfig(AbstractReducedFFNConfig):
    pass



@dataclass
class AvgPoolFastKANConfig(ReducedFastKANConfig):
    pass



@dataclass
class AvgPoolMLPConfig(AbstractReducedFFNConfig):
    pass



@dataclass
class LinearFastKANConfig(ReducedFastKANConfig):
    pass



@dataclass
class LinearMLPConfig(AbstractReducedFFNConfig):
    pass



@dataclass
class AttentionFastKANConfig(ReducedFastKANConfig):
    pass



@dataclass
class AttentionMLPConfig(AbstractReducedFFNConfig):
    pass



@dataclass
class PositionalFastKANConfig(ReducedWChannelsFastKANConfig):
    pass



@dataclass
class PositionalMLPConfig(AbstractReducedWChannelsFFNConfig):
    pass



@dataclass
class UNetFastKANConfig(ReducedWChannelsFastKANConfig):
    pass



@dataclass
class UNetMLPConfig(AbstractReducedWChannelsFFNConfig):
    pass



@dataclass
class AttentionLstmHybridConfig(AbstractFFNConfig):

    attention_num_heads: int
    lstm_hidden_size: int
    lstm_num_layers: int


    @property
    @override
    def expected_test_parameters(self) -> Set[str]:
        return super().expected_test_parameters.union({'attention_num_heads', 'lstm_hidden_size', 'lstm_num_layers'})


    @override
    def validate(self) -> None:
        super().validate()

        if self.attention_num_heads <= 0:
            raise ValueError(f"attention_heads must be a positive integer, but got {self.attention_num_heads}")
        if self.lstm_hidden_size <= 0:
            raise ValueError(f"lstm_hidden_size must be a positive integer, but got {self.lstm_hidden_size}")
        if self.lstm_num_layers <= 0:
            raise ValueError(f"lstm_num_layers must be a positive integer, but got {self.lstm_num_layers}")



@dataclass
class LstmReductionHybridConfig(AbstractReducedFFNConfig):

    lstm_hidden_size: int
    lstm_num_layers: int
    dropout_rate: float


    @property
    @override
    def expected_test_parameters(self) -> Set[str]:
        return super().expected_test_parameters.union({'lstm_hidden_size', 'lstm_num_layers', 'dropout_rate'})


    @override
    def validate(self) -> None:
        super().validate()

        if self.lstm_hidden_size <= 0:
            raise ValueError(f"lstm_hidden_size must be a positive integer, but got {self.lstm_hidden_size}")
        if self.lstm_num_layers <= 0:
            raise ValueError(f"lstm_num_layers must be a positive integer, but got {self.lstm_num_layers}")
        if not (0.0 <= self.dropout_rate < 1.0):
            raise ValueError(f"dropout_rate must be in the range [0.0, 1.0), but got {self.dropout_rate}")



@dataclass
class LightAttentionConfig(AbstractFFNConfig):

    kernel_size: int
    conv_dropout_rate: float
    ffn_dropout_rate: float  # Try [0.1; 0.5]


    @property
    @override
    def expected_test_parameters(self) -> Set[str]:
        return super().expected_test_parameters.union({'kernel_size', 'conv_dropout_rate'})


    @override
    def validate(self):
        super().validate()
        if self.kernel_size <= 0:
            raise ValueError(f"kernel_size must be a positive integer, but got {self.kernel_size}")
        if not (0.0 <= self.conv_dropout_rate < 1.0):
            raise ValueError(f"conv_dropout_rate must be in the range [0.0, 1.0), but got {self.conv_dropout_rate}")
        if not (0.0 <= self.ffn_dropout_rate < 1.0):
            raise ValueError(f"ffn_dropout_rate must be in the range [0.0, 1.0), but got {self.ffn_dropout_rate}")


@dataclass
class LightAttentionFastKANConfig(LightAttentionConfig, AbstractKANConfig):
    pass


# ---------------------------------------------------------

class ConfigLoader(yaml.SafeLoader):
    """A custom YAML loader to construct our specific dataclasses from tags."""
    pass

def _constructor_factory(cls):
    """A factory to create constructor functions for our dataclasses."""
    def constructor(loader: yaml.Loader, node: yaml.Node) -> Any:
        if isinstance(node, yaml.MappingNode):
            return cls(**loader.construct_mapping(node, deep=True))
        elif isinstance(node, yaml.SequenceNode):
            return cls(*loader.construct_sequence(node, deep=True))
        elif isinstance(node, yaml.ScalarNode):
            return cls(loader.construct_scalar(node))
        else:
            raise TypeError(f"Unsupported YAML node type for constructing '{cls.__name__}': {type(node).__name__}")
    return constructor

# Register constructors for custom YAML tags
ConfigLoader.add_constructor('!Categorical', _constructor_factory(CategoricalParameter))
ConfigLoader.add_constructor('!RangeInt', _constructor_factory(RangeParameterInt))
ConfigLoader.add_constructor('!RangeFloat', _constructor_factory(RangeParameterFloat))
ConfigLoader.add_constructor('!HiddenLayers', _constructor_factory(HiddenLayersParameter))


def __load_from_yaml(config_path: str = __config_path) -> dict:
    """Load the configuration file into a dictionary."""
    try:
        with open(config_path, "r") as file: return yaml.load(file, Loader=ConfigLoader)
    except FileNotFoundError: raise ValueError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e: raise ValueError(f"Error parsing YAML file: {e}")


def parse_config(config_type: ConfigType, config_path: str = __config_path):
    """
    Parses a specific section of the YAML config file into a validated dataclass object.

    Args:
        config_type: The enum member specifying which configuration to load.
        config_path: The path to the YAML configuration file.

    Returns:
        A validated instance of the corresponding BaseConfig subclass.

    Raises:
        ValueError: If the config section is not found or if parsing/validation fails.
    """

    yaml_dict = __load_from_yaml(config_path)

    if config_type.value not in yaml_dict:
        raise ValueError(f"Config section '{config_type.value}' not found in {config_path}.")

    config: BaseConfig

    # noinspection PyUnreachableCode
    match config_type:
        case ConfigType.Training: config = TrainingConfig(**yaml_dict[ConfigType.Training.value])
        case ConfigType.HyperParam: config = HyperParamConfig(**yaml_dict[ConfigType.HyperParam.value])
        case ConfigType.Encodings: config = EncodingsConfig(**yaml_dict[ConfigType.Encodings.value])
        case ConfigType.Metrics: config = MetricsConfig(**yaml_dict[ConfigType.Metrics.value])
        case ConfigType.Evaluation: config = EvaluationConfig(**yaml_dict[ConfigType.Evaluation.value])

        case ConfigType.MLP: config = MLPConfig(**yaml_dict[ConfigType.MLP.value])
        case ConfigType.MLP_PP: config = MLPPPConfig(**yaml_dict[ConfigType.MLP_PP.value])
        case ConfigType.FastKAN: config = FastKANConfig(**yaml_dict[ConfigType.FastKAN.value])

        case ConfigType.MaxPoolFastKAN: config = MaxPoolFastKANConfig(**yaml_dict[ConfigType.MaxPoolFastKAN.value])
        case ConfigType.MaxPoolMLP: config = MaxPoolMLPConfig(**yaml_dict[ConfigType.MaxPoolMLP.value])
        case ConfigType.AvgPoolFastKAN: config = AvgPoolFastKANConfig(**yaml_dict[ConfigType.AvgPoolFastKAN.value])
        case ConfigType.AvgPoolMLP: config = AvgPoolMLPConfig(**yaml_dict[ConfigType.AvgPoolMLP.value])
        case ConfigType.LinearFastKAN: config = LinearFastKANConfig(**yaml_dict[ConfigType.LinearFastKAN.value])
        case ConfigType.LinearMLP: config = LinearMLPConfig(**yaml_dict[ConfigType.LinearMLP.value])
        case ConfigType.AttentionFastKAN: config = AttentionFastKANConfig(**yaml_dict[ConfigType.AttentionFastKAN.value])
        case ConfigType.AttentionMLP: config = AttentionMLPConfig(**yaml_dict[ConfigType.AttentionMLP.value])
        case ConfigType.PositionalFastKAN: config = PositionalFastKANConfig(**yaml_dict[ConfigType.PositionalFastKAN.value])
        case ConfigType.PositionalMLP: config = PositionalMLPConfig(**yaml_dict[ConfigType.PositionalMLP.value])
        case ConfigType.UNetFastKAN: config = UNetFastKANConfig(**yaml_dict[ConfigType.UNetFastKAN.value])
        case ConfigType.UNetMLP: config = UNetMLPConfig(**yaml_dict[ConfigType.UNetMLP.value])

        case ConfigType.AttentionLstmHybrid: config = AttentionLstmHybridConfig(**yaml_dict[ConfigType.AttentionLstmHybrid.value])
        case ConfigType.LstmReductionHybrid: config = LstmReductionHybridConfig(**yaml_dict[ConfigType.LstmReductionHybrid.value])
        case ConfigType.LightAttention: config = LightAttentionConfig(**yaml_dict[ConfigType.LightAttention.value])
        case ConfigType.LightAttentionFastKAN: config = LightAttentionFastKANConfig(**yaml_dict[ConfigType.LightAttentionFastKAN.value])

        case _: raise ValueError(f"Invalid config type: {config_type}")

    config.validate()
    return config

    # I could do this...
    # # Dynamically find the correct config class based on the enum.
    #     config_class_name = config_type.value.replace("_", "") + "Config"
    #     try:
    #         # We need to capitalize 'PP' for MLP_Per_Protein
    #         if config_type == ConfigType.MLP_PP:
    #             config_class_name = "MLPPPConfig"
    #         config_class = globals()[config_class_name]
    #     except KeyError:
    #         raise ValueError(f"Could not find a corresponding config class for '{config_type.value}'")
    #
    #     # Instantiate the dataclass. PyYAML has already built all the nested objects.
    #     try:
    #         config = config_class(**config_data)
    #     except TypeError as e:
    #         raise ValueError(f"Failed to instantiate {config_class.__name__}. Check YAML keys match dataclass fields. Error: {e}") from e
    #
    #     config.validate()
    #     return config
