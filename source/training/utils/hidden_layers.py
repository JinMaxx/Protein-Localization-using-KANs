"""
This module provides a flexible configuration class, `HiddenLayers`,
for defining the architecture of hidden layers in a neural network.

It supports two primary methods for specifying layer sizes:

1. `EXACT`:
    Defines the exact integer size for each hidden layer.

2. `RELATIVE`:
    Defines layers based on a fractional reduction of the previous layer's size,
    providing a way to scale the network architecture dynamically.

This approach, inspired by Rust's enum patterns,
combines the definition and validation logic into a single, robust class,
ensuring that the network architecture is always well-defined.
"""

from enum import Enum
from typing_extensions import List, Tuple, cast
from source.data_scripts.read_data import num_classes



# Rust Like Enum
# Just so I have different options on how to define the sizes of my hidden layers.
class HiddenLayers:
    """
    A configuration class for defining the sizes of hidden layers in a network.

    This class encapsulates the logic for creating a list of hidden layer dimensions
    based on a specified strategy ('kind') and its corresponding value.
    It ensures that the provided configuration is valid upon initialization.
    """

    class Type(Enum):
        """Specifies the method for defining hidden layer sizes."""
        EXACT = 1
        RELATIVE = 3


    def __init__(self, kind: 'HiddenLayers.Type', value: List[int | float] | Tuple[float, int]):
        """
        Initializes the HiddenLayers configuration.

        :param kind: The method to use for defining layers (EXACT or RELATIVE).
        :param value: The configuration value, which depends on the `kind`:
                      - For `EXACT`, a list of integers (e.g., [512, 256, 128]).
                      - For `RELATIVE`, a tuple containing a float (reduction factor)
                        and an int (number of layers), (e.g., (0.5, 3)).
        :raises TypeError: If the type of `value` does not match the `kind`.
        :raises ValueError: If the contents of `value` are invalid for the `kind`.
        """
        self.kind: HiddenLayers.Type = kind

        self.value: List[int | float] | Tuple[float, int]

        # Validate value based on `kind`
        if self.kind == HiddenLayers.Type.EXACT:
            if not isinstance(value, list):
                raise TypeError(f"EXACT type requires value to be a list, got {type(value).__name__}")
            if not all(isinstance(v, int) for v in value):
                raise ValueError(f"EXACT type requires all elements in the list to be integers. Got: {value}")
            self.value = value

        elif self.kind == HiddenLayers.Type.RELATIVE:
            if not isinstance(value, tuple):
                raise TypeError(f"RELATIVE type requires value to be a tuple, got {type(value).__name__}")
            if len(value) != 2 or not (isinstance(value[0], float) and isinstance(value[1], int)):
                raise ValueError(f"RELATIVE type requires a tuple of (float, int). Got: {value}")
            self.value = value

        # else: raise ValueError(f"Invalid hidden layers type: {self.kind}")


    def calculate_layers(self, input_dim: int) -> list[int]:
        """
        Calculates the concrete list of hidden layer sizes based on the input dimension.

        :param input_dim: The size of the input layer, used as the starting point for
                          RELATIVE calculations.
        :return: A list of integers representing the size of each hidden layer.
        :raises ValueError: If a calculated layer size is too small to be meaningful
                            (less than twice the number of output classes), which helps
                            prevent invalid architectures during hyperparameter tuning.
        """
        hidden_layers: list[int]

        match self.kind:

            # Exact sizes of layers. (e.g. [525_000, 100_000, 20_000, 1_000, 100])
            case HiddenLayers.Type.EXACT:
                hidden_layers = cast(List[int], self.value)


            # Reduction relative to the previous layer size.
            # E.g., in_channels = 1023*1024 = 1047552: value = (0.2, 4)
            # -> [1047552*0.2=209510, 209510*0.2=41902, 41902*0.2=8380, 8380*0.5=1676]
            case HiddenLayers.Type.RELATIVE:
                # hidden_layers = [max(1, int(in_channels * percentage)) for percentage in self.value]

                hidden_layers = []
                current_layer_size = input_dim
                percentage, num_layers = self.value

                for layer in range(num_layers):
                    next_layer_size = max(1, int(current_layer_size * percentage))
                    hidden_layers.append(next_layer_size)
                    if next_layer_size <= 1: break
                    current_layer_size = next_layer_size

                # Throwing errors rather than continuing with the invalid layers or removing them.
                # Hyper Parameter search could get stuck in a false local minimum if layers were to be removed.
                for layer in hidden_layers:
                    if layer < 2 * num_classes:  # Meaningless layers!
                        raise ValueError(
                            f"One or more layer sizes in {hidden_layers} are less than 2 * num_classes ({2 * num_classes}).")
                        # hidden_layers.remove(layer)

            # case _: raise ValueError(f"Invalid hidden layers type: {self.kind}")


        # print(f"Exact hidden layers: {hidden_layers}")
        return hidden_layers


    def __str__(self) -> str:
        return f"HiddenLayers(kind={self.kind}, value={self.value})"


    def __repr__(self) -> str:
        return str(self)


    def __len__(self) -> int:
        """Returns the number of hidden layers defined by this configuration."""
        match self.kind:
            case HiddenLayers.Type.EXACT: return len(self.value)
            # case HiddenLayers.Type.REDUCTION: return len(self.value)
            case HiddenLayers.Type.RELATIVE: return self.value[1]
            # case _: raise ValueError(f"Invalid hidden layers type: {self.kind}")