"""
This module provides a robust and memory-efficient data loading pipeline for PyTorch,
designed specifically for handling large datasets stored in HDF5 files.

It leverages PyTorch's `IterableDataset` to stream data on-the-fly,
avoiding the need to load the entire dataset into memory.

Key components include:

- `_BaseIterDataset`: An abstract base class for streaming data.
- `_IterDataset`: Provides shuffled, sequential data streaming for epoch-based training.
- `_IterDatasetRandom`: Provides random data sampling with replacement.
- `DataHandler` classes: Act as factories to create fully configured `DataLoader`instances,
                         and also handle metadata loading and class weight calculation.
- `read_files`: A utility function to simplify the setup of training and validation data handlers.
"""

import torch

from random import Random
from abc import ABC, abstractmethod

from torch.utils.data import IterableDataset, DataLoader
from typing_extensions import Optional, Tuple, override

from source.data_scripts.read_data import Label, num_classes, SequenceEncodingData
from source.data_scripts.encodings import (
    load_metadata_from_hdf5,
    stream_seq_enc_data_from_hdf5,
    stream_seq_enc_data_random_from_hdf5
)

from source.custom_types import (
    Data_T,
    Label_T,
    Encoding_T,
    AttentionMask_T,
    Data_Generator_T,
    Collate_Function_T,
    Sequence_Encoding_Data_Generator_T
)

_seed = 69 # time.time()



class _BaseIterDataset(IterableDataset, ABC):
    """
    An abstract base class for creating iterable datasets from HDF5 files.

    This class streams data on-the-fly,
    making it suitable for large datasets that do not fit into memory.
    Subclasses must implement the `_gen_stream` method to define the data streaming logic.
    """

    def __init__(self,
                 encodings_file_path: str,
                 count: int,
                 buffer_size: int = 32,
                 seed: int = _seed):
        """
        Initializes the base-iterable dataset.

        :param encodings_file_path: Path to the HDF5 file containing the encodings.
        :param count: The total number of items in the dataset.
        :param buffer_size: The size of the buffer for streaming data.
        :param seed: The random seed for reproducibility.
        """
        super().__init__()
        self._count: int = count
        self._encodings_file_path = encodings_file_path
        self._buffer_size: int = buffer_size
        self._rng = Random(seed)


    @staticmethod
    def __prepare_output(seq_enc_data: SequenceEncodingData) -> Data_T:
        """
        Converts a SequenceEncodingData object into a trio of PyTorch tensors.

        :param seq_enc_data: The input data object from the stream.
        :return: A tuple containing the encoding, attention mask, and label tensors.
        """
        x: Encoding_T = torch.from_numpy(seq_enc_data.get_encoding()).float()
        y: Label_T = torch.tensor(seq_enc_data.label.value, dtype=torch.long)  # device=self.device)
        mask: AttentionMask_T = torch.from_numpy(seq_enc_data.get_attention_mask()).bool()
        return x, y, mask


    @abstractmethod
    def _gen_stream(self) -> Sequence_Encoding_Data_Generator_T:
        """
        Abstract method for creating a data streaming generator.
        Must be implemented by subclasses.

        :return: A generator that yields `SequenceEncodingData` objects.
        """
        pass


    @override
    def __iter__(self) -> Data_Generator_T:
        """
        The iterator method that yields processed data samples.

        :return: A generator that yields tuples of (encoding, attention_mask, label) tensors.
        """
        for seq_data in self._gen_stream():
            yield self.__prepare_output(seq_data)


    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return self._count




class _IterDataset(_BaseIterDataset):
    """
    An iterable dataset that streams data sequentially with shuffling for each epoch.

    This implementation ensures that the dataset is shuffled differently for each pass (epoch)
    while maintaining overall reproducibility via the initial seed.
    """

    def __init__(self,
            encodings_file_path: str,
            count: int,
            buffer_size: int = 32,
            seed: int = _seed):
        super().__init__(
            encodings_file_path = encodings_file_path,
            buffer_size = buffer_size,
            count = count,
            seed = seed
        )


    @override
    def _gen_stream(self) -> Sequence_Encoding_Data_Generator_T:
        """
        Creates a new shuffled stream of data for each epoch.

        A unique, but deterministically generated, seed is used for each call
        to ensure that each epoch has a different and reproducible shuffle order.

        :return: A shuffled generator of `SequenceEncodingData` objects.
        """
        # Generate a new, unique seed for each epoch/stream.
        # The overall reproducibility is controlled by the initial seed of `self._rng`.
        # This ensures that each epoch has a different, random, but reproducible shuffle order.
        epoch_seed = self._rng.randint(0, 2 ** 32 - 1)

        yield from stream_seq_enc_data_from_hdf5(
            file_path = self._encodings_file_path,
            shuffle = True,
            seed = epoch_seed
        )



class _IterDatasetRandom(_BaseIterDataset):
    """An iterable dataset that streams data in a random order with replacement."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__bits = 64


    @override
    def _gen_stream(self) -> Sequence_Encoding_Data_Generator_T:
        """
        Creates a stream of data samples in a random order with replacement.

        :return: A randomly sampled generator of `SequenceEncodingData` objects.
        """
        for seq_data in stream_seq_enc_data_random_from_hdf5(
            file_path = self._encodings_file_path,
            seed = self._rng.getrandbits(self.__bits) - (1 << (self.__bits - 1))
        ):
            yield seq_data



class _BaseDataHandler(ABC):
    """
    Abstract base class for managing data loading and preprocessing.

    It handles loading metadata, calculating class weights,
    and serves as a factory for creating `DataLoader` instances.
    """
    def __init__(self,
             encodings_file_path: str,
             batch_size: int,
             buffer_size: int = 32,
             seed: int = _seed):
        """
        Initializes the data handler.

        :param encodings_file_path: Path to the HDF5 file.
        :param batch_size: The number of samples per batch.
        :param buffer_size: The buffer size for data streaming.
        :param seed: The random seed for reproducibility.
        """
        self._encodings_file_path: str = encodings_file_path
        self.encoding_dim, self.total_count, self.label_count = load_metadata_from_hdf5(encodings_file_path)
        self.encoding_dim: int
        self.total_count: int
        self.label_count: dict[Label, int]
        self.class_weights: dict[Label, float] = {  # Weights are frequency relative (inverse frequency)
            label: (self.total_count / count) / num_classes if count > 0 else 0.0
            for label, count in self.label_count.items()
        }

        self._batch_size: int = batch_size
        self._buffer_size: int = buffer_size
        self._seed: int = seed

        # Hardcoded
        self._pin_memory: bool = torch.cuda.is_available()
        self._num_workers: int = 0


    @abstractmethod
    def create_dataloader(self, collate_function: Optional[Collate_Function_T] = None) -> DataLoader:
        """
        Abstract method for creating a DataLoader.

        :param collate_function: An optional custom collate function for the DataLoader.
        :return: A configured `DataLoader` instance.
        """
        pass



class DataHandler(_BaseDataHandler):
    """Standard data handler that creates a `DataLoader` for epoch-based, shuffled training."""

    def __init__(self,
             encodings_file_path: str,
             batch_size: int,
             buffer_size: int = 32,
             seed: int = _seed):
        super().__init__(
            encodings_file_path = encodings_file_path,
            batch_size = batch_size,
            buffer_size = buffer_size,
            seed = seed
        )


    @override
    def create_dataloader(self, collate_function: Optional[Collate_Function_T] = None) -> DataLoader:
        """
        Creates a DataLoader with a standard iterable dataset.

        :param collate_function: Custom function to merge a list of samples to form a mini-batch.
        :return: A `DataLoader` instance.
        """
        dataset = _IterDataset(
            encodings_file_path = self._encodings_file_path,
            count = self.total_count,
            buffer_size= self._buffer_size,
            seed = self._seed,
        )

        return DataLoader(
            dataset = dataset,
            batch_size = self._batch_size,
            shuffle = False,  # Streaming, no shuffle (shuffling is handled in IterDataset)
            collate_fn = collate_function,
            pin_memory = self._pin_memory,
            num_workers = self._num_workers
        )



class DataHandlerRandom(_BaseDataHandler):
    """Data handler that creates a `DataLoader` for random sampling with replacement."""

    def __init__(self,
                 encodings_file_path: str,
                 batch_size: int,
                 buffer_size: int = 32,
                 seed: int = _seed):
        super().__init__(
            encodings_file_path = encodings_file_path,
            batch_size = batch_size,
            buffer_size = buffer_size,
            seed = seed
        )


    @override
    def create_dataloader(self, collate_function: Optional[Collate_Function_T] = None) -> DataLoader:
        """
        Creates a DataLoader with a random-sampling (with replacement) iterable dataset.

        :param collate_function: Custom function to merge a list of samples to form a mini-batch.
        :return: A `DataLoader` instance.
        """
        dataset = _IterDatasetRandom(
            encodings_file_path= self._encodings_file_path,
            count = self.total_count,
            buffer_size= self._buffer_size,
            seed = self._seed,
        )

        return DataLoader(
            dataset = dataset,
            batch_size = self._batch_size,
            shuffle = False,
            collate_fn = collate_function,
            pin_memory = self._pin_memory,
            num_workers = self._num_workers
        )



def read_files(  # because this comes too often.
        train_encodings_file_path: str,
        val_encodings_file_path: str,
        batch_size: int,
        buffer_size: int = 32,
        seed: int = _seed
) -> Tuple[DataHandler, DataHandler, int]:
    """
    A utility function to create training and validation data handlers.

    :param train_encodings_file_path: Path to the training set HDF5 file.
    :param val_encodings_file_path: Path to the validation set HDF5 file.
    :param batch_size: The batch size for both data handlers.
    :param buffer_size: The buffer size for data streaming.
    :param seed: The random seed for reproducibility.
    :return: A tuple containing the training data handler, validation data handler, and the encoding dimension.
    """

    train_data_handler: DataHandler = DataHandler(
        encodings_file_path = train_encodings_file_path,
        batch_size = batch_size,
        buffer_size = buffer_size,
        seed = seed
    )
    val_data_handler: DataHandler = DataHandler(
        encodings_file_path = val_encodings_file_path,
        batch_size = batch_size,
        buffer_size = buffer_size,
        seed = seed
    )

    assert train_data_handler.encoding_dim == val_data_handler.encoding_dim, "Inconsistent encoding dimensions!"
    encoding_dim: int = train_data_handler.encoding_dim

    return train_data_handler, val_data_handler, encoding_dim