#!/usr/bin/python3
"""
This module provides a framework for generating and managing protein sequence encodings.

It defines a hierarchy of encoding models, including one-hot encoding and several
transformer-based embedding models like ESM, ProtT5, and ANKH.

Key functionalities include:
- A flexible, abstract base class (`AbstractEncodingModel`) for defining encoding models.
- Concrete implementations for various state-of-the-art protein language models.
- An efficient data pipeline for processing FASTA files, generating encodings in batches,
  and saving them to HDF5 files.
- Asynchronous processing capabilities to generate encodings for multiple files concurrently.
- Utility functions for reading and streaming encoding data and metadata from HDF5 files.
"""

import os
from typing import Union, List, Dict

import h5py
import numpy
import torch
import asyncio
import numpy as np

from Bio.Seq import Seq
from random import Random
from threading import Lock
from itertools import islice
from tqdm.asyncio import tqdm
from Bio.SeqIO import SeqRecord
from abc import ABC, abstractmethod
from typing_extensions import override, Optional
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, EsmModel, T5EncoderModel, T5Tokenizer, PreTrainedTokenizer

from source.data_scripts.read_data import SequenceData, SequenceEncodingData, Label, read_fasta, list_fasta_files
from source.config import EncodingsConfig, ConfigType, parse_config

from source.custom_types import (
    Seq_Data_Generator_T,
    Sequence_Encoding_Data_Generator_T
)

config: EncodingsConfig = parse_config(ConfigType.Encodings)


# For this project to generate embeddings, the code is copied and adapted from:
# https://colab.research.google.com/github/tsenoner/protspace/blob/main/examples/notebook/ClickThrough_GenerateEmbeddings.ipynb
# a lot has been changed, but at the core it follows the same logic.



class AbstractEncodingModel(ABC):
    """
    An abstract base class for protein sequence encoding models.

    This class uses a factory pattern (`setup_model`) to instantiate
    and cache different encoding models, preventing redundant initializations.
    It ensures that all concrete implementations provide an `encode` method.

    :param encoding_dim: The dimensionality of the output encoding vector.
    """

    __setup_lock = Lock()

    # model_name -> model_cache
    __per_model_cache: dict[str, "AbstractEncodingModel"] = {}  # This will store the cached ffn_layer, tokenizer, and other info


    def __init__(self, encoding_dim: int):
        self.encoding_dim: int = encoding_dim


    @abstractmethod
    def encode(self, seq_data_batch: list[SequenceData]) -> Sequence_Encoding_Data_Generator_T:
        """
        Encodes a batch of sequence data.

        :param seq_data_batch: A list of `SequenceData` objects to be encoded.
        :return: A generator yielding `SequenceEncodingData` objects.
        """
        raise NotImplementedError


    @staticmethod
    def clear_model_cache():
        """Clears the model cache, releasing models from memory."""
        AbstractEncodingModel.__per_model_cache.clear()


    @staticmethod
    def setup_model(model_name: str) -> "AbstractEncodingModel":
        """
        Factory method to get or create an encoding model instance.

        This method checks a cache for an existing model instance.
        If not found, it creates a new one, caches it, and returns it.

        :param model_name: The name of the model to set up (e.g., 'onehot', 'esm2_t33_650M_UR50D').
        :return: An instance of a concrete `AbstractEncodingModel` subclass.
        :raises ValueError: If the `model_name` is unknown.
        """
        with AbstractEncodingModel.__setup_lock:

            # Check if the ffn_layer is already set up and cached
            if model_name in AbstractEncodingModel.__per_model_cache:
                return AbstractEncodingModel.__per_model_cache[model_name]

            # If not cached, proceed with setup
            print(f"\nSetting up ffn_layer {model_name}")

            encoding_model: AbstractEncodingModel

            if model_name == "onehot":
                encoding_model = OneHotEncodingModel()

            else:  # Embedding
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if device.type == "cuda": print("Using CUDA...")
                elif device.type == "cpu": print("CUDA not available, using CPU. Performance may be slower")

                if "esm" in model_name: encoding_model = ESM(model_name, device)
                elif "ankh" in model_name: encoding_model = ANKH(model_name, device)
                elif "prot_t5" in model_name: encoding_model = ProtT5(model_name, device)
                else: raise ValueError(f"Unknown model name: {model_name}")

            AbstractEncodingModel.__per_model_cache[model_name] = encoding_model  # Cache setup

            return AbstractEncodingModel.__per_model_cache[model_name]



class OneHotEncodingModel(AbstractEncodingModel):
    """A model for generating one-hot encodings of amino acid sequences."""

    AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

    def __init__(self):
        super().__init__(encoding_dim=len(OneHotEncodingModel.AMINO_ACIDS))


    @override
    def encode(self, seq_data_batch: list[SequenceData]) -> Sequence_Encoding_Data_Generator_T:
        """
        Generates one-hot encodings for a batch of sequences.

        :param seq_data_batch: A list of `SequenceData` objects.
        :return: A generator yielding `SequenceEncodingData` objects with one-hot encodings.
        """
        for seq_data in seq_data_batch:
            sequence_str = str(seq_data.record.seq)
            onehot: np.ndarray = self.__encode_sequence(sequence_str)
            mask = np.ones(len(sequence_str), dtype=bool)  # For one-hot, all tokens are real, so the mask is all True.
            yield SequenceEncodingData.of(seq_data, onehot, mask)



    def __encode_sequence(self, sequence: str, alphabet: str = AMINO_ACIDS) -> np.ndarray:
        """
        Converts a single amino acid sequence string into a one-hot encoded numpy array.

        :param sequence: The amino acid sequence string.
        :return: A 2D numpy array representing the one-hot encoding.
        """
        char_to_index = {char: idx for idx, char in enumerate(alphabet)}
        one_hot = np.zeros((len(sequence), self.encoding_dim), dtype=np.float32)
        for i, char in enumerate(sequence):
            if char in char_to_index: one_hot[i, char_to_index[char]] = 1.0
            # else: remain zeros
        return one_hot



class AbstractEmbeddingModel(AbstractEncodingModel):
    """
    An abstract base for transformer-based embedding models.

    This class implements the Template Method Pattern for encoding.
    The `encode` method defines the overall algorithm,
    while subclasses must implement `_extract_embedding`
    and can optionally override `_preprocess` to handle model-specific logic.

    :param model: The pre-trained PyTorch model.
    :param tokenizer: The corresponding tokenizer.
    :param device: The torch device (`cpu` or `cuda`) to run the model on.
    :param encoding_dim: The dimensionality of the model's embeddings.
    """

    def __init__(self,
            model: torch.nn.Module,
            tokenizer: PreTrainedTokenizer,
            device: torch.device,
            encoding_dim: int):
        super().__init__(encoding_dim=encoding_dim)
        self.model: torch.nn.Module = model.to(device)
        self.model.eval()
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.device: torch.device = device


    @override  # template method pattern
    def encode(self, seq_data_batch: list[SequenceData]) -> Sequence_Encoding_Data_Generator_T:
        """
        Template method to generate embeddings for a batch of sequences.

        :param seq_data_batch: A list of `SequenceData` objects.
        :return: A generator yielding `SequenceEncodingData` objects with embeddings.
        """
        # HOOK: Preprocessing
        self._preprocess(seq_data_batch)

        # Tokenization
        inputs = self.tokenizer(
            [str(seq_data.record.seq) for seq_data in seq_data_batch],
            return_tensors="pt",
            max_length=self.encoding_dim,
            truncation=True,
            padding=True,
            add_special_tokens=True,
        )

        attention_masks = inputs['attention_mask'].cpu()
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs).last_hidden_state.cpu()

            # HOOK: Output adaptation
            for i, seq_data in enumerate(seq_data_batch):
                embedding, mask = self._extract_embedding_and_mask(outputs[i], attention_masks[i])
                yield SequenceEncodingData.of(seq_data, embedding.numpy(), mask.numpy())


    #### Hooks ####

    def _preprocess(self, seq_data_batch: list[SequenceData]) -> None:
        """
        Hook for any model-specific preprocessing. Can be overridden by subclasses.

        :param seq_data_batch: A list of `SequenceData` objects to be preprocessed.
                               Modifications should be made in-place.
        """
        pass


    @abstractmethod
    def _extract_embedding_and_mask(self, output: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Hook to extract the correct embedding and attention mask from the model's output.

        :param output: The output tensor for a single sequence from the transformer model.
        :param mask: The attention mask for a single sequence from the tokenizer.
        :return: A tuple of (extracted embedding, extracted mask).
        """
        raise NotImplementedError("Subclasses must implement _extract_embedding_and_mask.")



class ProtT5(AbstractEmbeddingModel):
    """Embedding model for the ProtT5-XL-U50 protein language model."""

    def __init__(self, model_name: str, device: torch.device):
        super().__init__(
            tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False),
            model = T5EncoderModel.from_pretrained(model_name, torch_dtype=torch.float16),
            device = device,
            encoding_dim = 1024
        )
        self.model.half()


    @override
    def _preprocess(self, seq_data_batch: list[SequenceData]) -> None:
        """Adds spaces between amino acids as required by ProtT5."""
        for sequence_data in seq_data_batch:
            sequence_data.record.seq = Seq(" ".join(str(sequence_data.record.seq)))


    @override
    def _extract_embedding_and_mask(self, output: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extracts the embedding and mask, removing the special end-of-sequence token."""
        # The mask from the tokenizer includes all tokens, including special ones.
        true_length = mask.sum()  # True length by summing the mask (where 1 is a real token).
        # ProtT5 adds one special token </s> at the end. We slice to remove it.
        embedding = output[:true_length - 1]
        new_mask = mask[:true_length - 1]
        return embedding, new_mask



class ESM(AbstractEmbeddingModel):
    """Embedding model for the ESM (Evolutionary Scale Modeling) protein language models."""

    def __init__(self, model_name: str, device: torch.device):
        super().__init__(
            tokenizer = AutoTokenizer.from_pretrained(model_name),
            model = EsmModel.from_pretrained(model_name),
            device = device,
            encoding_dim = 4000
        )

    
    @override
    def _extract_embedding_and_mask(self, output: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extracts the embedding and mask, removing the special <cls> and <sep> tokens."""
        true_length = mask.sum() 
        # ESM adds <cls> at the start and <sep> at the end. We slice it.
        embedding = output[1:true_length - 1]
        new_mask = mask[1:true_length - 1]
        return embedding, new_mask



class ANKH(AbstractEmbeddingModel):
    """Embedding model for the ANKH protein language model."""

    def __init__(self, model_name: str, device: torch.device):
        super().__init__(
            tokenizer = AutoTokenizer.from_pretrained(model_name),
            model = T5EncoderModel.from_pretrained(model_name),
            device = device,
            encoding_dim = 4000
        )


    @override
    def _extract_embedding_and_mask(self, output: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extracts the embedding and mask, removing the special end-of-sequence token."""
        true_length = mask.sum()
        # Like ProtT5, ANKH adds a special token at the end.
        embedding = output[:true_length - 1]
        new_mask = mask[:true_length - 1]
        return embedding, new_mask



def __write_hdf5(seq_data_enc_generator: Sequence_Encoding_Data_Generator_T, encoding_dim: int, file_path: str) -> None:
    """
    Saves sequence encodings to an HDF5 file with a structured hierarchy.

    :param seq_data_enc_generator: A generator of `SequenceEncodingData` objects.
    :param encoding_dim: The dimensionality of the encodings.
    :param file_path: Path to the output HDF5 file.
    :return: None. Encodings are written to the provided file in a clean hierarchical structure.
    """
    with h5py.File(file_path, "w") as hdf5_file:

        data_group = hdf5_file.create_group("data")
        metadata_group = hdf5_file.create_group("metadata")

        seq_data_counter: int = 0
        label_counts = {label: 0 for label in Label}

        for seq_data_index, seq_enc_data in enumerate(seq_data_enc_generator):
            seq_group = data_group.create_group(f"encoding_{seq_data_index}")

            seq_group.attrs["id"] = seq_enc_data.record.id
            seq_group.attrs["description"] = seq_enc_data.record.description
            seq_group.attrs["sequence"] = str(seq_enc_data.record.seq)
            seq_group.attrs["label"] = seq_enc_data.label.value
            seq_group.create_dataset("encoding", data=seq_enc_data.get_encoding(), compression="gzip")
            seq_group.create_dataset("attention_mask", data=seq_enc_data.get_attention_mask(), compression="gzip")

            seq_data_counter += 1
            label_counts[seq_enc_data.label] += 1

        if seq_data_counter < 1: raise ValueError("The dataset is empty.")

        metadata_group.attrs["encoding_dim"] = encoding_dim
        metadata_group.attrs["count"] = seq_data_counter
        label_counts_group = metadata_group.create_group("label_counts")
        for label, count in label_counts.items():  # Save counts as attributes
            label_counts_group.attrs[str(label.value)] = count



def __load_metadata_from_hdf5(file_path: str) -> tuple[int, int, Dict[Label, int]|None]:
    """
    Loads only the metadata from an HDF5 encoding file.

    :param file_path: Path to the HDF5 file.
    :return: A tuple containing (encoding_dim, total_count, label_counts).
    :raises FileNotFoundError: If the file does not exist.
    :raises ValueError: If the HDF5 file is missing required metadata.
    """
    if not os.path.exists(file_path): raise FileNotFoundError(f"File does not exist '{file_path}'.")

    with h5py.File(file_path, "r") as hdf5_file:

        metadata_group = hdf5_file.get("metadata")
        if metadata_group is None: raise ValueError("Metadata group not found in HDF5 file.")

        if "encoding_dim" not in metadata_group.attrs: raise ValueError("Encoding dimension not found in metadata group.")
        encoding_dim: int = int(metadata_group.attrs["encoding_dim"])

        count: int = int(metadata_group.attrs.get("count", 0))

        label_count: dict[Label, int]|None = {}
        if "label_counts" in metadata_group:  # Check if the group exists
            label_counts_group = metadata_group["label_counts"]
            for label_value, count_value in label_counts_group.attrs.items():
                _label: Label = Label(int(label_value))  # Get enum by value
                label_count[_label] = int(count_value)
            label_count = {label: label_count[label] for label in sorted(label_count.keys(), key=lambda l: l.value)}
        else: label_count = None

    return encoding_dim, count, label_count



def load_metadata_from_hdf5(file_path: Union[str, List[str]]) -> tuple[int, int, Dict[Label, int] | None]:
    """
    Loads and combines metadata from one or more HDF5 encoding files.

    It aggregates the total count and label counts.
    Verifies that all files share the same 'encoding_dim'.

    :param file_path: A single file path or a list of file paths.
    :return: A tuple containing the verified (encoding_dim, combined_total_count, combined_label_counts).
    :raises FileNotFoundError: If any file does not exist.
    :raises ValueError: If the file list is empty, if metadata is missing,
                        or if 'encoding_dim' is inconsistent across files.
    """
    paths = [file_path] if isinstance(file_path, str) else file_path
    if not paths: raise ValueError("Cannot load metadata from an empty list of files.")

    first_file_path = paths[0]
    encoding_dim, count, label_count = __load_metadata_from_hdf5(first_file_path)

    for file_path in paths[1:]:
        _encoding_dim, _count, _label_count = __load_metadata_from_hdf5(file_path)

        # Verify that the encoding dimension is consistent
        if _encoding_dim != encoding_dim:
            raise ValueError(
                f"Inconsistent encoding dimension in file '{file_path}'. "
                f"Expected {encoding_dim}, but got {_encoding_dim}."
            )

        count += _count

        if _label_count:
            for __label, __num in _label_count.items():
                label_count[__label] = label_count.get(__label, 0) + __num

    label_count = {label: label_count[label] for label in sorted(label_count.keys(), key=lambda l: l.value)}

    return encoding_dim, count, label_count



def __parse_seq_group(seq_group) -> SequenceEncodingData:
    """
    Parses an HDF5 group into a `SequenceEncodingData` object.

    :param seq_group: An `h5py.Group` object from an HDF5 file,
                      representing a single sequence's data.
    :return: A `SequenceEncodingData` object populated with data from the group.
    """

    seq_id: str = seq_group.attrs["id"]
    description: str = seq_group.attrs.get("description", "")
    sequence: Seq = Seq(seq_group.attrs["sequence"])
    label: Label = Label(int(seq_group.attrs["label"]))
    encoding: numpy.ndarray = seq_group["encoding"][:]
    attention_mask: numpy.ndarray = seq_group["attention_mask"][:]

    return SequenceEncodingData(
        seq_record = SeqRecord(
            seq = sequence,
            id = seq_id,
            description = description
        ),
        label = label,
        encoding = encoding,
        attention_mask = attention_mask
    )



def __stream_seq_enc_data_from_hdf5(
        file_path: str,
        shuffle: bool = False,
        seed: Optional[int] = None
) -> Sequence_Encoding_Data_Generator_T:
    """
    Streams sequence data from an HDF5 file, with optional shuffling.

    :param file_path: Path to the HDF5 file.
    :param shuffle: If True, shuffles the data before streaming. Defaults to False.
    :param seed: Seed for reproducible shuffling. Ignored if shuffle is False.
    :return: A generator yielding `SequenceEncodingData` objects.
    :raises FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File does not exist '{file_path}'.")

    try:
        with h5py.File(file_path, "r") as hdf5_file:
            data_group = hdf5_file.get("data")
            if not data_group: return  # yield nothing if data_group is empty or does not exist
            seq_keys = list(data_group.keys())
            if shuffle:  # Randomness: shuffle the list of keys in place.
                rng = Random(seed)
                rng.shuffle(seq_keys)
            for seq_key in seq_keys: yield __parse_seq_group(data_group[seq_key])
    except Exception as e:
        print(f"An error occurred while reading the HDF5 file: {e}")
        raise


def stream_seq_enc_data_from_hdf5(
        file_path: Union[str, List[str]],
        shuffle: bool = False,
        seed: Optional[int] = None
) -> Sequence_Encoding_Data_Generator_T:
    """
    Streams sequence data from one or more HDF5 files in a memory-efficient manner.

    This function handles both a single file path and a list of paths.
    If multiple files are provided, they are merged into a single stream and stream sequentially.

    :param file_path: A single file path or a list of file paths.
    :param shuffle: If True, shuffles the data using a streaming approach.
    :param seed: A seed for reproducible shuffling.
    :return: A generator yielding `SequenceEncodingData` objects.
    """
    for path in [file_path] if isinstance(file_path, str) else file_path:
        yield from __stream_seq_enc_data_from_hdf5(path, shuffle=shuffle, seed=seed)


def stream_seq_enc_data_random_from_hdf5(
        file_path: str,
        seed: int = 42,
        count: Optional[int] = None
) -> Sequence_Encoding_Data_Generator_T:
    """
    Streams a specified number of random samples (with replacement) from an HDF5 file.

    :param file_path: Path to the HDF5 file.
    :param seed: Random seed for reproducibility.
    :param count: The number of samples to yield. If None, yields the total number of items in the file.
    :return: A generator yielding `SequenceEncodingData` objects.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File does not exist '{file_path}'.")

    _, _count, _ = load_metadata_from_hdf5(file_path)
    count = count or _count
    if count < 1: raise ValueError("count must be greater than or equal to 1.")

    rng = Random(seed)

    with h5py.File(file_path, "r") as hdf5_file:
        data_group = hdf5_file.get("data")
        if not data_group: raise ValueError("The dataset is empty.")

        seq_keys = list(data_group)
        for _ in range(0, count):
            # Randomly choose an index with replacing
            seq_key = seq_keys[rng.randint(0, count - 1)]
            yield __parse_seq_group(data_group[seq_key])



def generate_encodings_of_fasta(
        fasta_file_path: str,
        output_file_path: str,
        model_name: str,
        batch_size: int = config.batch_size,
        disable_tqdm: bool = False) -> None:
    """
    Generates and saves encodings for a single FASTA file.

    :param fasta_file_path: Path to the input FASTA file.
    :param output_file_path: Path for the output HDF5 file.
    :param model_name: Name of the encoding model to use.
    :param batch_size: Number of sequences to process in each batch.
    :param disable_tqdm: If True, the progress bar is disabled.
    """
    print(f"Processing {os.path.basename(fasta_file_path)}")
    seq_data_generator: Seq_Data_Generator_T = read_fasta(fasta_file_path)
    assert seq_data_generator, "No sequences found in the input file"

    encoding_model: AbstractEncodingModel = AbstractEncodingModel.setup_model(model_name)

    def seq_data_batch_generator() -> Sequence_Encoding_Data_Generator_T:
        stream = iter(seq_data_generator)  # Working with streams
        with tqdm(desc=f"{os.path.basename(fasta_file_path)}", disable=disable_tqdm, leave=False, unit="seq") as progress_bar:
            while batch := list(islice(stream, batch_size)):  # Collect the next `__batch_size` elements from the stream
                for seq_enc_data in encoding_model.encode(batch): yield seq_enc_data
                progress_bar.update(len(batch))

    __write_hdf5(seq_data_batch_generator(), encoding_model.encoding_dim, output_file_path)
    print(f"\nEncodings saved to {model_name}/{os.path.basename(output_file_path)}")



async def generate_encodings_of_all_fasta(
        input_dir: str,  # I could change that to a list of file_paths
        output_dir: str,
        model_name: str,
        threads: int = config.threads,
        batch_size: int = config.batch_size) -> None:
    """
    Concurrently generates encodings for all FASTA files in a directory.

    This function uses a thread pool to process files concurrently.
    Each .fasta file is processed into encodings in parallel, increasing performance.
    However, concurrency increases the chances of GPU contention
    if multiple threads attempt to use the GPU simultaneously.
    To avoid performance bottlenecks, ensure that the GPU has sufficient memory
    and that the number of threads (`max_workers`) is set appropriately for your hardware.

    :param input_dir: Directory containing input .fasta files.
    :param output_dir: Directory to save the output HDF5 files.
    :param model_name: Name of the encoding model.
    :param threads: Number of parallel threads to use.
    :param batch_size: Batch size for encoding within each thread.
    :return: None. The encodings are written to HDF5 files in the specified output directory.
    """

    os.makedirs(output_dir, exist_ok=True)
    fasta_files = list_fasta_files(input_dir)  # parallel processing of all files
    assert fasta_files, "No .fasta files found in the input directory"

    using_gpu = torch.cuda.is_available() and model_name != "onehot"

    if threads <= 1 or using_gpu:
        for fasta_file_path in fasta_files:
            generate_encodings_of_fasta(
                fasta_file_path=fasta_file_path,
                output_file_path=os.path.join(output_dir, os.path.splitext(os.path.basename(fasta_file_path))[0] + ".h5"),
                model_name=model_name,
                batch_size=batch_size
            )
        return None
    else:
        with ThreadPoolExecutor(max_workers=threads) as executor:
            loop = asyncio.get_running_loop()
            tasks = [
                loop.run_in_executor(executor, generate_encodings_of_fasta,
                                     fasta_file_path,
                                     os.path.join(output_dir, os.path.splitext(os.path.basename(fasta_file_path))[0] + ".h5"),
                                     model_name,
                                     batch_size,
                                     True
                                     ) for fasta_file_path in fasta_files
            ]
            return await tqdm.gather(*tasks, desc="Processing all FASTA files")



async def main(model_name: str, input_dir: str, output_dir: str, batch_size: int, threads: int = 1):
    """
    Asynchronous main entry point for the encoding generation script.

    :param model_name: Name of the encoding model.
    :param input_dir: Directory with input FASTA files.
    :param output_dir: Directory for output HDF5 files.
    :param batch_size: Batch size for encoding.
    :param threads: Number of threads for concurrent processing.
    """
    print("Starting the encoding generation process...")
    await generate_encodings_of_all_fasta(input_dir, output_dir, model_name, threads, batch_size)
    print("All files have been processed successfully.")
    torch.cuda.empty_cache()



if __name__ == "__main__":  # calculating encodings locally (use onehot for simplicity and testing)
    from dotenv import load_dotenv
    load_dotenv()

    __model_name = os.getenv("ENCODING_MODEL_LOCAL")
    __input_dir  = os.getenv("ENCODINGS_INPUT_DIR_LOCAL")
    __output_dir = f"{os.getenv('ENCODINGS_OUTPUT_DIR_LOCAL')}/{__model_name}"

    asyncio.run(
        main(
            model_name=__model_name,
            input_dir=__input_dir,
            output_dir=__output_dir,
            batch_size=config.batch_size,
            threads=config.threads
        )
    )

    # Final Memory Cleanup
    AbstractEncodingModel.clear_model_cache()