"""
This module provides a comprehensive toolkit for handling, processing,
and encoding biological sequence data, primarily from FASTA files.

Key functionalities include:
- Data Structures: Defines `SequenceData` and `SequenceEncodingData` classes to encapsulate sequence records,
  their corresponding labels, and numerical encodings.
- Label Definition: A dedicated `_CellLocation` enum maps 10 distinct cellular locations, facilitating
  consistent and error-free labeling.
- FASTA File Processing: The `read_fasta` function efficiently parses and preprocesses FASTA files, yielding
  `SequenceData` objects in a memory-friendly manner.
- Preprocessing Utilities: Includes functions for sanitizing sequence headers for HDF5 compatibility and
  cleaning sequences by removing gaps, stop codons, and ambiguous amino acid codes.
- Label Parsing: A robust mechanism to parse cellular location labels directly from sequence descriptions.
- File System Utilities: A helper function `list_fasta_files` to locate all FASTA files within a specified
  directory tree.
"""

import os
import numpy

from enum import Enum, EnumType
from sys import stderr
from typing import Generator
from Bio.Seq import MutableSeq
from Bio.SeqIO import SeqRecord, parse



# Unique labels in the dataset (25):
class _CellLocation(Enum):
    Cell_membrane = 0  # Cell.membrane-M
    Cytoplasm = 1  # Cytoplasm-S, Cytoplasm-Nucleus-U
    Endoplasmic_reticulum = 2  # Endoplasmic.reticulum-M, Endoplasmic.reticulum-S, Endoplasmic.reticulum-U
    Extracellular = 3  # Extracellular-S
    Golgi_apparatus = 4  # Golgi.apparatus-M, Golgi.apparatus-S, Golgi.apparatus-U
    Lysosome_Vacuole = 5  # Lysosome/Vacuole-M, Lysosome/Vacuole-S, Lysosome/Vacuole-U
    Mitochondrion = 6  # Mitochondrion-M, Mitochondrion-S, Mitochondrion-U
    Nucleus = 7  # Nucleus-M, Nucleus-S, Nucleus-U
    Peroxisome = 8  # Peroxisome-M, Peroxisome-S, Peroxisome-U
    Plastid = 9  # Plastid-M, Plastid-S, Plastid-U


Label: EnumType = _CellLocation
all_labels: list[Label] = list(Label)
num_classes = len(all_labels)



class SequenceData:
    """
    A container for a biological sequence, its corresponding label, and an optional encoding.

    :param seq_record: The raw sequence record object from Biopython.
    :param label: The classification label for the sequence.
    """

    def __init__(self, seq_record: SeqRecord, label: Label):
        self.record: SeqRecord = seq_record
        self.label: Label = label


    def __str__(self):
        return f"SequenceData id={self.record.id}, label={self.label.name}, length={len(self.record.seq)}"



class SequenceEncodingData(SequenceData):
    """
    Extends SequenceData to specifically handle sequences with their required numerical encodings.

    :param seq_record: The raw sequence record object from Biopython.
    :param label: The classification label for the sequence.
    :param encoding: The numerical representation (embedding) of the sequence.
    :param attention_mask: An optional boolean mask indicating true tokens vs. padding.
    """

    def __init__(self, seq_record: SeqRecord, label: Label, encoding: numpy.ndarray, attention_mask: numpy.ndarray):
        super().__init__(seq_record=seq_record, label=label)
        self.__encoding: numpy.ndarray = encoding
        self.__attention_mask: numpy.ndarray = attention_mask


    @staticmethod
    def of(seq_data: SequenceData, encoding: numpy.ndarray, attention_mask: numpy.ndarray) -> 'SequenceEncodingData':
        """
        A factory method to create a SequenceEncodingData instance from a SequenceData object.

        :param seq_data: The original SequenceData object.
        :param encoding: The numerical encoding to add.
        :param attention_mask: The corresponding attention mask.
        :return: A new instance of SequenceEncodingData.
        """
        return SequenceEncodingData(
            seq_record = seq_data.record,
            label = seq_data.label,
            encoding = encoding,
            attention_mask = attention_mask
        )


    def get_encoding(self) -> numpy.ndarray:
        """
        Returns the numerical encoding of the sequence.

        :return: A numpy array representing the sequence encoding.
        """
        return self.__encoding


    def get_attention_mask(self) -> numpy.ndarray:
        """
        Returns the attention mask, if available.

        :return: A numpy array representing the attention mask, or None.
        """
        return self.__attention_mask


    def __str__(self):
        return (
            super().__str__() +
            f", encoding_shape={self.__encoding.shape}" +
            f", mask_shape={self.__attention_mask.shape}" if self.__attention_mask is not None else ""
        )




def __parse_label(seq_record: SeqRecord) -> Label:
    """
    Parses the cell location label from a formatted sequence description.

    :param seq_record: The sequence record containing the description (e.g., '>Q6QNK2 Cell.membrane-M').
    :return: The corresponding `Label` enum member.
    :raises ValueError: If the description is malformed or the label is unknown.
    """
    try:
        # Extract the label and apply replacements
        label_string: str = (
            seq_record.description.split(' ')[1]
            .replace('-Nucleus', '')
            .replace('-M', '')
            .replace('-S', '')
            .replace('-U', '')
            .replace('.', '_')
            .replace('/', '_')
            .replace('-', '_')
        )
        return Label.__getitem__(label_string)  # Match the label to an enum member
    except IndexError: raise ValueError(f"Malformed sequence description: '{seq_record.description}'. Missing label part.")
    except KeyError: raise ValueError(f"Unknown label in description: '{seq_record.description}'. Parsed label does not match any known CellLocation.")



def __seq_record_preprocess(seq_record: SeqRecord) -> None:
    """
    Sanitizes a sequence record's header and sequence in-place.

    The following steps are applied:
    1. Replaces '/' in the header (`record.id`) with '_' to ensure HDF5 compatibility.
    2. Removes gaps ('-') and stop codons ('*') from the sequence.
    3. Replaces ambiguous amino acid codes ('U', 'Z', 'O') with 'X'.

    :param seq_record: The `SeqRecord` object to preprocess.
    :raises ValueError: If invalid sequence headers are detected.
    :return: None. This function modifies the input `seq_records` in place.
    """
    if '/' in seq_record.id:
        error_msg = (
                f"'Sequence header contains '/' which is not allowed in HDF5 group names and will be removed automatically:\n"
                + f"'{seq_record.id}'\n"
        )
        print(error_msg, file=stderr)

    seq_record.id = seq_record.id.replace("/", "_")
    seq_record.seq = (MutableSeq(seq_record.seq)
     .replace(old="-", new="", inplace=True)  # Remove gaps
     .replace(old="*", new="", inplace=True)  # remove stop codons
     .replace(old="U", new="X", inplace=True)
     .replace(old="Z", new="X", inplace=True)
     .replace(old="O", new="X", inplace=True))



def list_fasta_files(directory: str) -> list[str]:
    """
    Lists all .fasta files in the given directory and its subdirectories.

    :param directory: The directory to search in.
    :return: A list of file paths for all found .fasta files.
    :raises FileNotFoundError: If the specified directory does not exist.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Error: Directory '{directory}' does not exist or is not a directory.")

    fasta_file_paths: list[str] = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".fasta"):
                fasta_file_paths.append(os.path.join(root, file))

    return fasta_file_paths



Seq_Data_Generator_T = Generator[SequenceData, None, None]
# declaring type here instead in custom_types.py because of a circular dependency problem

def read_fasta(input_file_path: str) -> Seq_Data_Generator_T:
    """
    Reads, parses, and preprocesses protein sequences from a FASTA file.

    This function yields sequence data one by one for memory-efficiency.

    :param input_file_path: The path to the input .fasta file.
    :return: A generator that yields a `SequenceData` object for each record.
    """
    with open(input_file_path, "r") as input_file_handle:
        for seq_record in parse(handle=input_file_handle, format="fasta"):
            seq_record: SeqRecord
            __seq_record_preprocess(seq_record=seq_record)
            yield SequenceData(seq_record=seq_record, label=__parse_label(seq_record))