"""
This module provides custom collate functions for PyTorch DataLoaders.

These functions are designed to process batches of protein sequence encodings,
which can have variable lengths.
They handle tasks such as padding, pooling,
and formatting the data into tensors suitable for model training.
The module offers two main collation strategies:

1. `per_residue_collate_function`:
    Pads sequences to a uniform length to preserve per-residue information.
    It can optionally flatten the encodings.

2. `per_protein_collate_function`:
    Converts variable-length per-residue encodings into fixed-size per-protein encodings
    using a specified pooling method (e.g., mean, max, sum).
"""

from enum import Enum
from typing import List

import torch

from typing_extensions import Union
# from typeguard import typechecked
# from torchtyping import patch_typeguard


from source.custom_types import (
    Encoding_PerProtein_T,
    Encoding_PerResidue_T,
    Encodings_Batch_PerProtein_T,
    Encodings_Batch_PerResidue_T,
    Encodings_Batch_Flattened_T,
    Label_T,
    Labels_Batch_T,
    AttentionMask_PerResidue_T,
    AttentionMask_Batch_PerProtein_T,
    AttentionMask_Batch_PerResidue_T,
    AttentionMask_Batch_Flattened_T,
)



def __flatten(encodings_batch: Encodings_Batch_PerResidue_T, attention_mask_batch: AttentionMask_Batch_PerResidue_T) \
        -> tuple[Encodings_Batch_Flattened_T, AttentionMask_Batch_Flattened_T]:
    """
    Flattens a batch of per-residue encodings and their attention masks.

    This helper function transforms a 3D batch of encodings (batch, sequence, features)
    into a 2D batch (batch, sequence * features) and expands the attention mask
    to match the new flattened dimension.

    :param encodings_batch: A batch of per-residue encodings with the shape [batch_size, seq_len, encoding_dim].
    :param attention_mask_batch: The corresponding attention mask with the shape [batch_size, seq_len].
    :return: A tuple containing the flattened encodings and the expanded attention mask.
    """
    batch_size, seq_len, encoding_dim = encodings_batch.size()  # Flatten into [batch_size, seq_len * in_channels]
    flattened_encodings = encodings_batch.view(batch_size, seq_len * encoding_dim)
    flattened_attention_mask = attention_mask_batch.repeat_interleave(encoding_dim, dim=1)  # Repeat mask values across encoding chunks
    return flattened_encodings, flattened_attention_mask



def per_residue_collate_function(batch: list[tuple[Encoding_PerResidue_T, Label_T, AttentionMask_PerResidue_T]],
                                 max_seq_len: int | None = None, flatten: bool = True) \
        -> tuple[Union[Encodings_Batch_PerResidue_T, Encodings_Batch_Flattened_T], Labels_Batch_T, AttentionMask_Batch_PerResidue_T]:
    """
    Collate function for per-residue encodings.

    Pads or truncates sequences to a uniform length and uses the attention mask
    to distinguish real data from padding.
    It can optionally flatten the encodings for use in models that expect 2D input.

    :param batch: A list of tuples, where each tuple contains a per-residue encoding and its corresponding label.
    :param max_seq_len: The target sequence length.
                        If provided, sequences are added with zeros or truncated to this length.
                        If None, it's assumed all sequences in the batch have the same length.
    :param flatten: If True, flattens the 3D encoding tensor [batch, seq, features]
                    into a 2D tensor [batch, seq * features].
    :return: A tuple containing:
             - The batch of encodings (padded and optionally flattened).
             - The batch of labels.
             - The attention mask for the padded sequences.
    """
    encodings, labels, attention_masks = zip(*batch)
    encodings: List[Encoding_PerResidue_T]
    labels: List[Label_T]
    attention_masks: List[AttentionMask_PerResidue_T]

    if max_seq_len is not None:
        # Truncate if they exceed the max_seq_len or pad to max_seq_len
        encodings = [
            seq[:max_seq_len] if seq.shape[0] >= max_seq_len  # truncate if exceeding max_seq_len
            else torch.cat([seq, torch.zeros((max_seq_len - seq.shape[0], seq.shape[1]), dtype=seq.dtype, device=seq.device)], dim=0)
            for seq in encodings       # else pad to max_seq_len
        ] # Shape: [max_seq_len, in_channels]

        encodings_batch: Encodings_Batch_PerResidue_T = torch.stack(encodings, dim=0)  # bundle to a batch
        # Shape: [batch_size, max_seq_len, in_channels]

        attention_masks = [
            mask[:max_seq_len] if mask.shape[0] >= max_seq_len # truncate if exceeding max_seq_len
            else torch.cat([mask, torch.full((max_seq_len - mask.shape[0],), fill_value=False, dtype=torch.bool, device=mask.device)], dim=0)
            for mask in attention_masks       # else pad to max_seq_len
        ] # Shape: [max_seq_len]

        attention_mask_batch: AttentionMask_Batch_PerResidue_T = torch.stack(attention_masks, dim=0)  # bundle to a batch
        # Shape: [batch_size, max_seq_len]

    else:
        # Supports both 3D (per-residue) and 2D (per-protein) encodings.
        # Assume already correctly sized tensors
        encodings_batch: Encodings_Batch_PerResidue_T = torch.stack(encodings, dim=0)
        # Shape: [batch_size, seq_len, in_channels]

        attention_mask_batch = torch.stack(attention_masks, dim=0)
        # Shape: [batch_size, seq_len]

    # Flatten encodings into [batch_size, seq_len * in_channels]
    if flatten:
        encodings_batch, attention_mask_batch = __flatten(encodings_batch, attention_mask_batch)
        encodings_batch: Encodings_Batch_Flattened_T = encodings_batch
        attention_mask_batch: AttentionMask_Batch_Flattened_T = attention_mask_batch

    # Stack all_labels into a 1D tensor
    labels_batch: Labels_Batch_T = torch.stack(labels, dim=0)  # Shape: [num_samples]

    return encodings_batch, labels_batch, attention_mask_batch



class PoolingMethod(Enum):
    """Specifies the pooling method for aggregating per-residue encodings."""
    Mean = 0
    Max = 1
    Sum = 2


def per_protein_collate_function(
        batch: list[tuple[Encoding_PerResidue_T, Label_T, AttentionMask_PerResidue_T]],
        pool_method: PoolingMethod = PoolingMethod.Mean
) -> tuple[Encodings_Batch_PerProtein_T, Labels_Batch_T, AttentionMask_Batch_PerProtein_T]:
    """
    Collate function that creates per-protein encodings by pooling per-residue encodings.

    This function takes a batch of variable-length per-residue encodings
    and converts each into a single, fixed-size per-protein encoding
    by applying a pooling operation (mean, max, or sum) across the sequence dimension.

    :param batch: A list of tuples, each containing a per-residue encoding and its label.
    :param pool_method: The pooling strategy to use (Mean, Max, or Sum).
    :return: A tuple containing:
             - The batch of fixed-size per-protein encodings.
             - The batch of labels.
             - A trivial attention mask (all True), as there is no sequence dimension.
    """

    encodings, labels, attention_masks = zip(*batch)
    encodings: List[Encoding_PerResidue_T]
    labels: List[Label_T]
    attention_masks: List[AttentionMask_PerResidue_T]

    def pool(encoding, mask): # Pool based on the mask for each item
        valid_encoding = encoding[mask] # Select only the valid residues using the boolean mask

        if valid_encoding.shape[0] == 0: # Handle cases where a sequence might be empty after masking
            return torch.zeros(encoding.shape[1], dtype=encoding.dtype, device=encoding.device)

        # For each per-residue encoding, calculate with the chosen pooling method across the sequence dimension (dim=0)
        # [seq_len, in_channels] -> [in_channels]

        match pool_method:
            case PoolingMethod.Mean: return torch.mean(valid_encoding, dim=0).values
            case PoolingMethod.Max:  return torch.max(valid_encoding, dim=0).values
            case PoolingMethod.Sum:  return torch.sum(valid_encoding, dim=0).values
            # case _: raise ValueError(f"Invalid pooling method: {pool_method}")


    per_protein_encodings: List[Encoding_PerProtein_T] = [pool(enc, mask) for enc, mask in zip(encodings, attention_masks)]

    # Stack along the batch dimension (dim=0) -> [batch_size, in_channels]
    encodings_batch = torch.stack(per_protein_encodings, dim=0)

    # Stack all_labels into a 1D tensor [batch_size]
    labels_batch = torch.stack(list(labels), dim=0)

    # Trivial attention mask where all positions are valid [batch_size]
    attention_mask_batch = torch.ones(encodings_batch.size(0), dtype=torch.bool)

    return encodings_batch, labels_batch, attention_mask_batch