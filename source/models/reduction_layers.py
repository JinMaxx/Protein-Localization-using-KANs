"""
This module defines a collection of PyTorch `nn.Module` classes
designed to reduce the dimensionality of sequence data.
These "reduction layers" can decrease sequence length, the number of feature channels, or both.

Each layer inherits from the `AbstractReductionLayer`,
which establishes a common interface and validation for input and output dimensions.
The layers offer various strategies for dimensionality reduction,
from simple pooling to more complex, learnable transformations like attention and convolutions.

The primary purpose of these layers is to act as a preprocessing step,
condensing long sequences into a smaller, fixed-size representation that can
be efficiently processed by a subsequent model, such as a Feed-Forward Network.
"""

import torch

from abc import ABC, abstractmethod
from torch import nn, device as torch_device
from typing_extensions import override, Optional

from source.custom_types import Encodings_Batch_PerResidue_T, AttentionMask_Batch_T



class AbstractReductionLayer(nn.Module, ABC):
    """
    Abstract base class for all reduction layers.

    It defines the common constructor signature and enforces basic constraints,
    such as ensuring that output dimensions are not larger than input dimensions.
    Subclasses must implement the `forward` pass and a `size` method to report
    the number of trainable parameters.
    """

    def __init__(self,
            in_channels: int,
            in_seq_len: int,
            out_seq_len: int,
            device: torch_device,
            out_channels: Optional[int] = None):
        super().__init__()

        self.in_channels: int = in_channels
        self.in_seq_len: int = in_seq_len
        self.out_seq_len: int = out_seq_len
        self.out_channels: int = out_channels if out_channels is not None else in_channels

        if self.out_seq_len > self.in_seq_len: raise ValueError("out_seq_len must be <= in_seq_len")
        if self.out_seq_len < 1: raise ValueError("out_seq_len must be > 0")
        if self.out_channels > self.in_channels: raise ValueError("out_channels must be <= in_channels")
        if self.out_channels < 1: raise ValueError("out_channels must be > 0")

        self.device: torch_device = device


    @abstractmethod
    def forward(self, x: Encodings_Batch_PerResidue_T, attention_mask: AttentionMask_Batch_T):
        """Processes the input tensor to reduce its dimensions."""
        pass



class MaxPoolConvLayer(AbstractReductionLayer):
    """
    Reduces sequence length using 1D adaptive max pooling.

    This layer applies max pooling over the sequence dimension to reduce it to `out_seq_len`.
    It preserves the most prominent features (highest values) within the pooling windows.
    This layer is not trainable and does not change the number of channels.
    """

    def __init__(self,
            in_channels: int,
            in_seq_len: int,
            out_seq_len: int,
            device: torch_device,
            **_):
        super().__init__(
            in_channels = in_channels,
            in_seq_len = in_seq_len,
            out_seq_len = out_seq_len,
            out_channels = in_channels,  # in_channels = out_channels
            device = device
        )
        if self.in_channels != self.out_channels:
            raise ValueError("MaxPoolConvLayer does not support changing the channel dimension.")


    @override
    def forward(self, x: Encodings_Batch_PerResidue_T, attention_mask: AttentionMask_Batch_T):
        x = x * attention_mask.unsqueeze(-1)

        # Only interpolate/pool the sequence to the output length
        # x shape: [batch_size, seq_len, in_channels]
        x = x.permute(0, 2, 1)  # [batch_size, in_channels, seq_len]
        x_pooled = torch.nn.functional.adaptive_max_pool1d(
            x, self.out_seq_len
        )
        out = x_pooled.permute(0, 2, 1)  # [batch_size, out_seq_len, in_channels]
        return out



class AvgPoolConvLayer(AbstractReductionLayer):
    """
    Reduces sequence length using 1D adaptive average pooling.

    This layer applies average pooling over the sequence dimension to reduce it to `out_seq_len`.
    It smooths the features by taking their average within the pooling windows.
    This layer is not trainable and does not change the number of channels.
    """

    def __init__(self,
            in_channels: int,
            in_seq_len: int,
            out_seq_len: int,
            device: torch_device,
            **_):
        super().__init__(
            in_channels = in_channels,
            in_seq_len = in_seq_len,
            out_seq_len = out_seq_len,
            out_channels = in_channels,  # in_channels = out_channels
            device = device
        )
        if self.in_channels != self.out_channels:
            raise ValueError("AvgPoolConvLayer does not support changing the channel dimension.")


    @override
    def forward(self, x: Encodings_Batch_PerResidue_T, attention_mask: AttentionMask_Batch_T):
        x = x * attention_mask.unsqueeze(-1)

        # Only interpolate/pool the sequence to the output length
        # Assuming x shape: [batch_size, seq_len, in_channels]
        x = x.permute(0, 2, 1)  # [batch_size, in_channels, seq_len]
        x_pooled = torch.nn.functional.adaptive_avg_pool1d(
            x, self.out_seq_len
        )
        out = x_pooled.permute(0, 2, 1)  # [batch_size, out_seq_len, in_channels]
        return out



class LinearReductionLayer(AbstractReductionLayer):
    """
    Reduces sequence length using a learnable Linear layer.

    This layer learns a linear projection to map the input sequence dimension
    (`in_seq_len`) to the target dimension (`out_seq_len`).
    The transformation is applied independently to each channel.
    This allows the model to learn an optimal way to combine sequence steps.
    """

    def __init__(self,
            in_channels: int,
            in_seq_len: int,
            out_seq_len: int,
            device: torch_device,
            **_):
        super().__init__(
            in_channels = in_channels,
            in_seq_len = in_seq_len,
            out_seq_len = out_seq_len,
            out_channels = in_channels,  # in_channels = out_channels
            device = device
        )
        if self.in_channels != self.out_channels:
            raise ValueError("LinearReductionLayer as designed does not support changing the channel dimension.")
        # Learnable projection matrix: projects in_seq_len → out_seq_len
        self.proj = torch.nn.Linear(self.in_seq_len, self.out_seq_len, bias=True, device=self.device)


    @override
    def forward(self, x: Encodings_Batch_PerResidue_T, attention_mask: AttentionMask_Batch_T):
        x = x * attention_mask.unsqueeze(-1)
        # x: [batch_size, in_seq_len, in_channels]
        x_t = x.transpose(1, 2)  # Transpose for linear layer: [batch_size, in_channels, in_seq_len]
        x_proj = self.proj(x_t)  # Apply projection: [batch_size, in_channels, out_seq_len]
        return x_proj.transpose(1, 2)  # Transpose back: [batch_size, out_seq_len, in_channels]



class AttentionLayer(AbstractReductionLayer):
    """
    Reduces sequence length using a multi-head attention mechanism.

    This layer uses a set of `out_seq_len` learnable "query" vectors.
    These queries attend to the input sequence (acting as key and value),
    producing a reduced sequence of `out_seq_len`.
    This allows the model to dynamically focus on and aggregate information
    from the most relevant parts of the input sequence.
    """

    def __init__(self,
            in_channels: int,
            in_seq_len: int,
            out_seq_len: int,
            device: torch_device,
            **_):
        super().__init__(
            in_channels = in_channels,
            in_seq_len = in_seq_len,
            out_seq_len = out_seq_len,
            out_channels = in_channels,   # in_channels = out_channels
            device = device
        )
        self.attention = torch.nn.MultiheadAttention(
            embed_dim = self.in_channels,
            num_heads = 4,  # Can tune this
            batch_first = True,
            device = device
        )
        # Learnable output queries that will attend to the sequence
        self.output_queries = torch.nn.Parameter(torch.randn(self.out_seq_len, self.in_channels, device=self.device))


    @override
    def forward(self, x: Encodings_Batch_PerResidue_T, attention_mask: AttentionMask_Batch_T):
        # x: [batch_size, seq_len, in_channels]
        batch_size = x.size(0)
        # Expand output_queries for the batch
        queries = self.output_queries.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, out_seq_len, in_channels]

        # MultiheadAttention expects `key_padding_mask` where True indicates a value to be ignored.
        # We assume the input `attention_mask` has True for valid tokens, so we invert it.
        key_padding_mask = ~attention_mask if attention_mask is not None else None

        # Attention: queries, keys, values
        attended, _ = self.attention(
            query = queries,  # [batch, out_seq_len, in_channels]
            key = x,          # [batch, seq_len, in_channels]
            value = x,        # [batch, seq_len, in_channels]
            key_padding_mask = key_padding_mask
        )
        return attended  # [batch, out_seq_len, in_channels]



class PositionalWeightedConvLayer(AbstractReductionLayer):
    """
    Reduces dimensions via learnable positional weights and a 1D convolution.

    This layer first applies learnable, position-specific weights to the input,
    allowing it to emphasize or de-emphasize features at different sequence positions.
    The weighted sequence is then passed through a 1D convolution to extract local patterns,
    followed by adaptive max pooling to achieve the target `out_seq_len`.
    This layer can change the number of channels.
    """

    def __init__(self,
            in_channels: int,
            in_seq_len: int,
            out_seq_len: int,
            device: torch_device,
            out_channels: Optional[int] = None):
        super().__init__(
            in_channels = in_channels,
            in_seq_len = in_seq_len,
            out_seq_len = out_seq_len,
            out_channels = out_channels,
            device = device
        )

        self.conv1d = nn.Conv1d(
            in_channels = self.in_channels,
            out_channels = self.out_channels,
            kernel_size = 3,
            padding = 1
        ).to(self.device)

        # Learnable positional weights
        pos_weights = torch.empty(1, self.in_seq_len, self.in_channels)  # Create an uninitialized tensor
        nn.init.xavier_uniform_(pos_weights, gain=2.0)  # Small values might hinder my learning rate -> up the gain
        self.positional_weights = nn.Parameter(pos_weights).to(self.device)
        self.positional_weights.register_hook(lambda grad: grad * 2)  # works fine in range [2;10]

        c = 4.0  # For sigmoid. Clamp the weights to keep them in range. Avoiding growing out of saturated regime.
        self.positional_weights.data.clamp_(-c, c)

        self.weights_scalar = torch.tensor(2.0).to(self.device)  # 2.0 works quite well.


    @override
    def forward(self, x: Encodings_Batch_PerResidue_T, attention_mask: AttentionMask_Batch_T):
        x = x * attention_mask.unsqueeze(-1)

        # Best results after testing different configurations with steady improvements:
        x_weighted = x * torch.sigmoid(self.positional_weights / self.weights_scalar)  # Applying positional weights
        # Regular unscaled sigmoid(weights) saturates too aggressively and grad becomes very small.

        x_weighted = x_weighted.permute(0, 2, 1)  # [batch_size, in_channels, seq_len]

        x_conv = nn.functional.relu(self.conv1d(x_weighted)) # Convolution operation

        # Before pooling, mask the output of the convolution by replacing padded values
        # with negative infinity, ensuring they are ignored by the max pooling operation.
        attention_mask_3d = attention_mask.unsqueeze(1)  # Shape: [batch_size, 1, seq_len]
        x_conv_masked = torch.where(attention_mask_3d, x_conv, torch.tensor(float('-inf'), device=x.device))

        # Pool/Interpolate to a fixed length
        x_out = nn.functional.adaptive_max_pool1d(x_conv_masked, self.out_seq_len)

        # Replace any remaining -inf values with 0 (can happen if a pool window is fully masked).
        x_out[x_out == float('-inf')] = 0

        # Final shape: [batch_size, out_seq_len, out_channels]
        x_final = x_out.permute(0, 2, 1)

        return x_final



class UNetReductionLayer(AbstractReductionLayer):
    """
    Reduces sequence length using a U-Net-like encoder-decoder architecture.

    This layer processes the sequence through an encoder path (convolutions and pooling)
    to create a compressed representation,
    and a decoder path (transposed convolutions) that upsamples it.
    Skip connections link the encoder and decoder,
    allowing the model to combine high-level abstract features with low-level details.
    A final adaptive pooling layer ensures the output matches `out_seq_len`.
    """
    
    def __init__(self,
            in_channels: int,
            in_seq_len: int,
            out_seq_len: int,
            device: torch_device,
            out_channels: Optional[int] = None):
        super().__init__(
            in_channels = in_channels,
            in_seq_len = in_seq_len,
            out_seq_len = out_seq_len,
            out_channels = out_channels,
            device = device
        )
        
        # Encoder: 1D convolutions with pooling
        self.enc_conv1 = torch.nn.Conv1d(self.in_channels, self.in_channels, kernel_size=3, padding=1, device=self.device)
        self.enc_pool1 = torch.nn.MaxPool1d(kernel_size=2)
        self.enc_conv2 = torch.nn.Conv1d(self.in_channels, self.in_channels, kernel_size=3, padding=1, device=self.device)
        self.enc_pool2 = torch.nn.MaxPool1d(kernel_size=2)

        # Decoder: Transposed convolutions (upsampling)
        self.dec_conv_trans1 = torch.nn.ConvTranspose1d(self.in_channels, self.in_channels, kernel_size=2, stride=2, device=self.device)
        self.dec_conv1 = torch.nn.Conv1d(2 * self.in_channels, self.in_channels, kernel_size=3, padding=1, device=self.device)
        self.dec_conv_trans2 = torch.nn.ConvTranspose1d(self.in_channels, self.in_channels, kernel_size=2, stride=2, device=self.device)
        self.dec_conv2 = torch.nn.Conv1d(2 * self.in_channels, self.out_channels, kernel_size=3, padding=1, device=self.device)

        # Final projection to the desired output sequence length
        self.final_pool = torch.nn.AdaptiveMaxPool1d(self.out_seq_len)

    
    @override
    def forward(self, x: Encodings_Batch_PerResidue_T, attention_mask: AttentionMask_Batch_T):
        x = x * attention_mask.unsqueeze(-1)
        x = x.transpose(1, 2)  # x: [batch, seq_len, emb] -> transpose for Conv1d: [batch, emb, seq_len]


        ### Encoder ###

        # Level 1
        x1 = torch.relu(self.enc_conv1(x))
        x1_masked = torch.where(attention_mask.unsqueeze(1), x1, -torch.inf)
        p1 = self.enc_pool1(x1_masked)
        p1[p1 == -torch.inf] = 0  # Reset any -inf values to 0 after pooling.

        # Downsample attention mask for the next encoder step
        mask1 = self.enc_pool1(attention_mask.float().unsqueeze(1)).squeeze(1).bool()

        # Level 2
        x2 = torch.relu(self.enc_conv2(p1))
        x2_masked = torch.where(mask1.unsqueeze(1), x2, -torch.inf)
        p2 = self.enc_pool2(x2_masked)
        p2[p2 == -torch.inf] = 0


        ### Decoder with Skip Connections ###

        u1 = self.dec_conv_trans1(p2)
        # If needed, pad/crop x2 to match u1's sequence length
        # noinspection DuplicatedCode
        diff1 = u1.size(-1) - x2.size(-1)
        if diff1 > 0: x2 = torch.nn.functional.pad(x2, (0, diff1))
        elif diff1 < 0: u1 = torch.nn.functional.pad(u1, (0, -diff1))
        d1 = torch.cat([u1, x2], dim=1)
        d1 = torch.relu(self.dec_conv1(d1))

        u2 = self.dec_conv_trans2(d1)
        # noinspection DuplicatedCode
        diff2 = u2.size(-1) - x1.size(-1)
        if diff2 > 0: x1 = torch.nn.functional.pad(x1, (0, diff2))
        elif diff2 < 0: u2 = torch.nn.functional.pad(u2, (0, -diff2))
        d2 = torch.cat([u2, x1], dim=1)
        d2 = torch.relu(self.dec_conv2(d2))


        ### Final Projection ###

        # Apply the original mask before the final adaptive pooling
        # The U-Net ops might slightly change seq length, so we align the mask to the data.
        final_seq_len = d2.size(-1)
        original_seq_len = attention_mask.size(-1)
        if final_seq_len > original_seq_len:
            padding = (0, final_seq_len - original_seq_len)
            final_mask = torch.nn.functional.pad(attention_mask, padding, "constant", False)
        else:
            final_mask = attention_mask[:, :final_seq_len]

        d2_masked = torch.where(final_mask.unsqueeze(1), d2, -torch.inf)
        out = self.final_pool(d2_masked)
        out[out == -torch.inf] = 0

        out = out.transpose(1, 2)  # [batch, out_seq_len, emb]
        return out