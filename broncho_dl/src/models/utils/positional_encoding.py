"""
Adapted from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
"""
import torch
import math


class StandardPositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int = 256, max_len: int = 5000, **kwargs) -> None:
        """Add positional encoding from tutorial 6 to the input tokens for transformer.

        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x: input for positional encoding [ B, N, D]
                N: sequence length B: batch size D: length of token

        Returns:
             output for positional encoding [ B, N, D]
                N: sequence length B: batch size D: length of token

        """
        x = x + self.pe[:, :x.size(1)]
        return x


class TemporalPositionalEncoding(torch.nn.Module):

    def __init__(self, time_len: int = 128, dim: int = 64) -> None:
        """Add positional encoding from tutorial 6 to the input tokens for transformer.

        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(time_len, dim)
        time_div_term = torch.exp(torch.arange(0, time_len, 1).float() * (-math.log(10000.0) / time_len))
        pe = pe + torch.sin(time_div_term).unsqueeze(1)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x: input for positional encoding [ B, N, D]
                N: sequence length B: batch size D: length of token

        Returns:
             output for positional encoding [ B, N, D]
                N: sequence length B: batch size D: length of token

        """
        x = x + self.pe[:, :x.size(1)]
        return x


class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
