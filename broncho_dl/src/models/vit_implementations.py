import torch
import torch.nn as nn
from src.models.utils.positional_encoding import StandardPositionalEncoding, TemporalPositionalEncoding
from src.models.utils.embeddings import VitPatchEmbedding
from src.models.transformer_implementations import TransformerEncoder, TransformerEncoderVanilla
from src.models.utils.to_patches import Img2Patches, Mel2Patches_Time_Axis, Video2Patches
from typing import Optional
from src.models.utils.helpers import SelectToken, Normalize1Dim, LearnableLogitScaling
import math


class Vit(nn.Module):
    """A complete ViT encoder w/ tokenization, learnable positional embedding."""

    def __init__(self, channel_size: int = 3, model_dim: int = 32, num_heads: int = 2,
                 dropout: float = 0.0, input_dropout: float = 0.0,
                 num_layers: int = 2, patch_size: tuple = (4, 4), input_size: Optional[tuple] = None,
                 num_emb: Optional[int] = 100,
                 **kwargs):
        """
        Inputs:
            channel_size - Hidden dimensionality of the input
            model_dim - Hidden dimensionality to use inside the Transformer
            num_classes - Number of classes to predict per sequence element
            num_heads - Number of heads to use in the Multi-Head Attention blocks
            dropout - Dropout to apply inside the model
            input_dropout - Dropout to apply on the input features
            add_positional_encoding - if positional encoding added
            num_layers - number of attention layers
        """
        super().__init__()
        self.channel_size = channel_size
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.num_layers = num_layers
        self.cls = nn.Parameter(torch.randn(1, 1, model_dim))

        # convert the input image tensor to patches
        self.to_patches = Img2Patches(input_size, patch_size)
        # Input dim -> Model dim
        patch_dim = patch_size[0] * patch_size[1] * channel_size
        self.input_emb = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, model_dim),
            nn.LayerNorm(model_dim),
        )
        min_emb = math.prod(input_size) // math.prod(patch_size)
        if min_emb > num_emb:
            print(f"at least {min_emb} embedding vectors needed for Vit model")
            num_emb = min_emb
        else:
            print(f"{num_emb} of embedding vectors initialized")
        self.positional_encoding = VitPatchEmbedding(num_patches=num_emb, emb_dim=model_dim)
        self.transformer_encoder = TransformerEncoderVanilla(token_dim=self.model_dim,
                                                             num_blocks=self.num_layers,
                                                             num_heads=self.num_heads,
                                                             dropout=self.dropout,
                                                             batch_first=True,
                                                             norm_first=True)

    def forward(self, x: torch.Tensor) -> [torch.Tensor, list]:
        """
        Inputs:
            x - Input img tensor of shape [batch size, channel size, height, width]
        Returns:
            x - Aggregation token of shape [Batch, model_dim]
            attn_map - list of attention maps of different with shape
                        [ num_layers x tensor(batch_size, num_heads, seq_len, seq_len) ]
        """
        x = self.to_patches(x)
        x = self.input_emb(x)
        cls = self.cls.expand(x.shape[0], self.cls.shape[1], self.cls.shape[2])
        x = torch.cat([cls, x], dim=1)
        x = self.positional_encoding(x)
        x, attn_maps = self.transformer_encoder(x)
        x = x[:, 0]
        return x, attn_maps


class LrnEmb_Agg_Trf(nn.Module):
    """A ViT encoder w/o tokenization, but w/ learnable positional embedding."""

    def __init__(self, model_dim: int = 32, num_heads: int = 2,
                 dropout: float = 0.0, input_dropout: float = 0.0,
                 num_layers: int = 2,
                 num_emb: Optional[int] = 100,
                 **kwargs):
        """
        Inputs:
            channel_size - Hidden dimensionality of the input
            model_dim - Hidden dimensionality to use inside the Transformer
            num_classes - Number of classes to predict per sequence element
            num_heads - Number of heads to use in the Multi-Head Attention blocks
            dropout - Dropout to apply inside the model
            input_dropout - Dropout to apply on the input features
            add_positional_encoding - if positional encoding added
            num_layers - number of attention layers
        """
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.num_layers = num_layers
        self.cls = nn.Parameter(torch.randn(1, 1, model_dim))
        self.positional_encoding = VitPatchEmbedding(num_patches=num_emb, emb_dim=model_dim)
        self.transformer_encoder = TransformerEncoderVanilla(token_dim=self.model_dim,
                                                             num_blocks=self.num_layers,
                                                             num_heads=self.num_heads,
                                                             dropout=self.dropout,
                                                             batch_first=True,
                                                             norm_first=True)

    def forward(self, x: torch.Tensor) -> [torch.Tensor, list]:
        """
        Inputs:
            x - tokens of shape [batch, SeqLen, model_dim]
        Returns:
            x - Output features of shape [Batch, SeqLen, model_dim]
            attn_map - list of attention maps of different with shape
                        [ num_layers x tensor(batch_size, num_heads, seq_len, seq_len) ]
        """
        cls = self.cls.expand(x.shape[0], self.cls.shape[1], self.cls.shape[2])
        x = torch.cat([cls, x], dim=1)
        x = self.positional_encoding(x)
        x, attn_maps = self.transformer_encoder(x)
        return x, attn_maps


class VitVATT3D(nn.Module):
    """A complete ViT for 3d video w/ tokenization, learnable positional embedding from VATT but with traditional
    vit emb."""

    def __init__(self, channel_size: int = 3, model_dim: int = 32, num_heads: int = 2,
                 dropout: float = 0.0, input_dropout: float = 0.0,
                 num_layers: int = 2, patch_size: tuple = (1, 4, 4), input_size: Optional[tuple] = None,
                 num_emb: Optional[int] = 100,
                 **kwargs):
        """
        Inputs:
            channel_size - Hidden dimensionality of the input
            model_dim - Hidden dimensionality to use inside the Transformer
            num_classes - Number of classes to predict per sequence element
            num_heads - Number of heads to use in the Multi-Head Attention blocks
            dropout - Dropout to apply inside the model
            input_dropout - Dropout to apply on the input features
            add_positional_encoding - if positional encoding added
            num_layers - number of attention layers
        """
        super().__init__()
        self.channel_size = channel_size
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.num_layers = num_layers
        self.cls = nn.Parameter(torch.randn(1, 1, model_dim))

        # convert the input image tensor to patches
        self.to_patches = Video2Patches(input_size, patch_size)
        # Input dim -> Model dim
        patch_dim = math.prod(patch_size) * channel_size
        self.input_emb = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, model_dim),
            nn.LayerNorm(model_dim),
        )
        min_emb = math.prod(input_size) // math.prod(patch_size)
        if min_emb > num_emb:
            print(f"at least {min_emb} embedding vector needed for VitVATT3D model")
            num_emb = min_emb
        else:
            print(f"{num_emb} of embedding vectors initialized")
        self.positional_encoding = VitPatchEmbedding(num_patches=num_emb, emb_dim=model_dim)
        self.transformer_encoder = TransformerEncoderVanilla(token_dim=self.model_dim,
                                                             num_blocks=self.num_layers,
                                                             num_heads=self.num_heads,
                                                             dropout=self.dropout,
                                                             batch_first=True,
                                                             norm_first=True)

    def forward(self, x: torch.Tensor) -> [torch.Tensor, list]:
        """
        Inputs:
            x - Input img tensor of shape [batch size, channel size, height, width]
        Returns:
            x - Aggregation token of shape [Batch, model_dim]
            attn_map - list of attention maps of different with shape
                        [ num_layers x tensor(batch_size, num_heads, seq_len, seq_len) ]
        """
        x = self.to_patches(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        x = self.input_emb(x)
        cls = self.cls.expand(x.shape[0], self.cls.shape[1], self.cls.shape[2])
        x = torch.cat([cls, x], dim=1)
        x = self.positional_encoding(x)
        x, attn_maps = self.transformer_encoder(x)
        x = x[:, 0]
        return x, attn_maps


class Vit_Classifier(nn.Module):
    """The ViT based Classifier w/ tokenization, standard 2d pos emb."""

    def __init__(self, channel_size: int = 3, model_dim: int = 32, num_classes: int = 10, num_heads: int = 2,
                 dropout: float = 0.0, input_dropout: float = 0.0, add_positional_encoding: bool = True,
                 num_layers: int = 2, patch_size: tuple = (4, 4), input_size: Optional[tuple] = None,
                 **kwargs):
        """
        Inputs:
            channel_size - Hidden dimensionality of the input
            model_dim - Hidden dimensionality to use inside the Transformer
            num_classes - Number of classes to predict per sequence element
            num_heads - Number of heads to use in the Multi-Head Attention blocks
            dropout - Dropout to apply inside the model
            input_dropout - Dropout to apply on the input features
            add_positional_encoding - if positional encoding added
            num_layers - number of attention layers
        """
        super().__init__()
        self.channel_size = channel_size
        self.model_dim = model_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.add_positional_encoding = add_positional_encoding
        self.num_layers = num_layers
        self.cls = nn.Parameter(torch.randn(1, 1, model_dim))

        # convert the input image tensor to patches
        self.to_patches = Img2Patches(input_size, patch_size)
        # Input dim -> Model dim
        patch_dim = patch_size[0] * patch_size[1] * channel_size
        self.input_emb = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, model_dim),
            nn.LayerNorm(model_dim),
        )

        # Positional encoding for sequences
        if add_positional_encoding:
            self.positional_encoding = StandardPositionalEncoding(d_model=self.model_dim)
        self.transformer_encoder = TransformerEncoderVanilla(token_dim=self.model_dim,
                                                             num_blocks=self.num_layers,
                                                             num_heads=self.num_heads,
                                                             dropout=self.dropout,
                                                             batch_first=True,
                                                             norm_first=True)
        # Output classifier per sequence element
        self.output_net = nn.Sequential(nn.Linear(self.model_dim, self.model_dim),
                                        nn.LayerNorm(self.model_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(self.dropout),
                                        nn.Linear(self.model_dim, self.num_classes))

    def forward(self, x: torch.Tensor) -> [torch.Tensor, list]:
        """
        Inputs:
            x - Input img tensor of shape [batch size, channel size, height, width]
        Returns:
            x - Output class possibility of shape [Batch, num_classes]
            attn_map - list of attention maps of different with shape
                        [ num_layers x tensor(batch_size, num_heads, seq_len, seq_len) ]
        """
        x = self.to_patches(x)
        x = self.input_emb(x)
        cls = self.cls.expand(x.shape[0], self.cls.shape[1], self.cls.shape[2])
        x = torch.cat([cls, x], dim=1)
        if self.add_positional_encoding:
            x = self.positional_encoding(x)
        x, attn_maps = self.transformer_encoder(x)
        x = x[:, 0]
        x = self.output_net(x)
        return x, attn_maps


class Vit_Classifier_Mel(nn.Module):
    """The ViT based Classifier for Mel Spectrogram input w/ tokenization and embedding."""

    def __init__(self, model_dim: int = 32, num_classes: int = 10, num_heads: int = 2,
                 dropout: float = 0.0, input_dropout: float = 0.0, add_positional_encoding: bool = True,
                 num_layers: int = 2, input_size: Optional[tuple] = None,
                 **kwargs):
        """
        Inputs:
            channel_size - Hidden dimensionality of the input
            model_dim - Hidden dimensionality to use inside the Transformer
            num_classes - Number of classes to predict per sequence element
            num_heads - Number of heads to use in the Multi-Head Attention blocks
            dropout - Dropout to apply inside the model
            input_dropout - Dropout to apply on the input features
            add_positional_encoding - if positional encoding added
            num_layers - number of attention layers
        """
        super().__init__()
        self.model_dim = model_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.add_positional_encoding = add_positional_encoding
        self.num_layers = num_layers
        self.cls = nn.Parameter(torch.randn(1, 1, model_dim))

        # convert the input image tensor to patches
        self.to_patches = Mel2Patches_Time_Axis()
        # Input dim -> Model dim
        patch_dim = input_size[0]
        seq_len = input_size[1]
        self.input_emb = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, model_dim),
            nn.LayerNorm(model_dim),
        )

        # Positional encoding for sequences
        if add_positional_encoding:
            # self.positional_encoding = TemporalPositionalEncoding(seq_len + 1, model_dim)
            self.positional_encoding = StandardPositionalEncoding(d_model=self.model_dim)
        self.transformer_encoder = TransformerEncoderVanilla(token_dim=self.model_dim,
                                                             num_blocks=self.num_layers,
                                                             num_heads=self.num_heads,
                                                             dropout=self.dropout,
                                                             batch_first=True,
                                                             norm_first=True)
        # Output classifier per sequence element
        self.output_net = nn.Sequential(nn.Linear(self.model_dim, self.model_dim),
                                        nn.LayerNorm(self.model_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(self.dropout),
                                        nn.Linear(self.model_dim, self.num_classes))

    def forward(self, x: torch.Tensor) -> [torch.Tensor, list]:
        """
        Inputs:
            x - Input mel spectrogram tensor of shape [batch size, channel size, height, width]
        Returns:
            x - Output features of shape [Batch, num_classes]
            attn_map - list of attention maps of different with shape
                        [ num_layers x tensor(batch_size, num_heads, seq_len, seq_len) ]
        """
        x = self.to_patches(x)
        x = self.input_emb(x)
        cls = self.cls.expand(x.shape[0], self.cls.shape[1], self.cls.shape[2])
        x = torch.cat([cls, x], dim=1)
        if self.add_positional_encoding:
            x = self.positional_encoding(x)
        x, attn_maps = self.transformer_encoder(x)
        x = x[:, 0]
        x = self.output_net(x)
        return x, attn_maps


class Transformer_Classifier_NoTokenNoEmb(nn.Module):
    """The ViT based Classifier w/o tokenization or embedding(directly takes embedded tokens as input)."""

    def __init__(self, model_dim: int = 32, num_classes: int = 10, num_heads: int = 2,
                 dropout: float = 0.0, input_dropout: float = 0.0,
                 num_layers: int = 2, **kwargs):
        """
        Inputs:
            channel_size - Hidden dimensionality of the input
            model_dim - Hidden dimensionality to use inside the Transformer
            num_classes - Number of classes to predict per sequence element
            num_heads - Number of heads to use in the Multi-Head Attention blocks
            dropout - Dropout to apply inside the model
            input_dropout - Dropout to apply on the input features
            add_positional_encoding - if positional encoding added
            num_layers - number of attention layers
        """
        super().__init__()
        self.model_dim = model_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.num_layers = num_layers
        self.transformer_encoder = TransformerEncoderVanilla(token_dim=self.model_dim,
                                                             num_blocks=self.num_layers,
                                                             num_heads=self.num_heads,
                                                             dropout=self.dropout,
                                                             batch_first=True,
                                                             norm_first=True)
        # Output classifier per sequence element
        self.output_net = nn.Sequential(nn.Linear(self.model_dim, self.model_dim),
                                        nn.LayerNorm(self.model_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(self.dropout),
                                        nn.Linear(self.model_dim, self.num_classes))

    def forward(self, x: torch.Tensor) -> [torch.Tensor, list]:
        """
        Inputs:
            x - Input sequence of tokens [batch size, SeqLen, model_dim]
        Returns:
            x - Output features of shape [Batch, num_classes]
            attn_map - list of attention maps of different with shape
                        [ num_layers x tensor(batch_size, num_heads, seq_len, seq_len) ]
        """
        x, attn_maps = self.transformer_encoder(x)
        x = x[:, 0]
        x = self.output_net(x)
        return x, attn_maps


# class TransformerClassifierVitVanillaNoPatch(nn.Module):
#     """The ViT based Classifier w/o tokenization or embedding(directly takes embedded tokens as input)."""
#
#     def __init__(self, model_dim: int = 32, num_classes: int = 10, num_heads: int = 2,
#                  dropout: float = 0.0, input_dropout: float = 0.0,
#                  num_layers: int = 2, **kwargs):
#         """
#         Inputs:
#             channel_size - Hidden dimensionality of the input
#             model_dim - Hidden dimensionality to use inside the Transformer
#             num_classes - Number of classes to predict per sequence element
#             num_heads - Number of heads to use in the Multi-Head Attention blocks
#             dropout - Dropout to apply inside the model
#             input_dropout - Dropout to apply on the input features
#             add_positional_encoding - if positional encoding added
#             num_layers - number of attention layers
#         """
#         super().__init__()
#         self.model_dim = model_dim
#         self.num_classes = num_classes
#         self.num_heads = num_heads
#         self.dropout = dropout
#         self.input_dropout = input_dropout
#         self.num_layers = num_layers
#         self.transformer_encoder = TransformerEncoderVanilla(token_dim=self.model_dim,
#                                                              num_blocks=self.num_layers,
#                                                              num_heads=self.num_heads,
#                                                              dropout=self.dropout,
#                                                              batch_first=True,
#                                                              norm_first=True)
#         # Output classifier per sequence element
#         self.output_net = nn.Sequential(nn.Linear(self.model_dim, self.model_dim),
#                                         nn.LayerNorm(self.model_dim),
#                                         nn.ReLU(inplace=True),
#                                         nn.Dropout(self.dropout),
#                                         nn.Linear(self.model_dim, self.num_classes))
#
#     def forward(self, x: torch.Tensor) -> [torch.Tensor, list]:
#         """
#         Inputs:
#             x - Input sequence of tokens [batch size, SeqLen, model_dim]
#         Returns:
#             x - Output features of shape [Batch, SeqLen, model_dim]
#             attn_map - list of attention maps of different with shape
#                         [ num_layers x tensor(batch_size, num_heads, seq_len, seq_len) ]
#         """
#         x, attn_maps = self.transformer_encoder(x)
#         x = x[:, 0]
#         x = self.output_net(x)
#         return x, attn_maps


class VitImageBind(nn.Module):
    """The ViT based transformer encoder takes Unembed tokens as input using TransformerEncoderVanilla with norm
    first and output the latent vector. borrowed from ImageBind source code"""

    def __init__(self, model_dim: int = 32, num_heads: int = 2,
                 dropout: float = 0.0, input_dropout: float = 0.0,
                 num_layers: int = 2, num_pos_emb: int = 31, **kwargs):
        """
        Inputs:
            channel_size - Hidden dimensionality of the input
            model_dim - Hidden dimensionality to use inside the Transformer
            num_classes - Number of classes to predict per sequence element
            num_heads - Number of heads to use in the Multi-Head Attention blocks
            dropout - Dropout to apply inside the model
            input_dropout - Dropout to apply on the input features
            add_positional_encoding - if positional encoding added
            num_layers - number of attention layers
        """
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.num_layers = num_layers
        self.cls = nn.Parameter(torch.randn(1, 1, model_dim))
        self.pos_emb = torch.nn.Parameter(torch.randn([1, num_pos_emb, model_dim]))
        self.transformer_encoder = TransformerEncoderVanilla(token_dim=self.model_dim,
                                                             num_blocks=self.num_layers,
                                                             num_heads=self.num_heads,
                                                             dropout=self.dropout,
                                                             batch_first=True,
                                                             norm_first=True)
        self.output_net = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            SelectToken(index=0),
            nn.Dropout(self.dropout),
            nn.Linear(self.model_dim, self.model_dim),
            Normalize1Dim(dim=-1),
            LearnableLogitScaling(logit_scale_init=5.0, learnable=True),
        )

    def forward(self, x: torch.Tensor) -> [torch.Tensor, list]:
        """
        Inputs:
            x - Input sequence of tokens [batch size, SeqLen, model_dim]
        Returns:
            x - Aggregation token of shape [Batch, model_dim]
            attn_map - list of attention maps of different with shape
                        [ num_layers x tensor(batch_size, num_heads, seq_len, seq_len) ]
        """
        cls = self.cls.expand(x.shape[0], self.cls.shape[1], self.cls.shape[2])
        x = torch.cat([cls, x], dim=1)
        x += self.pos_emb[:, :x.shape[1], :]
        x, attn_maps = self.transformer_encoder(x)
        x = self.output_net(x)
        return x, attn_maps
