# The codes in this files represent an adaptation of SwinUNETR originally developed by MONAI for the task of brain age prediction.
# To aid interpretation, the original variable names have been retained.

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm
from torch import prod, tensor

from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks import PatchEmbed
from monai.networks.layers import DropPath, trunc_normal_, get_act_layer
from monai.utils import ensure_tuple_rep, look_up_option, optional_import

rearrange, _ = optional_import("einops", name="rearrange")

__all__ = [
    "SwinAgeMapper",
    "window_partition",
    "window_reverse",
    "WindowAttention",
    "SwinTransformerBlock",
    "PatchMerging",
    "PatchMergingV2",
    "MERGING_MODE",
    "BasicLayer",
    "SwinTransformer",
]


class SwinAgeMapper(nn.Module):
    """
    SwinAgeMapper is a 3D Swin Transformer based model for brain age prediction.
    
    Adapted from Swin UNETR based on: "Hatamizadeh et al.,
    Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
    <https://arxiv.org/abs/2201.01266>"
    """

    def __init__(
        self,
        img_size: Sequence[int] | int,
        in_channels: int,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
        fully_connected_activation='relu',
        resolution='1mm',
        patch_size=2,
    ) -> None:
        
        """
        Parameters
        ----------
        img_size : Sequence[int] | int
            Dimension of input image. For SwinAgeMapper, this is (160, 192, 160)
        in_channels : int
            Dimension of input channels. For SwinAgeMapper, this is 1
        depths : Sequence[int], optional
            Number of layers in each stage, by default (2, 2, 2, 2). This corresponds to the number of SwinTransformerBlocks at each depth stage. 
        num_heads : Sequence[int], optional
            Number of attention heads, by default (3, 6, 12, 24). This corresponds to the number of attention heads at each depth stage.
        feature_size : int, optional
            Dimension of network feature size, by default 24. This corresponds to the arbitrary internal dimension in which the inputs are projected to before the attention operation.
        drop_rate : float, optional
            Dropout rate, by default 0.0. This corresponds to the dropout rate applied to the input of each SwinTransformerBlock.
        attn_drop_rate : float, optional
            Attention dropout rate, by default 0.0. This corresponds to the dropout rate applied to the attention weights of each SwinTransformerBlock.
        dropout_path_rate : float, optional
            Drop path rate, by default 0.0. This corresponds to the dropout rate applied to the skip connection of each SwinTransformerBlock.
        use_checkpoint : bool, optional
            Use gradient checkpointing for reduced memory usage, by default False. 
        spatial_dims : int, optional
            Number of spatial dims, by default 3. This corresponds to the number of spatial dimensions of the input image.
        downsample : str, optional
            Module used for downsampling. Available options are `"mergingv2"`, `"merging"` and a user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
            The default is currently `"merging"` (the original version defined in v0.9.0), by default "merging".
        fully_connected_activation : str, optional
            Activation function for fully connected layers, by default 'relu'. This corresponds to the activation function applied to the output of the Feed Forward Blocks.
        resolution : str, optional
            Resolution of the input image, by default '1mm' isotropic. If the resolution is 2mm, an Upsampling layer is added to the network.
        patch_size : int, optional
            Size of the patches, by default 2. This corresponds to the size of the patches that are extracted from the input image.

        Raises
        ------
        ValueError
            Input image size (img_size) should be divisible by stage-wise image resolution.
        ValueError
            Dropout rate should be between 0 and 1.
        ValueError
            Attention dropout rate should be between 0 and 1.
        ValueError
            Drop path rate should be between 0 and 1.

        Examples
        --------
        for 3D single channel input with size (96,96,96), 4-channel output and feature size of 48:
        >>> net = SwinAgeMapper(img_size=(96,96,96), in_channels=1, out_channels=4, feature_size=48)
        # for 3D 4-channel input with size (128,128,128), 3-channel output and (2,4,2,2) layers in each stage.
        >>> net = SwinAgeMapper(img_size=(128,128,128), in_channels=4, out_channels=3, depths=(2,4,2,2))
        # for 2D single channel input with size (96,96), 2-channel output and gradient checkpointing.
        >>> net = SwinAgeMapper(img_size=(96,96), in_channels=3, out_channels=2, use_checkpoint=True, spatial_dims=2)

        Returns
        -------
        None

        References
        ----------
        [1] Hatamizadeh et al., Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
            https://arxiv.org/abs/2201.01266

        """

        super().__init__()

        # Check input parameters and ensure they have the correct format and dimensions.
        # ensure_tuple_rep returns a copy of `input` with `spatial_dims` values by either shortened or duplicated input.
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)

        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        # for m, p in zip(img_size, patch_size):
        #     for i in range(5):
        #         if m % np.power(p, i + 1) != 0:
        #             raise ValueError("input image size (img_size) should be divisible by stage-wise image resolution.")

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        # if feature_size % 12 != 0:
        #     raise ValueError("feature_size should be divisible by 12.") # Don't know why this is necessary. 

        if resolution=='2mm':
            self.Upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        elif resolution=='1mm':
            self.Upsample = nn.Identity()
        else:
            print("ATTENTION! Resolution >>{}<< Not Supported!!!".format(resolution))

        self.swinViT = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample,
        )

        output_channels = feature_size * 2 ** len(depths)
        self.FullyConnected = nn.Sequential()
        # input_dimensions = 5 * 6 * 5 * output_channels
        input_dim1 = np.ceil(img_size[0]/(patch_size[0]*2**len(depths)))
        input_dim2 = np.ceil(img_size[1]/(patch_size[1]*2**len(depths)))
        input_dim3 = np.ceil(img_size[2]/(patch_size[2]*2**len(depths)))
        input_dimensions = int(input_dim1 * input_dim2 * input_dim3 * output_channels)

        self.FullyConnected.add_module(
            name = 'FullyConnected_3',
            module=nn.Linear(
                in_features=input_dimensions,
                out_features=96
            )
        )

        self.FullyConnected.add_module(
                name = 'Activation_3',
                module= get_act_layer(fully_connected_activation)
            )

        self.FullyConnected.add_module(
            name = 'FullyConnected_2',
            module=nn.Linear(
                in_features=96,
                out_features=32
            )
        )

        self.FullyConnected.add_module(
                name = 'Activation_2',
                module= get_act_layer(fully_connected_activation)
            )

        self.FullyConnected.add_module(
            name = 'FullyConnected_1',
            module=nn.Linear(
                in_features=32,
                out_features=1
            )
        )

        self.FullyConnected.add_module(
                name = 'LinearActivation',
                module= nn.Identity()
            )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.
        
        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (batch_size, in_channels, *img_size).
            
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_channels).
                
        """
        X = self.Upsample(X)
        X = self.swinViT(X)
        X = X.reshape(-1, prod(tensor(X.shape)[1:]))
        X = self.FullyConnected(X)
        return X


def window_partition(x: torch.Tensor,
                     window_size: tuple
                     ) -> torch.Tensor:
    """window partition operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

    Implenetation by: "Hatamizadeh et al., 
    Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
    <http:https://arxiv.org/pdf/2201.01266>"
    https://monai.io/research/swin-unetr

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (batch_size, in_channels, *img_size).
    window_size : tuple
        Size of the window.

    Returns
    -------
    torch.Tensor
        Output tensor of shape (batch_size, out_channels).

    """
    x_shape = x.size()
    if len(x_shape) == 5:
        b, d, h, w, c = x_shape
        x = x.view(
            b,
            d // window_size[0],
            window_size[0],
            h // window_size[1],
            window_size[1],
            w // window_size[2],
            window_size[2],
            c,
        )
        windows = (
            x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2], c)
        )
    elif len(x_shape) == 4:
        b, h, w, c = x.shape
        x = x.view(b, h // window_size[0], window_size[0], w // window_size[1], window_size[1], c)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0] * window_size[1], c)
    return windows


def window_reverse(windows: torch.Tensor, window_size: tuple, dims: list) -> torch.Tensor:
    """window reverse operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

    Implenetation by: "Hatamizadeh et al., 
    Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
    <http:https://arxiv.org/pdf/2201.01266>"
    https://monai.io/research/swin-unetr

    Parameters:
    ----------
    windows: torch.Tensor
        Input tensor of shape (batch_size * flatten sets volume, patch size* , feature space).
    window_size: tuple
        Size of the window.
    dims: list
        Dimensions of the input tensor before windowing. For a 3D input tensor, dims = [batch_size, depth, height, width].

    Returns:
    -------
    torch.Tensor
        Output tensor of shape (batch_size, tensor dimensions* , feature space).

    """
    if len(dims) == 4:
        b, d, h, w = dims
        x = windows.view(
            b,
            d // window_size[0],
            h // window_size[1],
            w // window_size[2],
            window_size[0],
            window_size[1],
            window_size[2],
            -1,
        )
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)

    elif len(dims) == 3:
        b, h, w = dims
        x = windows.view(b, h // window_size[0], w // window_size[1], window_size[0], window_size[1], -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


def get_window_size(x_size: tuple, 
                    window_size: tuple, 
                    shift_size: tuple
                    ) -> tuple:
    """Computing window size based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

    Implenetation by: "Hatamizadeh et al., 
    Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
    <http:https://arxiv.org/pdf/2201.01266>"
    https://monai.io/research/swin-unetr

    Parameters:
    ----------
    x_size: touple
        Input tensor size.
    window_size: Sequence[int]
        Local window size. By default, this is set to (7, 7, 7).
    shift_size: Sequence[int]
        Window shift size for the SW-MSA block. It is equal to to window_size // 2 = (3, 3, 3). If depth is 1, no SW-MSA blocks are used.

    Returns:
    -------
    tuple
        Touple corresponding to the window size.
    touple
        Touple corresponding to the window shifting size.
    """

    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class WindowAttention(nn.Module):
    """
    Window based multi-head self attention module with relative position bias based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    
    Implenetation by: "Hatamizadeh et al., 
    Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
    <http:https://arxiv.org/pdf/2201.01266>"
    https://monai.io/research/swin-unetr
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """
        Parameters:
        ----------
        dim: int
            Number of feature channels. This is based on the feature_size parameter in the Swin Transformer model. It is calculated as int(feature_size * 2 ** layer_index).
            Essentially, as the network depth increases, the number of features is doubled.
        num_heads: int
            Number of attention heads in the multi-head self-attention block.
        window_size: Sequence[int]
            Local window size. By default, this is set to (7, 7, 7).
        qkv_bias: bool
            Add a learnable bias to query, key, value. Default: False (but True in the original paper).
        attn_drop: float
            Attention dropout rate. By default, this is set to 0.0.
        proj_drop: float
            Dropout rate of output. By default, this is set to 0.0.

        Returns:
        -------
        None

        """

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        mesh_args = torch.meshgrid.__kwdefaults__

        # Start by defining the relative position bias table. This is a learnable parameter. 
        # The relative position bias is used to model the relative position between the tokens in the window.
            
        if len(self.window_size) == 3:
            # First, we define the relative position bias table as a learnable parameter.
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1),
                    num_heads,
                )
            )
            # Then, initialise coodinates along the depth, height and width dimensions. Then, we stack the coordinates to create a 3D grid.
            # We then flatten the grid to create a 3D grid of shape (3, window_size[0] * window_size[1] * window_size[2]).
            coords_d = torch.arange(self.window_size[0])
            coords_h = torch.arange(self.window_size[1])
            coords_w = torch.arange(self.window_size[2])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
            else:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)

            # We then compute the relative coordinates between each pair of points by subtracting the coordinates of each token from the coordinates of all other tokens.
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            # The code then transposes the dimensions of relative_coords to (1, 2, 0) and makes a contiguous copy of the tensor using contiguous().
            # This rearranges the tensor so that each element of the first two dimensions corresponds to a pair of coordinates, and each element of the third dimension corresponds to a different axis.
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            
            # We then adjust the relative coordinates based on the window size.
            # We first ensure that the coordinates are shifted to be relative to the center of the window. 
            # Then, we multiply the coordinates by the window size to get a unique index for each relative position, which is then used to index the relative position bias table.
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1
            relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1

        elif len(self.window_size) == 2:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
            )
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
            else:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        # The relative position index is calculated by summing the relative coordinates along the third dimension.
        # This gives a unique index for each relative position.
        relative_position_index = relative_coords.sum(-1)
        # We then register the relative position index as a buffer. This makes it a fixed tensor that is not updated during training, but can be used for inference.
        self.register_buffer("relative_position_index", relative_position_index)

        # We then define the query, key and value projections, the attention dropout and the projection dropout. 
        # The query, key and value projections are linear layers that map the input tensor to a tensor with the same number of channels as the input tensor.
        # The attention dropout is a dropout layer that is applied to the attention scores before they are passed to the softmax function.
        # The projection dropout is a dropout layer that is applied to the output of the projection layer.
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # Finally, we initialise the relative position bias table using a truncated normal distribution with a standard deviation of 0.02, and define the softmax function
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward function.

        Parameters:
        ----------
        x: torch.Tensor
            Input tensor with shape (batch_size, num_patches, dim).
        mask: torch.Tensor
            Attention mask with shape (batch_size, num_patches).

        Returns:
        -------
        torch.Tensor
            Output tensor with shape (batch_size, num_patches, dim).
        
        """
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        # Compute the relative position bias using the relative position bias table from the constructor.
        # Then, reshape it to match the shape of the attention scores and add it to the attention scores.
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:n, :n].reshape(-1)  # type: ignore
        ].reshape(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0) # include the relative position bias
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0) # include the shifted attention mask if block is a SW-MSA block.
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn).to(v.dtype)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer block based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

    Implenetation by: "Hatamizadeh et al., 
    Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
    <http:https://arxiv.org/pdf/2201.01266>"
    https://monai.io/research/swin-unetr

    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        shift_size: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: str = "GELU",
        norm_layer: type[LayerNorm] = nn.LayerNorm,
        use_checkpoint: bool = False,
    ) -> None:
        """

        Parameters:
        ----------
        dim: int
            Number of feature channels. This is based on the feature_size parameter in the Swin Transformer model. It is calculated as int(feature_size * 2 ** layer_index).
            Essentially, as the network depth increases, the number of features is doubled.
        num_heads: int
            Number of attention heads in the multi-head self-attention block.
        window_size: Sequence[int]
            Local window size. By default, this is set to (7, 7, 7).
        shift_size: Sequence[int]
            Window shift size for the SW-MSA block. It is equal to to window_size // 2 = (3, 3, 3). If depth is 1, no SW-MSA blocks are used.
        mlp_ratio: float
            Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias: bool
            Add a learnable bias to query, key, value. Default: False (but True in the original paper).
        drop: float
            Dropout rate. Default: 0.
        attn_drop: float
            Attention dropout rate. Default: 0.
        drop_path: float
            Stochastic depth rate for each layer. By default, these are always set to 0.
        act_layer: str
            Activation layer type. Default: `nn.GELU`.
        norm_layer: type[LayerNorm]
            Normalization layer. Default: `nn.LayerNorm`.
        use_checkpoint: bool
            Whether to use use gradient checkpointing for reduced memory usage. Default: False.

        Returns:
        -------
        None

        """

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(hidden_size=dim, mlp_dim=mlp_hidden_dim, act=act_layer, dropout_rate=drop, dropout_mode="swin")

    def forward_part1(self, x : torch.Tensor, mask_matrix: torch.Tensor) -> torch.Tensor:
        """
        Forward function for the first part of the Swin Transformer block, corresponding to the Multi-Head Self-Attention block.
        
        Parameters:
        ----------
        x: torch.Tensor
            Input tensor of shape (batch_size, num_patches, dim).
        mask_matrix: torch.Tensor
            Mask matrix of shape (batch_size, num_patches, num_patches).
            
        Returns:
        -------
        torch.Tensor
            Output tensor of shape (batch_size, num_patches, dim).
        
        """
        x_shape = x.size()
        x = self.norm1(x)
        # Determine the amount of padding necessary for the patches volume to be divisible by the window size.
        if len(x_shape) == 5:
            b, d, h, w, c = x.shape
            window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
            pad_l = pad_t = pad_d0 = 0
            pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
            pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
            pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
            _, dp, hp, wp, _ = x.shape
            dims = [b, dp, hp, wp]

        elif len(x_shape) == 4:
            b, h, w, c = x.shape
            window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
            pad_l = pad_t = 0
            pad_b = (window_size[0] - h % window_size[0]) % window_size[0]
            pad_r = (window_size[1] - w % window_size[1]) % window_size[1]
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, hp, wp, _ = x.shape
            dims = [b, hp, wp]

        # If block is of type SW-MSA, perform the shift operation and assign the attention mask. If not, do not shift and do not assign the attention mask.
        if any(i > 0 for i in shift_size):
            if len(x_shape) == 5:
                shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            elif len(x_shape) == 4:
                shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
            
        x_windows = window_partition(shifted_x, window_size) # Create windows of the input tensor.
        attn_windows = self.attn(x_windows, mask=attn_mask) # Perform the attention operation on the windows.
        attn_windows = attn_windows.view(-1, *(window_size + (c,))) # Reshape the attention windows to the original shape.
        shifted_x = window_reverse(attn_windows, window_size, dims)

        # If the block is of type SW-MSA, perform the reverse shift operation. If not, do not perform the reverse shift operation.
        if any(i > 0 for i in shift_size):
            if len(x_shape) == 5:
                x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
            elif len(x_shape) == 4:
                x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        # Remove the padding from the input tensor. This is necessary because the padding is only used to make the input tensor divisible by the window size. 
        if len(x_shape) == 5:
            if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
                x = x[:, :d, :h, :w, :].contiguous()
        elif len(x_shape) == 4:
            if pad_r > 0 or pad_b > 0:
                x = x[:, :h, :w, :].contiguous()

        return x

    def forward_part2(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function for the second part of the Swin Transformer block, corresponding to the Feed Forward block.

        Parameters:
        ----------
        x: torch.Tensor
            Input tensor of shape (batch_size, num_patches, dim).

        Returns:
        -------
        torch.Tensor
            Output tensor of shape (batch_size, num_patches, dim).

        """

        return self.drop_path(self.mlp(self.norm2(x)))

    def load_from(self, weights: torch.Tensor, n_block: str, layer:str) -> None:
        """
        Load weights from the specified weights.

        Parameters:
        ----------
        weights: torch.Tensor
            Weights.
        n_block: str
            Block number.
        layer: str
            Layer number.

        Returns:
        -------
        None
        
        """
        root = f"module.{layer}.0.blocks.{n_block}."
        block_names = [
            "norm1.weight",
            "norm1.bias",
            "attn.relative_position_bias_table",
            "attn.relative_position_index",
            "attn.qkv.weight",
            "attn.qkv.bias",
            "attn.proj.weight",
            "attn.proj.bias",
            "norm2.weight",
            "norm2.bias",
            "mlp.fc1.weight",
            "mlp.fc1.bias",
            "mlp.fc2.weight",
            "mlp.fc2.bias",
        ]
        with torch.no_grad():
            self.norm1.weight.copy_(weights["state_dict"][root + block_names[0]])
            self.norm1.bias.copy_(weights["state_dict"][root + block_names[1]])
            self.attn.relative_position_bias_table.copy_(weights["state_dict"][root + block_names[2]])
            self.attn.relative_position_index.copy_(weights["state_dict"][root + block_names[3]])  # type: ignore
            self.attn.qkv.weight.copy_(weights["state_dict"][root + block_names[4]])
            self.attn.qkv.bias.copy_(weights["state_dict"][root + block_names[5]])
            self.attn.proj.weight.copy_(weights["state_dict"][root + block_names[6]])
            self.attn.proj.bias.copy_(weights["state_dict"][root + block_names[7]])
            self.norm2.weight.copy_(weights["state_dict"][root + block_names[8]])
            self.norm2.bias.copy_(weights["state_dict"][root + block_names[9]])
            self.mlp.linear1.weight.copy_(weights["state_dict"][root + block_names[10]])
            self.mlp.linear1.bias.copy_(weights["state_dict"][root + block_names[11]])
            self.mlp.linear2.weight.copy_(weights["state_dict"][root + block_names[12]])
            self.mlp.linear2.bias.copy_(weights["state_dict"][root + block_names[13]])

    def forward(self, x: torch.Tensor, mask_matrix: torch.Tensor) -> torch.Tensor:
        """
        Global forward function.

        Parameters:
        ----------
        x: torch.Tensor
            Input tensor of shape (batch_size, num_patches, dim).
        mask_matrix: torch.Tensor
            Mask matrix.
        
        Returns:
        -------
        torch.Tensor
            Output tensor of shape (batch_size, num_patches, dim).

        """
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)
        return x


class PatchMergingV2(nn.Module):
    """
    Patch merging layer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

    Implenetation by: "Hatamizadeh et al., 
    Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
    <http:https://arxiv.org/pdf/2201.01266>"
    https://monai.io/research/swin-unetr

    """

    def __init__(self, dim: int, norm_layer: type[LayerNorm] = nn.LayerNorm, spatial_dims: int = 3) -> None:
        """
        Parameters
        ----------
        dim: int
            Number of feature channels. This is based on the feature_size parameter in the Swin Transformer model. It is calculated as int(feature_size * 2 ** layer_index).
            Essentially, as the network depth increases, the number of features is doubled.
        norm_layer: type of nn.LayerNorm
            Normalization layer. Default: `nn.LayerNorm`.
        spatial_dims: int
            Number of spatial dims, corresponding to the length of the window_size parameter in the Swin Transformer model. Default: 3.
        """

        super().__init__()
        self.dim = dim
        if spatial_dims == 3:
            self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(8 * dim)
        elif spatial_dims == 2:
            self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform patch merging operation.

        Parameters
        ----------
        x: torch.Tensor
            input tensor with shape (batch, spatial_dims, spatial_dims, spatial_dims, channels).

        Returns
        -------
        torch.Tensor
            output tensor with shape (batch, spatial_dims / 2, spatial_dims / 2, spatial_dims / 2, channels * 2).

        """
        x_shape = x.size()
        if len(x_shape) == 5:
            b, d, h, w, c = x_shape
            pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
            x = torch.cat(
                [x[:, i::2, j::2, k::2, :] for i, j, k in itertools.product(range(2), range(2), range(2))], -1
            )

        elif len(x_shape) == 4:
            b, h, w, c = x_shape
            pad_input = (h % 2 == 1) or (w % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
            x = torch.cat([x[:, j::2, i::2, :] for i, j in itertools.product(range(2), range(2))], -1)

        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchMerging(PatchMergingV2):
    """The `PatchMerging` module previously defined in v0.9.0.
    
    Inherits constructor from `PatchMergingV2` to keep the same API.

    Implenetation by: "Hatamizadeh et al., 
    Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
    <http:https://arxiv.org/pdf/2201.01266>"
    https://monai.io/research/swin-unetr
    """

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        """
        Parameters
        ----------
        x: torch.Tensor
            Input tensor with shape (B, C, H, W) or (B, C, D, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor with shape (B, C, H/2, W/2) or (B, C, D/2, H/2, W/2).

        """
        x_shape = x.size()
        if len(x_shape) == 4:
            return super().forward(x)
        if len(x_shape) != 5:
            raise ValueError(f"expecting 5D x, got {x.shape}.")
        b, d, h, w, c = x_shape
        pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))

        # Split the input tensor into 8 groups of patches (8 * C channels dimensional features)
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 0::2, 1::2, :]
        x5 = x[:, 0::2, 1::2, 0::2, :]
        x6 = x[:, 0::2, 0::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        # Concatenate each set of neighboring groups of patches, resulting in 8 groups of patches (8 * C channels dimensional concatenated features)
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
        # Apply layer normalisation
        x = self.norm(x)
        # Use a linear layer on the 8C-dimensional concatenated features to reduce the dimensionality to 2C
        x = self.reduction(x)
        return x


MERGING_MODE = {"merging": PatchMerging, "mergingv2": PatchMergingV2}


def compute_mask(dims: list[int],
                window_size: tuple[int, int, int],
                shift_size: tuple[int, int, int],
                device: torch.device
                ) -> torch.Tensor:
    """Computing region masks based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

    This code creates a masking mechanism which limits self-attention to sub-windows following window shifting.
    The mask is created by creating a 3D tensor of shape (window_size, window_size, window_size) and then shifting it
    by the shift_size. The mask is then flattened and the resulting tensor is used as a mask for the self-attention
    mechanism.

    An intuitive example of this is shown below:
    https://amaarora.github.io/posts/2022-07-04-swintransformerv1.html
    
    Implenetation by: "Hatamizadeh et al., 
    Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
    <http:https://arxiv.org/pdf/2201.01266>"
    https://monai.io/research/swin-unetr

    Parameters
    ----------
    dims: list
        Dimension values corresponding to the padded patches volume.
    window_size: tuple
        Local window size. By default, this is set to (7, 7, 7).
    shift_size: tuple
        Shift size for the local window. For a window size of (7, 7, 7), the shift size is set to (3, 3, 3), ie. window_size // 2.
    device: torch.device
        Device on which the tensor is allocated.

    Returns
    -------
    attn_mask: torch.Tensor
        Region mask for the local window, corresponging to relative posotion bias. 
    """

    cnt = 0

    if len(dims) == 3:
        d, h, w = dims
        img_mask = torch.zeros((1, d, h, w, 1), device=device)
        for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
            for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
                for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1


    elif len(dims) == 2:
        h, w = dims
        img_mask = torch.zeros((1, h, w, 1), device=device)
        for h in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
            for w in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
                img_mask[:, h, w, :] = cnt
                cnt += 1

    mask_windows = window_partition(img_mask, window_size) # partition patches space into a sets volume of non-overlapping local windows (eg. 84x98x84 patches, if window_size=7x7x7 -> 12x14x12 windows) 
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask


class BasicLayer(nn.Module):
    """
    Basic Swin Transformer layer in one stage based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

    Implenetation by: "Hatamizadeh et al., 
    Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
    <http:https://arxiv.org/pdf/2201.01266>"
    https://monai.io/research/swin-unetr

    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Sequence[int],
        drop_path: list,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: type[LayerNorm] = nn.LayerNorm,
        downsample: nn.Module | None = None,
        use_checkpoint: bool = False,
    ) -> None:
        
        """ 
        Parameters:
        ----------
        dim: int
            Number of feature channels. This is based on the feature_size parameter in the Swin Transformer model. It is calculated as int(feature_size * 2 ** layer_index).
            Essentially, as the network depth increases, the number of features is doubled.
        depth: int
            Number of layers in each stage. Default is 2, ie 2 transformer blocks in each stage.
        num_heads: int
            Number of attention heads in the multi-head self-attention block.
        window_size: tuple of int
            Local window size. By default, this is set to (7, 7, 7).
        drop_path: list of float
            Stochastic depth rate for each layer. By default, these are always set to 0.
        mlp_ratio: float
            Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias: bool
            Add a learnable bias to query, key, value. Default: False (but True in the original paper).
        drop: float
            Dropout rate. Default: 0.
        attn_drop: float
            Attention dropout rate. Default: 0.
        norm_layer: type of nn.LayerNorm
            Normalization layer. Default: `nn.LayerNorm`.
        downsample: nn.Module   
            Downsample layer at the end of the layer. Default: None. Selected from PatchMerging and PatchMergingV2.
        use_checkpoint: bool
            Whether to use use gradient checkpointing for reduced memory usage. Default: False.

        Returns:
        -------
        None

        """

        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size) # shift for SW-MSA; equal to window_size // 2 = (3, 3, 3)
        self.no_shift = tuple(0 for i in window_size) # no shift condition if depth is even
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=self.window_size,
                    shift_size=self.no_shift if (i % 2 == 0) else self.shift_size, # if depth 1, no SW-MSA blocks are used
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                )
                for i in range(depth)
            ]
        )
        self.downsample = downsample
        if callable(self.downsample):
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, spatial_dims=len(self.window_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function.

        Parameters:
        ----------
        x: torch.Tensor
            Input tensor.

        Returns:
        -------
        torch.Tensor
            Result tensor.

        """
        x_shape = x.size()
        if len(x_shape) == 5:
            b, c, d, h, w = x_shape
            window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
            x = rearrange(x, "b c d h w -> b d h w c") # move the channel dimension to the end
            # We pad the patches to make sure that the patches are divisible by the window size
            dp = int(np.ceil(d / window_size[0])) * window_size[0]
            hp = int(np.ceil(h / window_size[1])) * window_size[1]
            wp = int(np.ceil(w / window_size[2])) * window_size[2]
            attn_mask = compute_mask([dp, hp, wp], window_size, shift_size, x.device) # create the shifted-window attention mask
            for blk in self.blocks:
                x = blk(x, attn_mask)
            x = x.view(b, d, h, w, -1)
            if self.downsample is not None:
                x = self.downsample(x)
            x = rearrange(x, "b d h w c -> b c d h w")

        elif len(x_shape) == 4:
            b, c, h, w = x_shape
            window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
            x = rearrange(x, "b c h w -> b h w c")
            hp = int(np.ceil(h / window_size[0])) * window_size[0]
            wp = int(np.ceil(w / window_size[1])) * window_size[1]
            attn_mask = compute_mask([hp, wp], window_size, shift_size, x.device)
            for blk in self.blocks:
                x = blk(x, attn_mask)
            x = x.view(b, h, w, -1)
            if self.downsample is not None:
                x = self.downsample(x)
            x = rearrange(x, "b h w c -> b c h w")
        return x


class SwinTransformer(nn.Module):
    """ Swin Transformer for hierarchical vision transformer using shifted windows. 

    Swin Transformer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

    Implenetation by: "Hatamizadeh et al., 
    Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
    <http:https://arxiv.org/pdf/2201.01266>"
    https://monai.io/research/swin-unetr

    """

    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        window_size: Sequence[int],
        patch_size: Sequence[int],
        depths: Sequence[int],
        num_heads: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: type[LayerNorm] = nn.LayerNorm,
        patch_norm: bool = False,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample: str="merging",
    ) -> None:
        
        """
        Parameters
        ----------
        in_chans: int
            Dimension of input channels.
        embed_dim: int
            Number of linear projection output channels. This corresponds to the arbitrary internal dimension in which the inputs are projected to before the attention operation.
        window_size: Sequence[int]
            Local window size. By default, this is set to (7, 7, 7). The windows contains a a group of (2, 2, 2) voxel patches. Thus, a windows is of size (14, 14, 14) voxels. 
        patch_size: Sequence[int]
            Patch size, by default 2. This corresponds to the size of the patches that are extracted from the input image.
        depths: Sequence[int]
            Number of layers in each stage, by default (2, 2, 2, 2). This corresponds to the number of SwinTransformerBlocks at each depth stage. 
        num_heads: Sequence[int]
            Number of attention heads in the multi-head self-attention block, by default (3, 6, 12, 24). This corresponds to the number of attention heads at each depth stage.
        mlp_ratio: float
            Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias: bool
            Add a learnable bias to query, key, value. Default: True.
        drop_rate: float
            Dropout rate. Default: 0.
        attn_drop_rate: float
            Attention dropout rate. Default: 0.
        drop_path_rate: float
            Stochastic depth rate. Default: 0.
        norm_layer: type[LayerNorm]
            Normalization layer. Default: `nn.LayerNorm`.
        patch_norm: bool
            Add normalization after patch embedding. Default: True.
        use_checkpoint: bool
            Use gradient checkpointing for reduced memory usage. Default: False.
        spatial_dims: int
            Number of spatial dimensions. Default: 3.
        downsample: str
            Module used for downsampling, available options are `"mergingv2"`, `"merging"` and a user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
            The default is currently `"merging"` (the original version defined in v0.9.0).

        Retunrs
        -------
        None

        """

        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.window_size = window_size
        self.patch_size = patch_size

        # We define the patch embedding block based on the patch size and the input image size.
        # The patch embedding block is a convolutional layer that extracts patches from the input image.
        # A linear embedding layer is applied to each patch to project it to a vector of size embed_dim, which is the input to the transformer.
        # Unlike ViT patch embedding block: (1) input is padded to satisfy window size requirements (2) normalized if
        # specified (3) position embedding is not used.

        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,  # type: ignore
            spatial_dims=spatial_dims,
        )

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] # dropout rates at each layer
        
        if self.num_layers == 3:
            self.layers1 = nn.ModuleList()
            self.layers2 = nn.ModuleList()
            self.layers3 = nn.ModuleList()
        elif self.num_layers == 4:     
            self.layers1 = nn.ModuleList()
            self.layers2 = nn.ModuleList()
            self.layers3 = nn.ModuleList()
            self.layers4 = nn.ModuleList()
        else:
            raise ValueError(f"Number of layers {self.num_layers} not supported. Please use 3 or 4 layers.")
        
        down_sample_mod = look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=self.window_size,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                downsample=down_sample_mod,
                use_checkpoint=use_checkpoint,
            )
            if i_layer == 0:
                self.layers1.append(layer)
            elif i_layer == 1:
                self.layers2.append(layer)
            elif i_layer == 2:
                self.layers3.append(layer)
            elif i_layer == 3:
                self.layers4.append(layer)

        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

    def proj_out(self,
                x: torch.Tensor,
                normalize: bool=False
                ) -> torch.Tensor:
        """
        Project the output of the transformer to the desired output size.
        Parameters
        ----------
        x: torch.Tensor
            Input tensor.
        normalize: bool
            Whether to normalize the output. Default: False.
        
        Returns
        -------
        torch.Tensor
            Output tensor.
        
        """
        if normalize:
            x_shape = x.size()
            if len(x_shape) == 5:
                n, ch, d, h, w = x_shape
                x = rearrange(x, "n c d h w -> n d h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n d h w c -> n c d h w")
            elif len(x_shape) == 4:
                n, ch, h, w = x_shape
                x = rearrange(x, "n c h w -> n h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n h w c -> n c h w")
        return x

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Parameters
        ----------
        X: torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
                        
        """

        X = self.patch_embed(X)
        X = self.pos_drop(X)

        if self.num_layers == 3:
            X = self.layers1[0](X.contiguous())
            X = self.layers2[0](X.contiguous())
            X = self.layers3[0](X.contiguous())
        elif self.num_layers == 4:
            X = self.layers1[0](X.contiguous())
            X = self.layers2[0](X.contiguous())
            X = self.layers3[0](X.contiguous())
            X = self.layers4[0](X.contiguous())
        else:
            raise NotImplementedError("Only 3 or 4 layers are supported.")

        return X