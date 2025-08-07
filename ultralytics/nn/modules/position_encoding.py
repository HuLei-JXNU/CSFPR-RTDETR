import math
import functools
from typing import Tuple, Union
import warnings
import torch
from torch import Tensor, nn
from typing import Callable, List, Optional


class ConvNormActivation(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: int = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
        conv_layer: Callable[..., torch.nn.Module] = torch.nn.Conv2d,
    ) -> None:

        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None

        layers = [
            conv_layer(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels))

        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers)
        self.out_channels = out_channels

        if self.__class__ == ConvNormActivation:
            warnings.warn(
                "Don't use ConvNormActivation directly, please use Conv2dNormActivation and Conv3dNormActivation instead."
            )


class Conv2dNormActivation(ConvNormActivation):
    """
    Configurable block used for Convolution2d-Normalization-Activation blocks.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the Convolution-Normalization-Activation block
        kernel_size: (int, optional): Size of the convolving kernel. Default: 3
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: None, in which case it will calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer wont be used. Default: ``torch.nn.BatchNorm2d``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer wont be used. Default: ``torch.nn.ReLU``
        dilation (int): Spacing between kernel elements. Default: 1
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
        bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.

    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: int = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
    ) -> None:

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            norm_layer,
            activation_layer,
            dilation,
            inplace,
            bias,
            torch.nn.Conv2d,
        )

class PositionEmbeddingSine(nn.Module):
    """Sinusoidal position embedding used in DETR model. See `End-to-End Object Detection
    with Transformers <https://arxiv.org/pdf/2005.12872>`_ for more details.

    :param num_pos_feats: The feature dimension for each position along x-axis or y-axis.
        The final returned dimension for each position is 2 times of the input value,
        defaults to 64
    :param temperature: The temperature used for scaling the position embedding, defaults to 10000
    :param normalize: Whether to normalize the position embedding, defaults to False
    :param scale: A scale factor that scales the position embedding, which is used only when
        `normalize` is True, defaults to 2*math.pi
    :param eps: A value added to the denominator for numerical stability, defaults to 1e-6
    :param offset: An offset added to embed, defaults to 0.0
    """
    def __init__(
        self,
        num_pos_feats=64,
        temperature: Union[int, Tuple[int, int]] = 10000,
        normalize=False,
        scale=2 * math.pi,
        eps=1e-6,
        offset=0.0,
    ):
        super().__init__()
        assert isinstance(temperature, int) or len(temperature) == 2, \
            "Only support (t_x, t_y) or an integer t for temperature"

        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def get_dim_t(self, device: torch.device):
        if isinstance(self.temperature, int):
            dim_t = get_dim_t(self.num_pos_feats, self.temperature, device)
            return dim_t, dim_t
        return (get_dim_t(self.num_pos_feats, t, device) for t in self.temperature)

    def forward(self, mask: Tensor):
        not_mask = (~mask).int()  # onnx export does not support cumsum on bool tensor
        y_embed = not_mask.cumsum(1)
        x_embed = not_mask.cumsum(2)
        if self.normalize:
            y_embed = (y_embed + self.offset) / (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / (x_embed[:, :, -1:] + self.eps) * self.scale
        else:
            # RT-DETR uses unnormalized encoding with index from 0
            y_embed = y_embed + self.offset
            x_embed = x_embed + self.offset

        dim_tx, dim_ty = self.get_dim_t(mask.device)

        pos_x = x_embed.unsqueeze(-1) / dim_tx
        pos_y = y_embed.unsqueeze(-1) / dim_ty
        pos_x = torch.stack((pos_x.sin(), pos_x.cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y.sin(), pos_y.cos()), dim=-1).flatten(-2)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """Absolute pos embedding, learned."""
    def __init__(self, num_embeddings: int = 50, num_pos_feats: int = 256):
        super().__init__()
        self.row_embed = nn.Embedding(num_embeddings, num_pos_feats)
        self.col_embed = nn.Embedding(num_embeddings, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, mask: Tensor):
        h, w = mask.shape[-2:]
        i = torch.arange(w, device=mask.device)
        j = torch.arange(h, device=mask.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = (
            torch.cat(
                [
                    x_emb.unsqueeze(0).repeat(h, 1, 1),
                    y_emb.unsqueeze(1).repeat(1, w, 1),
                ],
                dim=-1,
            ).permute(2, 0, 1).unsqueeze(0).repeat(mask.shape[0], 1, 1, 1)
        )
        return pos


@functools.lru_cache  # use lru_cache to avoid redundant calculation for dim_t
def get_dim_t(num_pos_feats: int, temperature: int, device: torch.device):
    dim_t = torch.arange(num_pos_feats // 2, dtype=torch.float32, device=device)
    dim_t = temperature**(dim_t * 2 / num_pos_feats)
    return dim_t  # (0, 2, 4, ..., ⌊n/2⌋*2)

def exchange_xy_fn(pos_res):
    index = torch.cat([
        torch.arange(1, -1, -1, device=pos_res.device),
        torch.arange(2, pos_res.shape[-2], device=pos_res.device),
    ])
    pos_res = torch.index_select(pos_res, -2, index)
    return pos_res

def get_sine_pos_embed(
    pos_tensor: Tensor,
    num_pos_feats: int = 128,
    temperature: int = 10000,
    scale: float = 2 * math.pi,
    exchange_xy: bool = True,
) -> Tensor:
    """Generate sine position embedding for a position tensor

    :param pos_tensor: shape as (..., 2*n).
    :param num_pos_feats: projected shape for each float in the tensor, defaults to 128
    :param temperature: the temperature used for scaling the position embedding, defaults to 10000
    :param exchange_xy: exchange pos x and pos. For example,
        input tensor is [x, y], the results will be [pos(y), pos(x)], defaults to True
    :return: position embedding with shape (None, n * num_pos_feats)
    """
    dim_t = get_dim_t(num_pos_feats, temperature, pos_tensor.device)

    pos_res = pos_tensor.unsqueeze(-1) * scale / dim_t
    pos_res = torch.stack((pos_res.sin(), pos_res.cos()), dim=-1).flatten(-2)
    if exchange_xy:
        pos_res = exchange_xy_fn(pos_res)
    pos_res = pos_res.flatten(-2)
    return pos_res