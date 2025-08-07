# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Transformer modules."""
from torch import Tensor, nn
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_
from ultralytics.nn.modules.position_encoding import Conv2dNormActivation,get_sine_pos_embed
from .conv import Conv
from .utils import _get_clones, inverse_sigmoid, multi_scale_deformable_attn_pytorch

__all__ = ('TransformerEncoderLayer', 'TransformerLayer', 'TransformerBlock', 'MLPBlock', 'LayerNorm2d', 'AIFI',
           'DeformableTransformerDecoder', 'DeformableTransformerDecoderLayer', 'MSDeformAttn', 'MLP')


class TransformerEncoderLayer(nn.Module):
    """Defines a single layer of the transformer encoder."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0, act=nn.GELU(), normalize_before=False):
        """Initialize the TransformerEncoderLayer with specified parameters."""
        super().__init__()
        from ...utils.torch_utils import TORCH_1_9
        if not TORCH_1_9:
            raise ModuleNotFoundError(
                'TransformerEncoderLayer() requires torch>=1.9 to use nn.MultiheadAttention(batch_first=True).')
        self.ma = nn.MultiheadAttention(c1, num_heads, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)

        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.act = act
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor, pos=None):
        """Add position embeddings to the tensor if provided."""
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Performs forward pass with post-normalization."""
        q = k = self.with_pos_embed(src, pos)
        src2 = self.ma(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)

    def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Performs forward pass with pre-normalization."""
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.ma(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src2))))
        return src + self.dropout2(src2)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Forward propagates the input through the encoder module."""
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class AIFI(TransformerEncoderLayer):
    """Defines the AIFI transformer layer."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0, act=nn.GELU(), normalize_before=False):
        """Initialize the AIFI instance with specified parameters."""
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)

    def forward(self, x):
        """Forward pass for the AIFI transformer layer."""
        c, h, w = x.shape[1:]
        pos_embed = self.build_2d_sincos_position_embedding(w, h, c)
        # Flatten [B, C, H, W] to [B, HxW, C]
        x = super().forward(x.flatten(2).permute(0, 2, 1), pos=pos_embed.to(device=x.device, dtype=x.dtype))
        return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        """Builds 2D sine-cosine position embedding."""
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], 1)[None]

class TransformerLayer(nn.Module):
    """Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)."""

    def __init__(self, c, num_heads):
        """Initializes a self-attention mechanism using linear transformations and multi-head attention."""
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        """Apply a transformer block to the input x and return the output."""
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        return self.fc2(self.fc1(x)) + x


class TransformerBlock(nn.Module):
    """Vision Transformer https://arxiv.org/abs/2010.11929."""

    def __init__(self, c1, c2, num_heads, num_layers):
        """Initialize a Transformer module with position embedding and specified number of heads and layers."""
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        """Forward propagates the input through the bottleneck module."""
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


class MLPBlock(nn.Module):
    """Implements a single block of a multi-layer perceptron."""

    def __init__(self, embedding_dim, mlp_dim, act=nn.GELU):
        """Initialize the MLPBlock with specified embedding dimension, MLP dimension, and activation function."""
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the MLPBlock."""
        return self.lin2(self.act(self.lin1(x)))


class MLP(nn.Module):
    """Implements a simple multi-layer perceptron (also called FFN)."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        """Initialize the MLP with specified input, hidden, output dimensions and number of layers."""
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        """Forward pass for the entire MLP."""
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class LayerNorm2d(nn.Module):
    """
    2D Layer Normalization module inspired by Detectron2 and ConvNeXt implementations.

    Original implementations in
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py
    and
    https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py.
    """

    def __init__(self, num_channels, eps=1e-6):
        """Initialize LayerNorm2d with the given parameters."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        """Perform forward pass for 2D layer normalization."""
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class MSDeformAttn(nn.Module):
    """
    Multi-Scale Deformable Attention Module based on Deformable-DETR and PaddleDetection implementations.

    https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py
    """

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """Initialize MSDeformAttn with the given parameters."""
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f'd_model must be divisible by n_heads, but got {d_model} and {n_heads}')
        _d_per_head = d_model // n_heads
        # Better to set _d_per_head to a power of 2 which is more efficient in a CUDA implementation
        assert _d_per_head * n_heads == d_model, '`d_model` must be divisible by `n_heads`'

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        """Reset module parameters."""
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(
            1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, refer_bbox, value, value_shapes, value_mask=None):
        """
        Perform forward pass for multiscale deformable attention.

        https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py

        Args:
            query (torch.Tensor): [bs, query_length, C]
            refer_bbox (torch.Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (torch.Tensor): [bs, value_length, C]
            value_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, len_q = query.shape[:2]
        len_v = value.shape[1]
        assert sum(s[0] * s[1] for s in value_shapes) == len_v

        value = self.value_proj(value)
        if value_mask is not None:
            value = value.masked_fill(value_mask[..., None], float(0))
        value = value.view(bs, len_v, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(bs, len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(bs, len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(bs, len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        num_points = refer_bbox.shape[-1]
        if num_points == 2:
            offset_normalizer = torch.as_tensor(value_shapes, dtype=query.dtype, device=query.device).flip(-1)
            add = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            sampling_locations = refer_bbox[:, :, None, :, None, :] + add
        elif num_points == 4:
            add = sampling_offsets / self.n_points * refer_bbox[:, :, None, :, None, 2:] * 0.5
            sampling_locations = refer_bbox[:, :, None, :, None, :2] + add
        else:
            raise ValueError(f'Last dim of reference_points must be 2 or 4, but got {num_points}.')
        output = multi_scale_deformable_attn_pytorch(value, value_shapes, sampling_locations, attention_weights)
        return self.output_proj(output)


class DeformableTransformerDecoderLayer(nn.Module):
    """
    Deformable Transformer Decoder Layer inspired by PaddleDetection and Deformable-DETR implementations.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/deformable_transformer.py
    """

    def __init__(self, d_model=256, n_heads=8, d_ffn=1024, dropout=0., act=nn.ReLU(), n_levels=4, n_points=4):
        """Initialize the DeformableTransformerDecoderLayer with the given parameters."""
        super().__init__()
        self.num_heads = n_heads
        # Self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.act = act
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        """Add positional embeddings to the input tensor, if provided."""
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        """Perform forward pass through the Feed-Forward Network part of the layer."""
        tgt2 = self.linear2(self.dropout3(self.act(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        return self.norm3(tgt)

    def forward(self, embed, refer_bbox, feats, shapes, padding_mask=None, attn_mask=None, query_pos=None):
        """Perform the forward pass through the entire decoder layer."""

        # Self attention
        q = k = self.with_pos_embed(embed, query_pos)
        tgt = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), embed.transpose(0, 1),
                             attn_mask=attn_mask)[0].transpose(0, 1)
        embed = embed + self.dropout1(tgt)
        embed = self.norm1(embed)

        # Cross attention
        tgt = self.cross_attn(self.with_pos_embed(embed, query_pos), refer_bbox.unsqueeze(2), feats, shapes,
                              padding_mask)
        embed = embed + self.dropout2(tgt)
        embed = self.norm2(embed)

        # FFN
        return self.forward_ffn(embed)


class DeformableTransformerDecoder(nn.Module):
    """
    Implementation of Deformable Transformer Decoder based on PaddleDetection.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    """

    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        """Initialize the DeformableTransformerDecoder with the given parameters."""
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(
            self,
            embed,  # decoder embeddings
            refer_bbox,  # anchor
            feats,  # image features
            shapes,  # feature shapes
            bbox_head,
            score_head,
            pos_mlp,
            attn_mask=None,
            padding_mask=None):
        """Perform the forward pass through the entire decoder."""
        output = embed
        dec_bboxes = []
        dec_cls = []
        last_refined_bbox = None
        refer_bbox = refer_bbox.sigmoid()
        for i, layer in enumerate(self.layers):
            output = layer(output, refer_bbox, feats, shapes, padding_mask, attn_mask, pos_mlp(refer_bbox))

            bbox = bbox_head[i](output)
            refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox))

            if self.training:
                dec_cls.append(score_head[i](output))
                if i == 0:
                    dec_bboxes.append(refined_bbox)
                else:
                    dec_bboxes.append(torch.sigmoid(bbox + inverse_sigmoid(last_refined_bbox)))
            elif i == self.eval_idx:
                dec_cls.append(score_head[i](output))
                dec_bboxes.append(refined_bbox)
                break

            last_refined_bbox = refined_bbox
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox

        return torch.stack(dec_bboxes), torch.stack(dec_cls)
    





# class Deformable_position_TransformerDecoder(nn.Module):
#     """
#     Implementation of Deformable Transformer Decoder based on PaddleDetection.

#     https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
#     """

#     def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
#         """Initialize the DeformableTransformerDecoder with the given parameters."""
#         super().__init__()
#         self.layers = _get_clones(decoder_layer, num_layers)
#         self.num_layers = num_layers
#         self.hidden_dim = hidden_dim
#         self.num_heads = decoder_layer.num_heads
#         self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
#         self.position_relation_embedding = PositionRelationEmbedding(4, self.num_heads) 

#     def forward(
#             self,
#             embed,  # decoder embeddings
#             refer_bbox,  # anchor
#             feats,  # image features
#             shapes,  # feature shapes
#             bbox_head,
#             score_head,
#             pos_mlp,
#             attn_mask=None,
#             padding_mask=None):
#         """Perform the forward pass through the entire decoder."""
#         output = embed
#         dec_bboxes = []
#         dec_cls = []
#         last_refined_bbox = None
#         refer_bbox = refer_bbox.sigmoid()
#         pos_relation = attn_mask  # fallback pos_relation to attn_mask
#         for i, layer in enumerate(self.layers):
#             output = layer(output, refer_bbox, feats, shapes, padding_mask, pos_relation, pos_mlp(refer_bbox))
            
#             bbox = bbox_head[i](output)
#             refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox))
#             if i>0:
#                 dec_bbox = torch.sigmoid(bbox + inverse_sigmoid(last_refined_bbox))

#             if self.training:
#                 dec_cls.append(score_head[i](output))
#                 if i == 0:
#                     dec_bboxes.append(refined_bbox)
#                 else:

#                     dec_bboxes.append(dec_bbox)
#             elif i == self.eval_idx:
#                 dec_cls.append(score_head[i](output))
#                 dec_bboxes.append(refined_bbox)
#                 break
            
#             if i == self.num_layers - 1:
#                 break
#             src_boxes = tgt_boxes if i >=1 else refer_bbox
#             tgt_boxes = refined_bbox if i==0 else dec_bbox
#             pos_relation = self.position_relation_embedding(src_boxes,tgt_boxes).flatten(0,1)

#             if attn_mask is not None:
#                 pos_relation.masked_fill_(attn_mask,float("-inf"))
            
#             last_refined_bbox = refined_bbox
#             refer_bbox = refined_bbox.detach() if self.training else refined_bbox

#         return torch.stack(dec_bboxes), torch.stack(dec_cls)
    

class Deformable_position_TransformerDecoder(nn.Module):
    """
    Implementation of Deformable Transformer Decoder based on PaddleDetection.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    """

    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        """Initialize the DeformableTransformerDecoder with the given parameters."""
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = decoder_layer.num_heads
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.position_relation_embedding = PositionRelationEmbedding(4, self.num_heads) 

    def forward(
            self,
            embed,  # decoder embeddings
            refer_bbox,  # anchor
            feats,  # image features
            shapes,  # feature shapes
            bbox_head,
            score_head,
            pos_mlp,
            attn_mask=None,
            padding_mask=None):
        """Perform the forward pass through the entire decoder."""
        output = embed
        dec_bboxes = []
        dec_cls = []
        last_refined_bbox = None
        refer_bbox = refer_bbox.sigmoid()
        pos_relation = attn_mask  # fallback pos_relation to attn_mask
        for i, layer in enumerate(self.layers):
            output = layer(output, refer_bbox, feats, shapes, padding_mask, pos_relation, pos_mlp(refer_bbox))
            
            bbox = bbox_head[i](output)
            refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox))
            if i>0:
                dec_bbox = torch.sigmoid(bbox + inverse_sigmoid(last_refined_bbox))

            if self.training:
                dec_cls.append(score_head[i](output))
                if i == 0:
                    dec_bboxes.append(refined_bbox)
                else:

                    dec_bboxes.append(dec_bbox)
            elif i == self.eval_idx:
                dec_cls.append(score_head[i](output))
                dec_bboxes.append(refined_bbox)
                break
            
            if i == self.num_layers - 1:
                break
            src_boxes = tgt_boxes if i >=1 else refer_bbox
            tgt_boxes = refined_bbox if i==0 else dec_bbox
            pos_relation = self.position_relation_embedding(src_boxes,tgt_boxes).flatten(0,1)

            if attn_mask is not None:
                pos_relation.masked_fill_(attn_mask,float("-inf"))
            
            last_refined_bbox = refined_bbox
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox

        return torch.stack(dec_bboxes), torch.stack(dec_cls)

    
def box_rel_encoding(src_boxes, tgt_boxes, eps=1e-5):
    # construct position relation
    xy1, wh1 = src_boxes.split([2, 2], -1)
    xy2, wh2 = tgt_boxes.split([2, 2], -1)
    delta_xy = torch.abs(xy1.unsqueeze(-2) - xy2.unsqueeze(-3))
    delta_xy = torch.log(delta_xy / (wh1.unsqueeze(-2) + eps) + 1.0)
    delta_wh = torch.log((wh1.unsqueeze(-2) + eps) / (wh2.unsqueeze(-3) + eps))
    # 计算IoU
    x1y1_1, x2y2_1 = xy1 - wh1/2, xy1 + wh1/2
    x1y1_2, x2y2_2 = xy2 - wh2/2, xy2 + wh2/2
    
    x1y1_inter = torch.maximum(x1y1_1.unsqueeze(-2), x1y1_2.unsqueeze(-3))
    x2y2_inter = torch.minimum(x2y2_1.unsqueeze(-2), x2y2_2.unsqueeze(-3))
    
    wh_inter = torch.clamp(x2y2_inter - x1y1_inter, min=0)
    area_inter = wh_inter.prod(dim=-1)
    
    area_1 = wh1.prod(dim=-1).unsqueeze(-1)
    area_2 = wh2.prod(dim=-1).unsqueeze(-2)
    
    iou = area_inter / (area_1 + area_2 - area_inter + eps)
    
    # 计算相对距离
    center1 = xy1
    center2 = xy2
    # rel_distance = torch.norm(center1.unsqueeze(-2) - center2.unsqueeze(-3), dim=-1)
    # rel_distance = torch.log(rel_distance / torch.sqrt(wh1.prod(dim=-1)).unsqueeze(-1) + eps)
    
    # 计算相对角度
    delta_center = center2.unsqueeze(-3) - center1.unsqueeze(-2)
    angle = torch.atan2(delta_center[..., 1], delta_center[..., 0])
    
    # 组合所有特征
    pos_embed = torch.cat([
        delta_xy,  # 相对位置 (2)
        delta_wh,  # 相对尺寸 (2)
        iou.unsqueeze(-1),  # IoU (1)
        # rel_distance.unsqueeze(-1),  # 相对距离 (1)
        angle.unsqueeze(-1),  # 相对角度 (1)
    ], -1)
    # pos_embed = torch.cat([delta_xy, delta_wh], -1)  # [batch_size, num_boxes1, num_boxes2, 4]

    return pos_embed
# def box_rel_encoding(src_boxes, tgt_boxes, eps=1e-5):
#     # construct position relation
#     xy1, wh1 = src_boxes.split([2, 2], -1)
#     xy2, wh2 = tgt_boxes.split([2, 2], -1)
#     delta_xy = torch.abs(xy1.unsqueeze(-2) - xy2.unsqueeze(-3))
#     delta_xy = torch.log(delta_xy / (wh1.unsqueeze(-2) + eps) + 1.0)
#     delta_wh = torch.log((wh1.unsqueeze(-2) + eps) / (wh2.unsqueeze(-3) + eps))
#     pos_embed = torch.cat([delta_xy, delta_wh], -1)  # [batch_size, num_boxes1, num_boxes2, 4]

#     return pos_embed
class PosEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(6, 12)
        self.linear2 = nn.Linear(12, 24)
        self.relu = nn.GELU()
    #     self._init_weights()

    # def _init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.xavier_uniform_(m.weight)
    #             if m.bias is not None:
    #                 nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        return self.linear2(x)

import functools

class PositionRelationEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        temperature=10000.0,
        scale=100.0,
        activation_layer=nn.ReLU,
        inplace=True,
    ):
        super().__init__()
        self.pos_proj = Conv2dNormActivation(
            embed_dim * 6,
            num_heads,
            kernel_size=1,
            inplace=inplace,
            norm_layer=None,
            activation_layer=activation_layer,
        )
        # self.pos_func = functools.partial(
        #     get_sine_pos_embed,
        #     num_pos_feats=embed_dim,
        #     temperature=temperature,
        #     scale=scale,
        #     exchange_xy=False,
        # )
        self.pos_func =PosEmbedding()

    def forward(self, src_boxes: Tensor, tgt_boxes: Tensor = None):
        if tgt_boxes is None:
            tgt_boxes = src_boxes
        # src_boxes: [batch_size, num_boxes1, 4]
        # tgt_boxes: [batch_size, num_boxes2, 4]
        torch._assert(src_boxes.shape[-1] == 4, f"src_boxes much have 4 coordinates")
        torch._assert(tgt_boxes.shape[-1] == 4, f"tgt_boxes must have 4 coordinates")
        with torch.no_grad():
            pos_embed = box_rel_encoding(src_boxes, tgt_boxes)
            pos_embed = self.pos_func(pos_embed).permute(0, 3, 1, 2)
        pos_embed = self.pos_proj(pos_embed)

        return pos_embed.clone()



    






































# # position decoder

# class Deformable_position_TransformerDecoderLayer(nn.Module):
#     """
#     Deformable Transformer Decoder Layer inspired by PaddleDetection and Deformable-DETR implementations.

#     https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
#     https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/deformable_transformer.py
#     """

#     def __init__(self, d_model=256, n_heads=8, d_ffn=1024, dropout=0.1, act=nn.ReLU(), n_levels=4, n_points=4):
#         """Initialize the DeformableTransformerDecoderLayer with the given parameters."""
#         super().__init__()
#         self.num_heads = n_heads
        
#         # self.pos_relation = PositionRelationEmbedding(16,n_heads)
#         # Self attention
#         self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
#         self.dropout1 = nn.Dropout(dropout)
#         self.norm1 = nn.LayerNorm(d_model)

#         # Cross attention
#         self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
#         self.dropout2 = nn.Dropout(dropout)
#         self.norm2 = nn.LayerNorm(d_model)

#         # FFN
#         self.linear1 = nn.Linear(d_model, d_ffn)
#         self.act = act
#         self.dropout3 = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(d_ffn, d_model)
#         self.dropout4 = nn.Dropout(dropout)
#         self.norm3 = nn.LayerNorm(d_model)

#     @staticmethod
#     def with_pos_embed(tensor, pos):
#         """Add positional embeddings to the input tensor, if provided."""
#         return tensor if pos is None else tensor + pos

#     def forward_ffn(self, tgt):
#         """Perform forward pass through the Feed-Forward Network part of the layer."""
#         tgt2 = self.linear2(self.dropout3(self.act(self.linear1(tgt))))
#         tgt = tgt + self.dropout4(tgt2)
#         return self.norm3(tgt)

#     def forward(self, embed, refer_bbox, feats, shapes, padding_mask=None, attn_mask=None, query_pos=None):
#         """Perform the forward pass through the entire decoder layer."""
#         # boxes = refer_bbox # [batch_size, num_queries, 4]
#         # pos_relation = self.pos_relation(boxes).flatten(0, 1)
#         # attn_mask=pos_relation
#         # Self attention
#         q = k = self.with_pos_embed(embed, query_pos)
#         tgt = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), embed.transpose(0, 1),
#                              attn_mask=attn_mask)[0].transpose(0, 1) # <<< 使用 attn_mask 传递位置关系嵌入
#         embed = embed + self.dropout1(tgt)
#         embed = self.norm1(embed)

#         # Cross attention
#         tgt = self.cross_attn(self.with_pos_embed(embed, query_pos), refer_bbox.unsqueeze(2), feats, shapes,
#                               padding_mask)
#         embed = embed + self.dropout2(tgt)
#         embed = self.norm2(embed)

#         # FFN
#         return self.forward_ffn(embed)


# class Deformable_position_TransformerDecoder(nn.Module):
#     """
#     Implementation of Deformable Transformer Decoder based on PaddleDetection.

#     https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
#     """

#     def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
#         """Initialize the DeformableTransformerDecoder with the given parameters."""
#         super().__init__()
#         self.layers = _get_clones(decoder_layer, num_layers)
#         self.num_layers = num_layers
#         self.hidden_dim = hidden_dim
#         self.num_heads = decoder_layer.num_heads
#         self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        
#         # <<< 添加 PositionRelationEmbedding 模块
#         self.position_relation_embedding = PositionRelationEmbedding(4, self.num_heads) 

#     def forward(       
#             self,
#             embed,  # decoder embeddings
#             refer_bbox,  # anchor
#             feats,  # image features
#             shapes,  # feature shapes
#             bbox_head,
#             score_head,
#             pos_mlp,
#             attn_mask=None,
#             padding_mask=None):
#         """Perform the forward pass through the entire decoder."""
#         output = embed
#         dec_bboxes = []
#         dec_cls = []
#         last_refined_bbox = None
#         refer_bbox = refer_bbox.sigmoid()
#         pos_relation = attn_mask  # fallback pos_relation to attn_mask
#         for i, layer in enumerate(self.layers):
#             # if self.training:
#             #     pass
#             # else:
#             pos_relation = self.position_relation_embedding(refer_bbox, refer_bbox).flatten(0, 1)
#             output = layer(output, refer_bbox, feats, shapes, padding_mask, pos_relation, pos_mlp(refer_bbox)) # <<< 传递位置关系嵌入

#             bbox = bbox_head[i](output)
#             refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox))
#             # dec_bbox = None
#             if i == 0:
#                 dec_bbox = torch.sigmoid(bbox + inverse_sigmoid(refined_bbox))
#             else:
#                 dec_bbox = torch.sigmoid(bbox + inverse_sigmoid(last_refined_bbox))
#             if self.training:
#                 dec_cls.append(score_head[i](output))
#                 if i == 0:
#                     dec_bboxes.append(refined_bbox)
#                 else:
#                     # dec_bbox = torch.sigmoid(bbox + inverse_sigmoid(refined_bbox))
#                     dec_bboxes.append(torch.sigmoid(bbox + inverse_sigmoid(last_refined_bbox)))
#             # dec_cls.append(score_head[i](output))
#             # if i == 0:
#             #     dec_bboxes.append(refined_bbox)
#             # else:
#                 # dec_bbox = torch.sigmoid(bbox + inverse_sigmoid(refined_bbox))
#             #     dec_bboxes.append(torch.sigmoid(bbox + inverse_sigmoid(refined_bbox)))
#             elif i == self.eval_idx:
#             # if i == self.eval_idx:
#                 dec_cls.append(score_head[i](output))
#                 dec_bboxes.append(refined_bbox)
#                 break
#             if i == self.num_layers - 1:
#                 break
#              # 第一层不需要计算关系嵌入
#             # src_boxes = tgt_boxes if i >= 1 else refer_bbox
#             tgt_boxes = dec_bbox
            
#             pos_relation = self.position_relation_embedding(tgt_boxes, tgt_boxes).flatten(0, 1)
#             # from torch.utils.checkpoint import checkpoint

#             # # 原本的代码： pos_relation = self.position_relation_embedding(src_boxes, tgt_boxes).flatten(0, 1)
#             # pos_relation = checkpoint(
#             # self.position_relation_embedding, tgt_boxes, tgt_boxes
#             # ).flatten(0, 1)
            
#             # print(pos_relation)
#             if attn_mask is not None:
#                 pos_relation = pos_relation.masked_fill(attn_mask, float("-inf"))
#             # last_refined_bbox = refined_bbox
#             # refer_bbox = refined_bbox.detach() if self.training else refer_bbox
#             # refer_bbox = inverse_sigmoid(refer_bbox)
#             # refer_bbox = bbox_head[i](output)+refer_bbox
#             # refer_bbox = torch.sigmoid(refer_bbox)
#             last_refined_bbox = refined_bbox
#             refer_bbox = refined_bbox.detach() if self.training else refined_bbox
#             # if self.training else refined_bbox
#         return torch.stack(dec_bboxes), torch.stack(dec_cls)


# from typing import Callable, List, Optional
# import warnings
# from torch import Tensor, nn
# import functools

# class ConvNormActivation(torch.nn.Sequential):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         kernel_size: int = 3,
#         stride: int = 1,
#         padding: Optional[int] = None,
#         groups: int = 1,
#         norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
#         activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
#         dilation: int = 1,
#         inplace: Optional[bool] = True,
#         bias: Optional[bool] = None,
#         conv_layer: Callable[..., torch.nn.Module] = torch.nn.Conv2d,
#     ) -> None:

#         if padding is None:
#             padding = (kernel_size - 1) // 2 * dilation
#         if bias is None:
#             bias = norm_layer is None

#         layers = [
#             conv_layer(
#                 in_channels,
#                 out_channels,
#                 kernel_size,
#                 stride,
#                 padding,
#                 dilation=dilation,
#                 groups=groups,
#                 bias=bias,
#             )
#         ]

#         if norm_layer is not None:
#             layers.append(norm_layer(out_channels))

#         if activation_layer is not None:
#             params = {} if inplace is None else {"inplace": inplace}
#             layers.append(activation_layer(**params))
#         super().__init__(*layers)
#         self.out_channels = out_channels

#         if self.__class__ == ConvNormActivation:
#             warnings.warn(
#                 "Don't use ConvNormActivation directly, please use Conv2dNormActivation and Conv3dNormActivation instead."
#             )
# class Conv2dNormActivation(ConvNormActivation):
#     """
#     Configurable block used for Convolution2d-Normalization-Activation blocks.

#     Args:
#         in_channels (int): Number of channels in the input image
#         out_channels (int): Number of channels produced by the Convolution-Normalization-Activation block
#         kernel_size: (int, optional): Size of the convolving kernel. Default: 3
#         stride (int, optional): Stride of the convolution. Default: 1
#         padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: None, in which case it will calculated as ``padding = (kernel_size - 1) // 2 * dilation``
#         groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
#         norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer wont be used. Default: ``torch.nn.BatchNorm2d``
#         activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer wont be used. Default: ``torch.nn.ReLU``
#         dilation (int): Spacing between kernel elements. Default: 1
#         inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
#         bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.

#     """
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         kernel_size: int = 3,
#         stride: int = 1,
#         padding: Optional[int] = None,
#         groups: int = 1,
#         norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
#         activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
#         dilation: int = 1,
#         inplace: Optional[bool] = True,
#         bias: Optional[bool] = None,
#     ) -> None:

#         super().__init__(
#             in_channels,
#             out_channels,
#             kernel_size,
#             stride,
#             padding,
#             groups,
#             norm_layer,
#             activation_layer,
#             dilation,
#             inplace,
#             bias,
#             torch.nn.Conv2d,
#         )

# def box_rel_encoding(src_boxes, tgt_boxes, eps=1e-5):
#     # construct position relation
#     xy1, wh1 = src_boxes.split([2, 2], -1)
#     xy2, wh2 = tgt_boxes.split([2, 2], -1)
#     delta_xy = torch.abs(xy1.unsqueeze(-2) - xy2.unsqueeze(-3))
#     delta_xy = torch.log(delta_xy / (wh1.unsqueeze(-2) + eps) + 1.0)
#     delta_wh = torch.log((wh1.unsqueeze(-2) + eps) / (wh2.unsqueeze(-3) + eps))
#     # 计算IoU
#     x1y1_1, x2y2_1 = xy1 - wh1/2, xy1 + wh1/2
#     x1y1_2, x2y2_2 = xy2 - wh2/2, xy2 + wh2/2
    
#     x1y1_inter = torch.maximum(x1y1_1.unsqueeze(-2), x1y1_2.unsqueeze(-3))
#     x2y2_inter = torch.minimum(x2y2_1.unsqueeze(-2), x2y2_2.unsqueeze(-3))
    
#     wh_inter = torch.clamp(x2y2_inter - x1y1_inter, min=0)
#     area_inter = wh_inter.prod(dim=-1)
    
#     area_1 = wh1.prod(dim=-1).unsqueeze(-1)
#     area_2 = wh2.prod(dim=-1).unsqueeze(-2)
    
#     iou = area_inter / (area_1 + area_2 - area_inter + eps)
    
#     # 计算相对距离
#     center1 = xy1
#     center2 = xy2
#     # rel_distance = torch.norm(center1.unsqueeze(-2) - center2.unsqueeze(-3), dim=-1)
#     # rel_distance = torch.log(rel_distance / torch.sqrt(wh1.prod(dim=-1)).unsqueeze(-1) + eps)
    
#     # 计算相对角度
#     delta_center = center2.unsqueeze(-3) - center1.unsqueeze(-2)
#     angle = torch.atan2(delta_center[..., 1], delta_center[..., 0])
    
#     # 组合所有特征
#     pos_embed = torch.cat([
#         delta_xy,  # 相对位置 (2)
#         delta_wh,  # 相对尺寸 (2)
#         iou.unsqueeze(-1),  # IoU (1)
#         # rel_distance.unsqueeze(-1),  # 相对距离 (1)
#         angle.unsqueeze(-1),  # 相对角度 (1)
#     ], -1)
#     # pos_embed = torch.cat([delta_xy, delta_wh], -1)  # [batch_size, num_boxes1, num_boxes2, 4]

#     return pos_embed

# @functools.lru_cache  # use lru_cache to avoid redundant calculation for dim_t
# def get_dim_t(num_pos_feats: int, temperature: int, device: torch.device):
#     dim_t = torch.arange(num_pos_feats // 2, dtype=torch.float32, device=device)
#     dim_t = temperature**(dim_t * 2 / num_pos_feats)
#     return dim_t  # (0, 2, 4, ..., ⌊n/2⌋*2)

# def exchange_xy_fn(pos_res):
#     index = torch.cat([
#         torch.arange(1, -1, -1, device=pos_res.device),
#         torch.arange(2, pos_res.shape[-2], device=pos_res.device),
#     ])
#     pos_res = torch.index_select(pos_res, -2, index)
#     return pos_res
# @torch.no_grad()
# def get_sine_pos_embed(
#     pos_tensor: Tensor,
#     num_pos_feats: int = 128,
#     temperature: int = 10000,
#     scale: float = 2 * math.pi,
#     exchange_xy: bool = True,
# ) -> Tensor:
#     """Generate sine position embedding for a position tensor

#     :param pos_tensor: shape as (..., 2*n).
#     :param num_pos_feats: projected shape for each float in the tensor, defaults to 128
#     :param temperature: the temperature used for scaling the position embedding, defaults to 10000
#     :param exchange_xy: exchange pos x and pos. For example,
#         input tensor is [x, y], the results will be [pos(y), pos(x)], defaults to True
#     :return: position embedding with shape (None, n * num_pos_feats)
#     """
#     dim_t = get_dim_t(num_pos_feats, temperature, pos_tensor.device)

#     pos_res = pos_tensor.unsqueeze(-1) * scale / dim_t
#     pos_res = torch.stack((pos_res.sin(), pos_res.cos()), dim=-1).flatten(-2)
#     if exchange_xy:
#         pos_res = exchange_xy_fn(pos_res)
#     pos_res = pos_res.flatten(-2)
#     return pos_res

# class PosEmbedding(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear1 = nn.Linear(6, 12)
#         self.linear2 = nn.Linear(12, 24)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = self.relu(self.linear1(x))
#         return self.linear2(x)
# @torch.no_grad()
# def get_sine_pos_embed(
#     pos_tensor: Tensor,
#     num_pos_feats: int = 128,
#     temperature: int = 10000,
#     scale: float = 2 * math.pi,
#     exchange_xy: bool = True,
# ) -> Tensor:
#     """使用线性层替代sine position embedding"""
#     device = pos_tensor.device
    
#     if not hasattr(get_sine_pos_embed, '_pos_embed'):
#         get_sine_pos_embed._pos_embed = PosEmbedding().to(device)
    
#     # 确保模型和输入在同一设备上
#     get_sine_pos_embed._pos_embed = get_sine_pos_embed._pos_embed.to(device)
    
#     # 保持形状处理兼容性
#     orig_shape = pos_tensor.shape[:-1]
#     x = pos_tensor.reshape(-1, 6)
    
#     # 前向传播
#     pos_res = get_sine_pos_embed._pos_embed(x)
    
#     # 还原形状
#     pos_res = pos_res.view(*orig_shape, 24)
    
#     return pos_res
# class PosEmbedding(nn.Module):
#     def __init__(self, seed=1):
#         super().__init__()
#         torch.manual_seed(seed)  # 固定随机种子
#         self.pos_embed = nn.Sequential(
#             nn.Linear(6, 12),
#             nn.ReLU(),
#             nn.Linear(12, 24)
#         )
    
#     def forward(self, x):
#         return self.pos_embed(x)

# @torch.no_grad()
# def get_sine_pos_embed(
#     pos_tensor: Tensor,
#     num_pos_feats: int = 128,
#     temperature: int = 10000,
#     scale: float = 2 * math.pi,
#     exchange_xy: bool = True,
# ) -> Tensor:
#     """使用线性层替代sine position embedding"""
#     device = pos_tensor.device
    
#     if not hasattr(get_sine_pos_embed, '_pos_embed'):
#         get_sine_pos_embed._pos_embed = PosEmbedding().to(device)
    
#     # 确保模型处于评估模式
#     get_sine_pos_embed._pos_embed.eval()
#     get_sine_pos_embed._pos_embed = get_sine_pos_embed._pos_embed.to(device)
    
#     orig_shape = pos_tensor.shape[:-1]
#     x = pos_tensor.reshape(-1, 6)
#     pos_res = get_sine_pos_embed._pos_embed(x)
#     pos_res = pos_res.view(*orig_shape, 24)
    
#     return pos_res
# @torch.no_grad()
# def get_sine_pos_embed(
#     pos_tensor: Tensor,
#     num_pos_feats: int = 128,
#     temperature: int = 10000,
#     scale: float = 2 * math.pi,
#     exchange_xy: bool = True,
# ) -> Tensor:
#     """确保所有计算都在GPU上进行的位置编码"""
#     # 获取当前可用的CUDA设备
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     pos_tensor = pos_tensor.to(device)
    
#     # 使用ModuleList创建模型
#     if not hasattr(get_sine_pos_embed, '_pos_embed'):
#         model = nn.ModuleList([
#             nn.Linear(6, 12),
#             nn.ReLU(),
#             nn.Linear(12, 24)
#         ])
#         model = model.to(device)  # 确保模型在GPU上
#         get_sine_pos_embed._pos_embed = model
#     else:
#         get_sine_pos_embed._pos_embed = get_sine_pos_embed._pos_embed.to(device)
    
#     # 执行前向传播
#     x = pos_tensor.reshape(-1, 6)
#     for layer in get_sine_pos_embed._pos_embed:
#         x = layer.to(device)(x)  # 确保每一层都在GPU上
    
#     return x.view(*pos_tensor.shape[:-1], 24)

# class PosEmbedding(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear1 = nn.Linear(6, 12)
#         self.linear2 = nn.Linear(12, 24)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = self.relu(self.linear1(x))
#         return self.linear2(x)
# class PosEmbedding(nn.Module):
#     def __init__(self,seed = 100):
#         super().__init__()
#         torch.manual_seed(seed)  # 固定随机种子
#         self.pos_embed = nn.Sequential(
#             nn.Linear(6, 12),
#             nn.ReLU(),
# 		    #nn.GELU(),
#             nn.Linear(12, 24)
#         )
    
#     def forward(self, x):
#         return self.pos_embed(x)
# @torch.no_grad()
# def get_sine_pos_embed(
#     pos_tensor: Tensor,
#     num_pos_feats: int = 128,
#     temperature: int = 10000,
#     scale: float = 2 * math.pi,
#     exchange_xy: bool = True,
# ) -> Tensor:
#     """使用线性层替代sine position embedding"""
#     device = pos_tensor.device
    
#     if not hasattr(get_sine_pos_embed, '_pos_embed'):
#         get_sine_pos_embed._pos_embed = PosEmbedding().to(device)
    
#     # 确保模型和输入在同一设备上
#     get_sine_pos_embed._pos_embed = get_sine_pos_embed._pos_embed.to(device)
    
#     # 保持形状处理兼容性
#     orig_shape = pos_tensor.shape[:-1]
#     x = pos_tensor.reshape(-1, 6)
    
#     # 前向传播
#     pos_res = get_sine_pos_embed._pos_embed(x)
    
#     # 还原形状
#     pos_res = pos_res.view(*orig_shape, 24)
    
#     return pos_res
# # class PosEmbedding(nn.Module):
# #     def __init__(self, seed=3407):
# #         super().__init__()
# #         torch.manual_seed(seed)  # 固定随机种子
# #         self.pos_embed = nn.Sequential(
# #             nn.Linear(6, 12),
# #             nn.ReLU(),
# #             nn.Linear(12, 24)
# #         )
    
# #     def forward(self, x):
# #         return self.pos_embed(x)



    
# # class PosEmbeddingWithNorm(nn.Module):
# #     def __init__(self):
# #         super().__init__()
# #         self.linear1 = nn.Linear(6, 12)
# #         self.norm1 = nn.LayerNorm(12)
# #         self.linear2 = nn.Linear(12, 24) 
# #         self.norm2 = nn.LayerNorm(24)
# #         self.relu = nn.ReLU()

# #     def forward(self, x):
# #         x = self.linear1(x)
# #         x = self.norm1(x)  # 归一化
# #         x = self.relu(x)
# #         x = self.linear2(x)
# #         x = self.norm2(x)  # 归一化
# #         return x
# class PositionRelationEmbedding(nn.Module):
#     def __init__(
#         self,
#         embed_dim=256,
#         num_heads=8,
#         temperature=10000.,
#         scale=100.,
#         activation_layer=nn.ReLU,
#         inplace=True,
#     ):
#         super().__init__()
#         self.pos_proj = Conv2dNormActivation(
#             embed_dim * 6,
#             num_heads,
#             kernel_size=1,
#             inplace=inplace,
#             norm_layer=None,
#             activation_layer=activation_layer,
#         )
#         # self.pos_func = functools.partial(
#         #     get_sine_pos_embed,
#         #     num_pos_feats=embed_dim,
#         #     temperature=temperature,
#         #     scale=scale,
#         #     exchange_xy=False,
#         # )
#         self.pos_func = PosEmbedding()

#     def forward(self, src_boxes: Tensor, tgt_boxes: Tensor = None):
#         if tgt_boxes is None:
#             tgt_boxes = src_boxes
#         # src_boxes: [batch_size, num_boxes1, 4]
#         # tgt_boxes: [batch_size, num_boxes2, 4]
#         # torch._assert(src_boxes.shape[-1] == 4, f"src_boxes much have 4 coordinates")
#         # torch._assert(tgt_boxes.shape[-1] == 4, f"tgt_boxes must have 4 coordinates")
#         with torch.no_grad():
#             pos_embed = box_rel_encoding(src_boxes, tgt_boxes)
#             pos_embed = self.pos_func(pos_embed).permute(0, 3, 1, 2)
#         pos_embed = self.pos_proj(pos_embed)

#         return pos_embed.clone()
# class PosEmbedding(nn.Module):
#     def __init__(self, seed=1000):
#         super().__init__()
#         torch.manual_seed(seed)  # 固定随机种子
#         self.pos_embed = nn.Sequential(
#             nn.Linear(6, 12),
#             nn.ReLU(),
# 		    #nn.GELU(),
#             nn.Linear(12, 24)
#         )
    
#     def forward(self, x):
#         return self.pos_embed(x)

# @torch.no_grad()
# def get_sine_pos_embed(
#     pos_tensor: Tensor,
#     num_pos_feats: int = 128,
#     temperature: int = 10000,
#     scale: float = 2 * math.pi,
#     exchange_xy: bool = True,
# ) -> Tensor:
#     """使用线性层替代sine position embedding"""
#     device = pos_tensor.device
    
#     if not hasattr(get_sine_pos_embed, '_pos_embed'):
#         get_sine_pos_embed._pos_embed = PosEmbedding().to(device)
    
#     # 确保模型和输入在同一设备上
#     get_sine_pos_embed._pos_embed = get_sine_pos_embed._pos_embed.to(device)
    
#     # 保持形状处理兼容性
#     orig_shape = pos_tensor.shape[:-1]
#     x = pos_tensor.reshape(-1, 6)
    
#     # 前向传播
#     pos_res = get_sine_pos_embed._pos_embed(x)
    
#     # 还原形状
#     pos_res = pos_res.view(*orig_shape, 24)
    
#     return pos_res

# class PosEmbedding(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear1 = nn.Linear(6, 12)
#         self.linear2 = nn.Linear(12, 24)
#         self.relu = nn.GELU()
#         self._init_weights()

#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)

#     def forward(self, x):
#         x = self.relu(self.linear1(x))
#         return self.linear2(x)
    
# class PositionRelationEmbedding(nn.Module):
#     def __init__(
#         self,
#         embed_dim=256,
#         num_heads=8,
#         temperature=10000.,
#         scale=100.,
#         activation_layer=nn.ReLU,
#         inplace=True,
#     ):
#         super().__init__()
#         self.pos_proj = Conv2dNormActivation(
#             embed_dim * 6,
#             num_heads,
#             kernel_size=1,
#             inplace=inplace,
#             norm_layer=None,
#             activation_layer=activation_layer,
#         )
#         self.pos_func = functools.partial(
#             get_sine_pos_embed,
#             num_pos_feats=embed_dim,
#             temperature=temperature,
#             scale=scale,
#             exchange_xy=False,
#         )
#         # self.pos_func = PosEmbedding()

#     def forward(self, src_boxes: Tensor, tgt_boxes: Tensor = None):
#         if tgt_boxes is None:
#             tgt_boxes = src_boxes
#         # src_boxes: [batch_size, num_boxes1, 4]
#         # tgt_boxes: [batch_size, num_boxes2, 4]
#         # torch._assert(src_boxes.shape[-1] == 4, f"src_boxes much have 4 coordinates")
#         # torch._assert(tgt_boxes.shape[-1] == 4, f"tgt_boxes must have 4 coordinates")
#         with torch.no_grad():
#             pos_embed = box_rel_encoding(src_boxes, tgt_boxes)
#         pos_embed = self.pos_func(pos_embed).permute(0, 3, 1, 2)
#         pos_embed = self.pos_proj(pos_embed)

#         return pos_embed.clone()
# class PositionRelationEmbedding(nn.Module):
#     def __init__(
#         self,
#         embed_dim=256,
#         num_heads=8,
#         activation_layer=nn.GELU,
#         inplace=True,
#     ):
#         super().__init__()
#         self.pos_proj = Conv2dNormActivation(
#             6, # 修改为 6
#             num_heads,
#             kernel_size=1,
#             inplace=inplace,
#             norm_layer=None,
#             activation_layer=activation_layer,
#         )

#     def forward(self, src_boxes: Tensor, tgt_boxes: Tensor = None):
#         if tgt_boxes is None:
#             tgt_boxes = src_boxes
#         # src_boxes: [batch_size, num_boxes, 4]
#         # tgt_boxes: [batch_size, num_boxes, 4]
#         num_boxes = src_boxes.shape[1]
#         with torch.no_grad():
#             pos_embed = box_rel_encoding(src_boxes, tgt_boxes) # [batch_size, num_boxes, num_boxes, 6]
#             pos_embed = pos_embed.permute(0, 3, 1, 2)
#         pos_embed = pos_embed.reshape(-1, 6, num_boxes, num_boxes)
#         pos_embed = self.pos_proj(pos_embed)
#         pos_embed = pos_embed.reshape(-1, num_boxes, num_boxes, 8)
#          # [batch_size, num_boxes, num_boxes, 8]
#         pos_embed = pos_embed.permute(0, 3, 1, 2)  # [batch_size, num_heads, num_boxes, num_boxes]
#         # return pos_embed
#         return pos_embed.clone()   
# class PositionRelationEmbedding(nn.Module):
#     def __init__(
#         self,
#         embed_dim=256,
#         num_heads=8,
#         activation_layer=nn.GELU,  # 使用 GELU
#         inplace=True,
#     ):
#         super().__init__()
#         self.pos_proj1 = nn.Linear(6, embed_dim * num_heads)
#         self.pos_proj2 = nn.Linear(embed_dim * num_heads, embed_dim * num_heads)
#         self.act = activation_layer()
#         self.num_heads = num_heads

#     def forward(self, src_boxes: Tensor, tgt_boxes: Tensor = None):
#         if tgt_boxes is None:
#             tgt_boxes = src_boxes
#         pos_embed = box_rel_encoding(src_boxes, tgt_boxes)  # [batch_size, num_boxes1, num_boxes2, 6]
#         pos_embed = self.pos_proj1(pos_embed)  # [batch_size, num_boxes1, num_boxes2, embed_dim * num_heads]
#         pos_embed = self.act(pos_embed)
#         pos_embed = self.pos_proj2(pos_embed) # [batch_size, num_boxes1, num_boxes2, embed_dim * num_heads]
#         pos_embed = pos_embed.permute(0, 3, 1, 2).reshape(-1, self.num_heads, src_boxes.shape[1], tgt_boxes.shape[1])
#         return pos_embed
# from typing import Callable, List, Optional
# import warnings
# from torch import Tensor, nn
# import functools

# class ConvNormActivation(torch.nn.Sequential):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         kernel_size: int = 3,
#         stride: int = 1,
#         padding: Optional[int] = None,
#         groups: int = 1,
#         norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
#         activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
#         dilation: int = 1,
#         inplace: Optional[bool] = True,
#         bias: Optional[bool] = None,
#         conv_layer: Callable[..., torch.nn.Module] = torch.nn.Conv2d,
#     ) -> None:

#         if padding is None:
#             padding = (kernel_size - 1) // 2 * dilation
#         if bias is None:
#             bias = norm_layer is None

#         layers = [
#             conv_layer(
#                 in_channels,
#                 out_channels,
#                 kernel_size,
#                 stride,
#                 padding,
#                 dilation=dilation,
#                 groups=groups,
#                 bias=bias,
#             )
#         ]

#         if norm_layer is not None:
#             layers.append(norm_layer(out_channels))

#         if activation_layer is not None:
#             params = {} if inplace is None else {"inplace": inplace}
#             layers.append(activation_layer(**params))
#         super().__init__(*layers)
#         self.out_channels = out_channels

#         if self.__class__ == ConvNormActivation:
#             warnings.warn(
#                 "Don't use ConvNormActivation directly, please use Conv2dNormActivation and Conv3dNormActivation instead."
#             )
# class Conv2dNormActivation(ConvNormActivation):
#     """
#     Configurable block used for Convolution2d-Normalization-Activation blocks.

#     Args:
#         in_channels (int): Number of channels in the input image
#         out_channels (int): Number of channels produced by the Convolution-Normalization-Activation block
#         kernel_size: (int, optional): Size of the convolving kernel. Default: 3
#         stride (int, optional): Stride of the convolution. Default: 1
#         padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: None, in which case it will calculated as ``padding = (kernel_size - 1) // 2 * dilation``
#         groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
#         norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer wont be used. Default: ``torch.nn.BatchNorm2d``
#         activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer wont be used. Default: ``torch.nn.ReLU``
#         dilation (int): Spacing between kernel elements. Default: 1
#         inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
#         bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.

#     """
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         kernel_size: int = 3,
#         stride: int = 1,
#         padding: Optional[int] = None,
#         groups: int = 1,
#         norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
#         activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
#         dilation: int = 1,
#         inplace: Optional[bool] = True,
#         bias: Optional[bool] = None,
#     ) -> None:

#         super().__init__(
#             in_channels,
#             out_channels,
#             kernel_size,
#             stride,
#             padding,
#             groups,
#             norm_layer,
#             activation_layer,
#             dilation,
#             inplace,
#             bias,
#             torch.nn.Conv2d,
#         )




# utils.py
# import torch
# from typing import Callable, Optional
# from functools import wraps

# def chunk_processor(chunk_size: int = 4000):
#     """分块处理装饰器"""
#     def decorator(func: Callable):
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             src_boxes = args[0]
#             tgt_boxes = args[1] if len(args) > 1 else src_boxes
            
#             B, N1, _ = src_boxes.shape
#             _, N2, _ = tgt_boxes.shape
#             device = src_boxes.device
            
#             # 处理点对点的情况
#             if N1 == N2:
#                 # 初始化结果tensor
#                 result = torch.zeros(B, N1, N1, 6, device=device)
                
#                 # 分块处理
#                 for i in range(0, N1, chunk_size):
#                     i_end = min(i + chunk_size, N1)
#                     chunk = src_boxes[:, i:i_end]
                    
#                     # 处理当前块
#                     out_chunk = func(chunk, chunk, *args[2:])
#                     # 直接写入结果tensor对应位置
#                     result[:, i:i_end, i:i_end] = out_chunk
                    
#                     if torch.cuda.is_available():
#                         torch.cuda.empty_cache()
                
#                 return result
#             else:
#                 # 处理不同尺寸的情况
#                 result = torch.zeros(B, N1, N2, 6, device=device)
                
#                 for i in range(0, N1, chunk_size):
#                     i_end = min(i + chunk_size, N1)
#                     src_chunk = src_boxes[:, i:i_end]
                    
#                     for j in range(0, N2, chunk_size):
#                         j_end = min(j + chunk_size, N2)
#                         tgt_chunk = tgt_boxes[:, j:j_end]
                        
#                         # 处理当前块
#                         out_chunk = func(src_chunk, tgt_chunk, *args[2:])
#                         # 直接写入结果tensor对应位置
#                         result[:, i:i_end, j:j_end] = out_chunk
                        
#                         if torch.cuda.is_available():
#                             torch.cuda.empty_cache()
                
#                 return result
            
#         return wrapper
#     return decorator
# def sine_chunk_processor(chunk_size: int = 4000):
#     """正弦位置编码分块处理装饰器"""
#     def decorator(func: Callable):
#         @wraps(func)
#         def wrapper(pos_tensor: torch.Tensor, *args, **kwargs):
#             chunks = []
#             for i in range(0, pos_tensor.shape[0], chunk_size):
#                 chunk = pos_tensor[i:i + chunk_size]
#                 chunks.append(func(chunk, *args, **kwargs))
#                 if torch.cuda.is_available():
#                     torch.cuda.empty_cache()
                    
#             return torch.cat(chunks, dim=0)
#         return wrapper
#     return decorator
# @chunk_processor(chunk_size=4000)
# # PositionRelationEmbedding
# def box_rel_encoding(src_boxes, tgt_boxes, eps=1e-5):
#     # construct position relation
#     xy1, wh1 = src_boxes.split([2, 2], -1)
#     # xy2, wh2 = tgt_boxes.split([2, 2], -1)
#     delta_xy = torch.abs(xy1.unsqueeze(-2) - xy1.unsqueeze(-3))
#     delta_xy = torch.log(delta_xy / (wh1.unsqueeze(-2) + eps) + 1.0)
#     delta_wh = torch.log((wh1.unsqueeze(-2) + eps) / (wh1.unsqueeze(-3) + eps))
#     # 计算IoU
#     x1y1_1, x2y2_1 = xy1 - wh1/2, xy1 + wh1/2
#     # x1y1_2, x2y2_2 = xy2 - wh2/2, xy2 + wh2/2
    
#     x1y1_inter = torch.maximum(x1y1_1.unsqueeze(-2), x1y1_1.unsqueeze(-3))
#     x2y2_inter = torch.minimum(x2y2_1.unsqueeze(-2), x2y2_1.unsqueeze(-3))
    
#     wh_inter = torch.clamp(x2y2_inter - x1y1_inter, min=0)
#     area_inter = wh_inter.prod(dim=-1)
    
#     area_1 = wh1.prod(dim=-1).unsqueeze(-1)
#     area_2 = wh1.prod(dim=-1).unsqueeze(-2)
    
#     iou = area_inter / (area_1 + area_2 - area_inter + eps)
    
#     # 计算相对距离
#     center1 = xy1
#     # center2 = xy2
#     # rel_distance = torch.norm(center1.unsqueeze(-2) - center2.unsqueeze(-3), dim=-1)
#     # rel_distance = torch.log(rel_distance / torch.sqrt(wh1.prod(dim=-1)).unsqueeze(-1) + eps)
    
#     # 计算相对角度
#     delta_center = center1.unsqueeze(-3) - center1.unsqueeze(-2)
#     angle = torch.atan2(delta_center[..., 1], delta_center[..., 0])
    
#     # 组合所有特征
#     pos_embed = torch.cat([
#         delta_xy,  # 相对位置 (2)
#         delta_wh,  # 相对尺寸 (2)
#         iou.unsqueeze(-1),  # IoU (1)
#         # rel_distance.unsqueeze(-1),  # 相对距离 (1)
#         angle.unsqueeze(-1),  # 相对角度 (1)
#     ], -1)
#     # pos_embed = torch.cat([delta_xy, delta_wh], -1)  # [batch_size, num_boxes1, num_boxes2, 4]

#     return pos_embed



# @functools.lru_cache  # use lru_cache to avoid redundant calculation for dim_t
# def get_dim_t(num_pos_feats: int, temperature: int, device: torch.device):
#     dim_t = torch.arange(num_pos_feats // 2, dtype=torch.float32, device=device)
#     dim_t = temperature**(dim_t * 2 / num_pos_feats)
#     return dim_t  # (0, 2, 4, ..., ⌊n/2⌋*2)

# def exchange_xy_fn(pos_res):
#     index = torch.cat([
#         torch.arange(1, -1, -1, device=pos_res.device),
#         torch.arange(2, pos_res.shape[-2], device=pos_res.device),
#     ])
#     pos_res = torch.index_select(pos_res, -2, index)
#     return pos_res
# @torch.no_grad()
# @sine_chunk_processor(chunk_size=4000)
# def get_sine_pos_embed(
#     pos_tensor: Tensor,
#     num_pos_feats: int = 128,
#     temperature: int = 10000,
#     scale: float = 2 * math.pi,
#     exchange_xy: bool = True,
# ) -> Tensor:
#     """Generate sine position embedding for a position tensor

#     :param pos_tensor: shape as (..., 2*n).
#     :param num_pos_feats: projected shape for each float in the tensor, defaults to 128
#     :param temperature: the temperature used for scaling the position embedding, defaults to 10000
#     :param exchange_xy: exchange pos x and pos. For example,
#         input tensor is [x, y], the results will be [pos(y), pos(x)], defaults to True
#     :return: position embedding with shape (None, n * num_pos_feats)
#     """
#     dim_t = get_dim_t(num_pos_feats, temperature, pos_tensor.device)

#     pos_res = pos_tensor.unsqueeze(-1) * scale / dim_t
#     pos_res = torch.stack((pos_res.sin(), pos_res.cos()), dim=-1).flatten(-2)
#     if exchange_xy:
#         pos_res = exchange_xy_fn(pos_res)
#     pos_res = pos_res.flatten(-2)
#     return pos_res
# class PositionRelationEmbedding(nn.Module):
#     def __init__(
#         self,
#         embed_dim=256,
#         num_heads=8,
#         temperature=10000.,
#         scale=100.,
#         activation_layer=nn.ReLU,
#         inplace=True,
#     ):
#         super().__init__()
#         self.pos_proj = Conv2dNormActivation(
#             embed_dim * 6,
#             num_heads,
#             kernel_size=1,
#             inplace=inplace,
#             norm_layer=None,
#             activation_layer=activation_layer,
#         )
#         self.pos_func = functools.partial(
#             get_sine_pos_embed,
#             num_pos_feats=embed_dim,
#             temperature=temperature,
#             scale=scale,
#             exchange_xy=False,
#         )

#     def forward(self, src_boxes: Tensor, tgt_boxes: Tensor = None):
#         if tgt_boxes is None:
#             tgt_boxes = src_boxes
#         # src_boxes: [batch_size, num_boxes1, 4]
#         # tgt_boxes: [batch_size, num_boxes2, 4]
#         # torch._assert(src_boxes.shape[-1] == 4, f"src_boxes much have 4 coordinates")
#         # torch._assert(tgt_boxes.shape[-1] == 4, f"tgt_boxes must have 4 coordinates")
#         with torch.no_grad():
#             pos_embed = box_rel_encoding(src_boxes, tgt_boxes)
#             pos_embed = self.pos_func(pos_embed).permute(0, 3, 1, 2)
#         pos_embed = self.pos_proj(pos_embed)

#         return pos_embed.clone()
    

    




