"""FLCT with subsegment-based CNN encoding and RoPE self-attention.

The active model definitions are unchanged from comparison_network.py.
d_model is the subsegment size (256 samples), not the latent width.
Inputs: (batch, 1, samples) or (batch, samples); outputs: (batch, samples).
"""

import math

import torch
from torch import nn


class ResConvBlock(nn.Module):
    def __init__(self, cnn_kernel_size, in_feature, out_feature, dropout, stride):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_feature, out_feature, cnn_kernel_size, padding='same', bias=True
        )
        self.conv2 = nn.Conv1d(
            out_feature, out_feature, cnn_kernel_size, padding='same', bias=True
        )
        self.bn1 = nn.BatchNorm1d(out_feature)
        self.bn2 = nn.BatchNorm1d(out_feature)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act1 = nn.PReLU()
        self.act2 = nn.PReLU()
        self.pooling = nn.AvgPool1d(kernel_size=stride)
        self.conv_residual = nn.Conv1d(
            in_feature, out_feature, kernel_size=1, padding='same', bias=True
        )
        self.pooling_residual = nn.AvgPool1d(kernel_size=stride)
        self.in_feature, self.out_feature, self.stride = (in_feature, out_feature, stride)

    def forward(self, x):
        batch, seq_len, d_model = x.shape
        x = x.reshape(batch * seq_len, self.in_feature, int(d_model / self.in_feature))
        residual = self.pooling_residual(self.conv_residual(x))
        x = self.conv1(x)
        x = self.act1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.pooling(x)
        x = x + residual
        x = x.reshape(
            batch, seq_len,
            int(d_model * self.out_feature / (self.in_feature * self.stride)),
        )
        return x


class DeconvResBlock(nn.Module):
    def __init__(self, cnn_kernel_size, in_feature, out_feature, dropout, stride):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.stride = stride
        padding = (cnn_kernel_size - 1) // 2
        self.conv1 = nn.ConvTranspose1d(
            in_feature, out_feature, cnn_kernel_size,
            stride=stride, padding=padding, output_padding=1, bias=True,
        )
        self.conv2 = nn.ConvTranspose1d(
            out_feature, out_feature, cnn_kernel_size, padding=padding, bias=True
        )
        self.bn1 = nn.BatchNorm1d(out_feature)
        self.bn2 = nn.BatchNorm1d(out_feature)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act1 = nn.PReLU()
        self.act2 = nn.PReLU()
        self.conv_residual = nn.ConvTranspose1d(
            in_feature, out_feature, kernel_size=2,
            stride=stride, output_padding=0, bias=True,
        )

    def forward(self, x):
        batch, seq_len, d_model = x.shape
        x = x.reshape(batch * seq_len, self.in_feature, d_model // self.in_feature)
        residual = self.conv_residual(x)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        x = x + residual
        x = x.reshape(
            batch, seq_len,
            d_model * self.out_feature // self.in_feature * self.stride,
        )
        return x

def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0, device='cpu'):
    """预计算旋转因子（复数形式）"""
    freqs = 1.0 / theta ** (torch.arange(0, dim, 2, device=device)[:dim // 2].float() / dim)
    t = torch.arange(seq_len, device=device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def reshape_for_broadcast(freqs_cis, x):
    """调整 freqs_cis 形状以便广播到 x"""
    ndim = x.ndim
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq, xk, freqs_cis):
    """对 Query 和 Key 应用旋转位置编码"""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return (xq_out.type_as(xq), xk_out.type_as(xk))


class MultiHeadAttentionWithRoPE(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, bias=True):
        super().__init__()
        assert d_model % nhead == 0, 'd_model must be divisible by nhead'
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim % 2 == 0, 'head_dim must be even for RoPE'
        self.wq = nn.Linear(d_model, d_model, bias=bias)
        self.wk = nn.Linear(d_model, d_model, bias=bias)
        self.wv = nn.Linear(d_model, d_model, bias=bias)
        self.wo = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.batch_first = True
        self.freqs_cis_cache = {}

    def forward(
        self, query, key, value, attn_mask=None, key_padding_mask=None, is_causal=False
    ):
        batch, seq_len, _ = query.shape
        device = query.device
        cache_key = (seq_len, self.head_dim, device)
        if cache_key not in self.freqs_cis_cache:
            self.freqs_cis_cache[cache_key] = precompute_freqs_cis(
                self.head_dim, seq_len, device=device
            )
        freqs_cis = self.freqs_cis_cache[cache_key]
        q = self.wq(query).view(batch, seq_len, self.nhead, self.head_dim)
        k = self.wk(key).view(batch, seq_len, self.nhead, self.head_dim)
        v = self.wv(value).view(batch, seq_len, self.nhead, self.head_dim)
        q, k = apply_rotary_emb(q, k, freqs_cis)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )
        if is_causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1
            )
            attn_weights = attn_weights + causal_mask
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.wo(out)


class TransformerEncoderLayer_self(nn.Module):
    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
        activation='gelu', bias=True, norm_first=True,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttentionWithRoPE(d_model, nhead, dropout=dropout, bias=bias)
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.norm_first = norm_first

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        """
        显式接受 is_causal 参数，并传递给注意力模块
        """
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal)
            )
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal):
        x = self.self_attn(
            x, x, x, attn_mask=attn_mask,
            key_padding_mask=key_padding_mask, is_causal=is_causal,
        )
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class transformer_denoise(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.res_cnn = nn.Sequential(
            ResConvBlock(cnn_kernel_size=3, in_feature=1, out_feature=8, dropout=0.1, stride=2),
            ResConvBlock(cnn_kernel_size=3, in_feature=8, out_feature=16, dropout=0.2, stride=2),
            ResConvBlock(cnn_kernel_size=3, in_feature=16, out_feature=32, dropout=0.3, stride=2),
            ResConvBlock(cnn_kernel_size=3, in_feature=32, out_feature=64, dropout=0.4, stride=2),
        )
        encoder_layer = TransformerEncoderLayer_self(
            d_model=d_model * 4,
            nhead=nhead,
            dim_feedforward=d_model * 4 * 2,
            dropout=0.2,
            activation='gelu',
            bias=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.decoder = nn.Sequential(
            ResConvBlock(cnn_kernel_size=3, in_feature=64, out_feature=128, dropout=0.5, stride=2),
            DeconvResBlock(3, 128, 64, 0.5, 2),
            DeconvResBlock(3, 64, 32, 0.4, 2),
            DeconvResBlock(3, 32, 16, 0.3, 2),
            DeconvResBlock(3, 16, 8, 0.2, 2),
            nn.LazyLinear(d_model),
        )

    def forward(self, x):
        datanum = x.shape[-1]
        x_reshape = x.reshape(-1, int(datanum / self.d_model), self.d_model)
        x_cnn = self.res_cnn(x_reshape)
        transformer_output = self.transformer_encoder(x_cnn)
        denoise_feature = self.decoder(transformer_output)
        transformer_output_reshape = denoise_feature.reshape(-1, datanum)
        return transformer_output_reshape


FLCT = transformer_denoise
