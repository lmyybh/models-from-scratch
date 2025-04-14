from typing import Optional, Union
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .configuration_transformer import TransformerConfig


class PositionalEncoding(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        """正弦位置编码

        Args:
            config (TransformerConfig): 关于 Transformer 模型的配置信息
        """
        super().__init__()

        # shape: [max_len, 1]
        pos = torch.arange(0, config.max_position_embeddings).unsqueeze(1)
        div_term = 1000 ** (
            -torch.arange(0, config.hidden_size, 2) / config.hidden_size
        )  # shape: [d / 2]

        pe = torch.zeros(config.max_position_embeddings, config.hidden_size)
        pe[:, 0::2] = torch.sin(pos * div_term)  # shape: [max_len, d / 2]
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, max_len, d]

        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """前向计算

        Args:
            x (Tensor): 序列编码, shape: [bz, len, d]

        Returns:
            Tensor: 添加位置编码后的序列编码
        """
        return x + self.pe[:, : x.size()[1]]


class TokenEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        """token 编码层

        Args:
            vocab_size (int): 词汇表大小
            hidden_size (int): 编码后的维度
        """

        super().__init__()
        self.lut = nn.Embedding(vocab_size, hidden_size)
        self.scaling = hidden_size**0.5

    def forward(self, tokens: Tensor) -> Tensor:
        """前向计算

        Args:
            tokens (Tensor): 序列 tokens

        Returns:
            Tensor: 序列编码
        """
        return self.lut(tokens) * self.scaling


def generate_causal_mask(size: int) -> Tensor:
    """生成 casusal mask

    Args:
        size (int): attention 矩阵维度

    Returns:
        Tensor: 下三角矩阵，0 表示显示，1 表示遮挡
    """
    return torch.triu(torch.ones((size, size)), diagonal=1)


def attention_forward(
    module: nn.Module,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scaling: float,
    dropout: float = 0.0,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """attention 计算

    Args:
        module (nn.Module):
        query (Tensor): [bz, h, len_q, d_q]
        key (Tensor): [bz, h, len_k, d_k]
        value (Tensor): [bz, h, len_v, d_v]
        scaling (float): 缩放系数，一般为 d_k**-0.5
        dropout (float, optional): dropout 比率. Defaults to 0.0.
        mask (Optional[Tensor], optional): 应用于 attention 矩阵的 mask，1 代表遮挡. Defaults to None.

    Returns:
        Tensor: attention 计算结果，[bz, h, len_q, d_v]
    """
    # w = q @ k.T / sqrt(d_k), shape: [bz, h, len_q, len_k]
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling
    if mask is not None:
        attn_weights.masked_fill_(
            mask, float("-inf")
        )  # 将遮挡区域填充 -inf，后续经 softmax 可变为 0

    # s = softmax(w), shape: [bz, h, len_q, len_k]
    attn_weights = torch.softmax(attn_weights, dim=-1)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)

    # out = s @ v, shape: [bz, h, len_q, d_v]
    attn_output = torch.matmul(attn_weights, value)

    return attn_output


class MultiHeadAttention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        """多头注意力

        Args:
            config (TransformerConfig): 关于 Transformer 模型的配置信息
        """
        super().__init__()

        # head 的个数：h = num_attention_heads
        self.num_attention_heads = config.num_attention_heads
        # 每个 head 的维度: d = d_q = d_k = d_v = hidden_size / h
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        # 所有 head 合并后的维度: h * d
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # 缩放系数：sqrt(d_k)
        self.scaling = self.attention_head_size**-0.5
        # attention 中的 dropout 比率
        self.attention_dropout = config.attention_dropout

        # 所有 head 的 query 参数矩阵, shape: [hidden_size, h * d]
        self.query = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.qkv_bias
        )
        # 所有 head 的 key 参数矩阵, shape: [hidden_size, h * d]
        self.key = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.qkv_bias
        )
        # 所有 head 的 value 参数矩阵, shape: [hidden_size, h * d]
        self.value = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.qkv_bias
        )

        # MHA 输出时的 linear 层
        self.linear = nn.Linear(self.all_head_size, config.hidden_size)
        # linear 后的 dropout 比率
        self.dropout = nn.Dropout(config.hidden_dropout)

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        """调整多头 q, k, v 的维度，用于 attention 计算

        Args:
            x (Tensor): 输入的 query 或 key 或 value, shape: [bz, len, h * d]

        Returns:
            Tensor: 调整维度后的 Tensor, shape: [bz, h, len, d]
        """
        # shape: [bz, len, h * d] -> [bz, len, h, d]
        new_size = x.size()[:2] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_size)

        # shape: [bz, len, h, d] -> [bz, h, len, d]
        return x.permute(0, 2, 1, 3)

    def merge_masks(
        self, key_padding_mask: Optional[Tensor], attn_mask: Optional[Tensor]
    ) -> Union[Tensor, None]:
        """合并 key_padding_mask 和 attn_mask, 并调整维度到 [bz, h, 1 or len_q, len_k], 用于广播应用到 attention 矩阵

        Args:
            key_padding_mask (Optional[Tensor]): key 的 padding mask, shape: [bz, len_k], 可以为 None
            attn_mask (Optional[Tensor]): 下三角 causal mask, shape: [len_q, len_k], 可以为 None
        Returns:
            Tensor | None: 合并后的 mask, shape: [bz, h, 1 or len_q, len_k]
        """
        if key_padding_mask is None and attn_mask is None:
            return None

        mask = None
        if key_padding_mask is not None:
            # shape: [bz, len_k] -> [bz, 1, 1, len_k] -> [bz, h, 1, len_k]
            bz, len_k = key_padding_mask.shape
            key_padding_mask = key_padding_mask.view(bz, 1, 1, len_k).expand(
                -1, self.num_attention_heads, -1, -1
            )
            mask = key_padding_mask

        if attn_mask is not None:
            # shape: [len_q, len_k] -> [1, 1, len_q, len_k] -> [bz, h, len_q, len_k]
            attn_mask = attn_mask.view(
                1, 1, attn_mask.size()[0], attn_mask.size()[1]
            ).expand(bz, self.num_attention_heads, -1, -1)

            if mask is None:
                mask = attn_mask
            else:
                mask = mask.logical_or(attn_mask)

        return mask

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """前向计算

        Args:
            q (Tensor): 输入的 query, shape: [bz, len_q, h*d]
            k (Tensor): 输入的 key, shape: [bz, len_k, h*d]
            v (Tensor): 输入的 value, shape: [bz, len_v, h*d]
            key_padding_mask (Optional[Tensor], optional): key 的 padding mask, shape: [bz, len_k]. Defaults to None.
            attn_mask (Optional[Tensor], optional): 下三角 causal mask, shape: [len_q, len_k]. Defaults to None.

        Returns:
            Tensor: MHA 输出 Tensor, shape: [bz, len, hidden_size]
        """
        # 生成多头的 query, key, value
        q = self.transpose_for_scores(self.query(q))  # [bz, h, len_q, d]
        k = self.transpose_for_scores(self.key(k))  # [bz, h, len_k, d]
        v = self.transpose_for_scores(self.value(v))  # [bz, h, len_v, d]

        # 合并两种 mask
        mask = self.merge_masks(key_padding_mask, attn_mask)

        # 计算 attention, shape: [bz, h, len_q, d]
        output = attention_forward(
            self,
            q,
            k,
            v,
            self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            mask=mask,
        )
        # shape: [bz, len_q, h, d] -> [bz, len_q, h * d]
        output = output.transpose(1, 2).contiguous()
        new_size = output.size()[:2] + (self.all_head_size,)
        output = output.reshape(new_size)

        # linear 层, shape: [bz, len_q, h * d] -> [bz, len_q, hidden_size]
        output = self.linear(output)
        output = self.dropout(output)

        return output


class FeedForwardNetworks(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        """前向网络

        Args:
            config (TransformerConfig): 关于 Transformer 模型的配置信息
        """
        super().__init__()

        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, x: Tensor) -> Tensor:
        """前向计算

        Args:
            x (Tensor): 输入 Tensor, shape: [bz, len, hidden_size]

        Returns:
            Tensor: 输出 Tensor, shape: [bz, len, hidden_size]
        """

        # 两个 Linear 层，维度变化为 hidden_size -> intermediate_size -> hidden_size
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.dropout(x)

        return x


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        """LayerNorm

        Args:
            normalized_shape (int): 归一化的维度
            eps (float, optional): 避免分母为 0 的附加值. Defaults to 1e-5.
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))  # 缩放参数
        self.beta = nn.Parameter(torch.zeros(normalized_shape))  # 偏置参数
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """前向计算

        Args:
            x (Tensor): 输入 Tensor

        Returns:
            Tensor: 归一化后的 Tensor
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x = (x - mean) / std + self.eps

        return x * self.gamma + self.beta


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        """Transformer 编码层

        Args:
            config (TransformerConfig): 关于 Transformer 模型的配置信息
        """
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.norm1 = LayerNorm(config.hidden_size)
        self.ffn = FeedForwardNetworks(config)
        self.norm2 = LayerNorm(config.hidden_size)

    def forward(
        self,
        src: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        src_attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """前向计算

        Args:
            src (Tensor): 源序列 Tensor
            src_key_padding_mask (Optional[Tensor], optional): src 的 padding mask, shape: [bz, len_src]. Defaults to None.
            src_attn_mask (Optional[Tensor], optional): 下三角 causal mask, shape: [len_src, len_src]. Defaults to None.

        Returns:
            Tensor: 编码层输出 Tensor, shape: [bz, len_src, hidden_size]
        """
        # 多头注意力 + 残差连接 + 归一化
        residual = src
        x = self.attention(
            q=src,
            k=src,
            v=src,
            key_padding_mask=src_key_padding_mask,
            attn_mask=src_attn_mask,
        )
        x += residual
        x = self.norm1(x)

        # 前向网络 + 残差连接 + 归一化
        residual = x
        x = self.ffn(x)
        x += residual
        x = self.norm2(x)

        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        """Transformer 解码层

        Args:
            config (TransformerConfig): 关于 Transformer 模型的配置信息
        """
        super().__init__()

        self.self_attention = MultiHeadAttention(config)
        self.norm1 = LayerNorm(config.hidden_size)

        self.cross_attention = MultiHeadAttention(config)
        self.norm2 = LayerNorm(config.hidden_size)

        self.ffn = FeedForwardNetworks(config)
        self.norm3 = LayerNorm(config.hidden_size)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_key_padding_mask: Optional[Tensor] = None,
        tgt_attn_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        memory_attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """前向计算

        Args:
            tgt (Tensor): 目标序列 Tensor
            memory (Tensor): 编码层输出的 Tensor
            tgt_key_padding_mask (Optional[Tensor], optional): tgt 的 padding mask, shape: [bz, len_tgt]. Defaults to None.
            tgt_attn_mask (Optional[Tensor], optional): 自注意力中的下三角 causal mask, shape: [len_tgt, len_tgt]. Defaults to None.
            memory_key_padding_mask (Optional[Tensor], optional): memory 的 padding mask, shape: [bz, len_src]. Defaults to None.
            memory_attn_mask (Optional[Tensor], optional): 交叉注意力中的下三角 causal mask, shape: [len_tgt, len_src]. Defaults to None.

        Returns:
            Tensor: 解码层输出 Tensor, shape: [bz, len_tgt, hidden_size]
        """
        # 自注意力 + 残差连接 + 归一化
        residual = tgt
        x = self.self_attention(
            q=tgt,
            k=tgt,
            v=tgt,
            key_padding_mask=tgt_key_padding_mask,
            attn_mask=tgt_attn_mask,
        )
        x += residual
        x = self.norm1(x)

        # 交叉注意力 + 残差连接 + 归一化
        residual = x
        x = self.cross_attention(
            q=x,
            k=memory,
            v=memory,
            key_padding_mask=memory_key_padding_mask,
            attn_mask=memory_attn_mask,
        )
        x += residual
        x = self.norm2(x)

        # 前向网络 + 残差连接 + 归一化
        residual = x
        x = self.ffn(x)
        x += residual
        x = self.norm3(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        """Transformer 编码器

        Args:
            config (TransformerConfig): 关于 Transformer 模型的配置信息
        """
        super().__init__()

        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(config) for _ in range(config.num_layers)]
        )

    def forward(
        self,
        src: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        src_attn_mask: Optional[Tensor] = None,
    ) -> tuple:
        """前向计算

        Args:
            src (Tensor): 源序列 Tensor
            src_key_padding_mask (Optional[Tensor], optional): src 的 padding mask, shape: [bz, len_src]. Defaults to None.
            src_attn_mask (Optional[Tensor], optional): 下三角 causal mask, shape: [len_src, len_src]. Defaults to None.

        Returns:
            tuple: 所有编码层输出的编码 Tensor 组成的元组
        """
        x = src
        all_hidden_states = ()
        for i, layer in enumerate(self.layers):
            x = layer(x, src_key_padding_mask, src_attn_mask)

            all_hidden_states += (x,)

        return all_hidden_states


class TransformerDecoder(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        """Transformer 解码器

        Args:
            config (TransformerConfig): 关于 Transformer 模型的配置信息
        """
        super().__init__()

        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(config) for _ in range(config.num_layers)]
        )

    def forward(
        self,
        tgt: Tensor,
        memory_tuple: tuple,
        tgt_key_padding_mask: Optional[Tensor] = None,
        tgt_attn_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        memory_attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """_summary_

        Args:
            tgt (Tensor): 目标序列 Tensor
            memory_tuple (tuple): 编码器输出的所有编码 Tensor 组成的元组
            tgt_key_padding_mask (Optional[Tensor], optional): tgt 的 padding mask, shape: [bz, len_tgt]. Defaults to None.
            tgt_attn_mask (Optional[Tensor], optional): 自注意力中的下三角 causal mask, shape: [len_tgt, len_tgt]. Defaults to None.
            memory_key_padding_mask (Optional[Tensor], optional): memory 的 padding mask, shape: [bz, len_src]. Defaults to None.
            memory_attn_mask (Optional[Tensor], optional): 交叉注意力中的下三角 causal mask, shape: [len_tgt, len_src]. Defaults to None.

        Returns:
            Tensor: 解码器输出 Tensor, shape: [bz, len_tgt, hidden_size]
        """
        x = tgt
        for i, layer in enumerate(self.layers):
            x = layer(
                x,
                memory_tuple[i],
                tgt_key_padding_mask,
                tgt_attn_mask,
                memory_key_padding_mask,
                memory_attn_mask,
            )

        return x


class TransformerOutputLayer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        """Transformer 输出层

        Args:
            config (TransformerConfig): 关于 Transformer 模型的配置信息
        """
        super().__init__()
        self.linear = nn.Linear(config.hidden_size, config.tgt_vocab_size)

    def forward(self, hidden_state: Tensor) -> Tensor:
        """前向计算

        Args:
            hidden_state (Tensor): 编码器输出的 Tensor, shape: [bz, len_tgt, hidden_size]

        Returns:
            Tensor: 输出 Tensor, shape: [bz, len_tgt, tgt_vocab_size]
        """
        return self.linear(hidden_state)


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        """Transformer

        Args:
            config (TransformerConfig): 关于 Transformer 模型的配置信息
        """
        super().__init__()
        self.max_length = config.max_position_embeddings
        self.input_embeddings = TokenEmbeddings(
            config.src_vocab_size, config.hidden_size
        )
        self.output_embeddings = TokenEmbeddings(
            config.tgt_vocab_size, config.hidden_size
        )
        self.positional_encoding = PositionalEncoding(config)
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)
        self.output = TransformerOutputLayer(config)

        self._init_weights()

    def _init_weights(self) -> None:
        """初始化模型参数"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

    def encode(
        self,
        src: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        src_attn_mask: Optional[Tensor] = None,
    ) -> tuple:
        """编码函数

        Args:
            src (Tensor): 源序列 tokens
            src_key_padding_mask (Optional[Tensor], optional): src 的 padding mask, shape: [bz, len_src]. Defaults to None.
            src_attn_mask (Optional[Tensor], optional): 下三角 causal mask, shape: [len_src, len_src]. Defaults to None.

        Returns:
            tuple: 所有编码层输出的编码 Tensor 组成的元组
        """
        src = self.input_embeddings(src)
        src = self.positional_encoding(src)

        encoder_outputs = self.encoder(src, src_key_padding_mask, src_attn_mask)

        return encoder_outputs

    def decode(
        self,
        tgt: Tensor,
        memory_tuple: tuple,
        tgt_key_padding_mask: Optional[Tensor] = None,
        tgt_attn_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        memory_attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """_summary_

        Args:
            tgt (Tensor): 目标序列 tokens
            memory_tuple (tuple): 编码器输出的所有编码 Tensor 组成的元组
            tgt_key_padding_mask (Optional[Tensor], optional): tgt 的 padding mask, shape: [bz, len_tgt]. Defaults to None.
            tgt_attn_mask (Optional[Tensor], optional): 自注意力中的下三角 causal mask, shape: [len_tgt, len_tgt]. Defaults to None.
            memory_key_padding_mask (Optional[Tensor], optional): memory 的 padding mask, shape: [bz, len_src]. Defaults to None.
            memory_attn_mask (Optional[Tensor], optional): 交叉注意力中的下三角 causal mask, shape: [len_tgt, len_src]. Defaults to None.

        Returns:
            Tensor: Transformer 输出的 Tensor, shape: [bz, len_tgt, tgt_vocab_size]
        """
        tgt = self.output_embeddings(tgt)
        tgt = self.positional_encoding(tgt)
        output = self.decoder(
            tgt,
            memory_tuple,
            tgt_key_padding_mask,
            tgt_attn_mask,
            memory_key_padding_mask,
            memory_attn_mask,
        )
        output = self.output(output)

        return output

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        src_attn_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        tgt_attn_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        memory_attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """训练时的前向计算

        Args:
            src (Tensor): 源序列 tokens
            tgt (Tensor): 目标序列 tokens
            src_key_padding_mask (Optional[Tensor], optional): src 的 padding mask, shape: [bz, len_src]. Defaults to None.
            src_attn_mask (Optional[Tensor], optional): 下三角 causal mask, shape: [len_src, len_src]. Defaults to None.
            tgt_key_padding_mask (Optional[Tensor], optional): tgt 的 padding mask, shape: [bz, len_tgt]. Defaults to None.
            tgt_attn_mask (Optional[Tensor], optional): 自注意力中的下三角 causal mask, shape: [len_tgt, len_tgt]. Defaults to None.
            memory_key_padding_mask (Optional[Tensor], optional): memory 的 padding mask, shape: [bz, len_src]. Defaults to None.
            memory_attn_mask (Optional[Tensor], optional): 交叉注意力中的下三角 causal mask, shape: [len_tgt, len_src]. Defaults to None.

        Returns:
            Tensor: Transformer 输出的 Tensor, shape: [bz, len_tgt, tgt_vocab_size]
        """
        encoder_outputs = self.encode(src, src_key_padding_mask, src_attn_mask)

        output = self.decode(
            tgt,
            encoder_outputs,
            tgt_key_padding_mask,
            tgt_attn_mask,
            memory_key_padding_mask,
            memory_attn_mask,
        )

        return output

    @torch.inference_mode()
    def inference(
        self,
        src: Tensor,
        tgt_start_token_id: int,
        tgt_end_token_id: int,
        src_key_padding_mask: Optional[Tensor] = None,
        src_attn_mask: Optional[Tensor] = None,
    ) -> list:
        """推理时的前向计算

        Args:
            src (Tensor): 源序列 tokens
            tgt_start_token_id (int): 目标序列的起始 token id
            tgt_end_token_id (int): 目标序列的结束 token id
            src_key_padding_mask (Optional[Tensor], optional): src 的 padding mask, shape: [bz, len_src]. Defaults to None.
            src_attn_mask (Optional[Tensor], optional): 下三角 causal mask, shape: [len_src, len_src]. Defaults to None.

        Returns:
            list: 生成的 token_ids 列表
        """
        assert src.shape[0] == 1, "batch_size must be 1"
        device = src.device

        memory_tuple = self.encode(src, src_key_padding_mask, src_attn_mask)

        tgt_tokens = [tgt_start_token_id]

        for _ in range(self.max_length):
            tgt = torch.LongTensor([tgt_tokens]).to(device)
            tgt_padding_mask = torch.zeros_like(tgt, device=device, dtype=torch.bool)
            causal_mask = generate_causal_mask(tgt.size()[1]).to(device)

            output = self.decode(
                tgt,
                memory_tuple,
                tgt_key_padding_mask=tgt_padding_mask,
                tgt_attn_mask=causal_mask,
                memory_key_padding_mask=src_key_padding_mask,
                memory_attn_mask=None,
            )[0]
            token_id = output[-1].argmax().item()
            tgt_tokens.append(token_id)

            if token_id == tgt_end_token_id:
                break

        return tgt_tokens


if __name__ == "__main__":
    from ..utils import generate_batch_text_tokens

    config = TransformerConfig()

    device = torch.device(0)
    pad_index = 0

    src = generate_batch_text_tokens(
        [4, 8, 6], max_len=10, vocab_size=config.src_vocab_size, pad_index=pad_index
    ).to(device)
    src_padding_mask = src == pad_index

    tgt = generate_batch_text_tokens(
        [6, 8, 12], max_len=12, vocab_size=config.tgt_vocab_size, pad_index=pad_index
    ).to(device)
    tgt_padding_mask = tgt == pad_index

    causal_mask = generate_causal_mask(tgt.size()[1]).to(device)

    model = Transformer(config).to(device)

    output = model.forward(
        src,
        tgt,
        src_key_padding_mask=src_padding_mask,
        src_attn_mask=None,
        tgt_key_padding_mask=tgt_padding_mask,
        tgt_attn_mask=causal_mask,
        memory_key_padding_mask=src_padding_mask,
        memory_attn_mask=None,
    )
    print(output.shape)
