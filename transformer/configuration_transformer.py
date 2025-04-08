from ..utils import ModelConfig


class TransformerConfig(ModelConfig):

    model_name = "transformer"

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        hidden_size: int = 512,
        max_position_embeddings: int = 512,
        num_layers: int = 6,
        num_attention_heads: int = 8,
        attention_dropout: float = 0.1,
        intermediate_size: int = 2048,
        hidden_dropout: float = 0.1,
        qkv_bias: bool = False,
        **kwargs
    ) -> None:
        """关于 Transformer 模型的配置信息

        Args:
            src_vocab_size (int): 源文本的词汇表大小
            tgt_vocab_size (int): 目标文本的词汇表大小
            hidden_size (int, optional): 隐藏层的维度. Defaults to 512.
            max_position_embeddings (int, optional): 最大文本长度. Defaults to 512.
            num_layers (int, optional): 编码器和解码器的层数. Defaults to 6.
            num_attention_heads (int, optional): 多头注意力机制中头的数量. Defaults to 8.
            attention_dropout (float, optional): dropout 比率. Defaults to 0.1.
            intermediate_size (int, optional): FFN 中的隐藏层维度. Defaults to 2048.
            hidden_dropout (float, optional): dropout 比率. Defaults to 0.1.
            qkv_bias (bool, optional): attention 是否包括 bias 参数. Defaults to False.
        """

        super().__init__(**kwargs)

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.intermediate_size = intermediate_size
        self.qkv_bias = qkv_bias
