from ..utils import ModelConfig


class TransformerConfig(ModelConfig):

    model_name = "transformer"

    def __init__(
        self,
        src_vocab_size=1000,
        tgt_vocab_size=1200,
        hidden_size=512,
        max_position_embeddings=512,
        num_layers=6,
        num_attention_heads=8,
        attention_dropout=0.1,
        intermediate_size=2048,
        hidden_dropout=0.1,
        qkv_bias=False,
        **kwargs
    ):
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
