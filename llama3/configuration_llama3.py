from ..utils import ModelConfig


class Llama3Config(ModelConfig):

    model_name = "llama3"

    def __init__(
        self,
        vocab_size: int,
        max_position_embeddings: int = 512,
        rope_theta: float = 10000.0,
        rope_scaling: float = None,
        hidden_size: int = 512,
        head_dim: int = None,
        num_layers: int = 8,
        num_attention_heads: int = 8,
        num_kv_heads: int = 4,
        intermediate_size: int = 2048,
        qkv_bias: bool = False,
        attention_dropout: float = 0.0,
        mlp_bias: bool = False,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        **kwargs
    ) -> None:
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = (
            num_kv_heads if num_kv_heads is not None else self.num_attention_heads
        )

        self.head_dim = (
            head_dim
            if head_dim is not None
            else self.hidden_size // self.num_attention_heads
        )

        self.intermediate_size = intermediate_size
        self.qkv_bias = qkv_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
