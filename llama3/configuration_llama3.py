from ..utils import ModelConfig


class Llama3Config(ModelConfig):

    model_name = "llama3"

    def __init__(
        self,
        vocab_size: int,
        max_batch_size: int = 4,
        max_seq_len: int = 512,
        rope_method: str = "llama3",
        rope_theta: float = 10000.0,
        rope_scaling: bool = False,
        rope_scale_factor: float = 8.0,
        rope_low_freq_factor: float = 1.0,
        rope_high_freq_factor: float = 4.0,
        rope_old_context_len: int = 8192,
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
        """关于 Llama3 模型的配置信息

        Args:
            vocab_size (int): 词汇表大小
            max_batch_size (int, optional): 最大 batch_size, 用于预留 kv cache 空间. Defaults to 4.
            max_seq_len (int, optional): 最大文本长度. Defaults to 512.
            rope_method (str, optional): RoPE 的实现方式，可选 "llam3" 或 "transformers". Defaults to "llama3".
            rope_theta (float, optional): RoPE 超参数 theta. Defaults to 10000.0.
            rope_scaling (bool, optional): 是否在 RoPE 中应用 scaling. Defaults to False.
            rope_scale_factor (float, optional): RoPE 超参数 scale_factor. Defaults to 8.0.
            rope_low_freq_factor (float, optional): RoPE 超参数 low_freq_factor. Defaults to 1.0.
            rope_high_freq_factor (float, optional): RoPE 超参数 high_freq_factor. Defaults to 4.0.
            rope_old_context_len (int, optional): RoPE 超参数 old_context_len. Defaults to 8192.
            hidden_size (int, optional): 隐藏层的维度. Defaults to 512.
            head_dim (int, optional): 注意力头的维度. Defaults to None.
            num_layers (int, optional): 解码器的层数. Defaults to 8.
            num_attention_heads (int, optional): GQA 中 query 头的数量. Defaults to 8.
            num_kv_heads (int, optional): GQA 中 key 或 value 头的数量. Defaults to 4.
            intermediate_size (int, optional): FFN 中隐藏层的维度. Defaults to 2048.
            qkv_bias (bool, optional): attention 是否包括 bias 参数. Defaults to False.
            attention_dropout (float, optional): dropout 比率. Defaults to 0.0.
            mlp_bias (bool, optional): embedding 和 FFN 是否包括 bias 参数. Defaults to False.
            initializer_range (float, optional): 初始化参数时的方差. Defaults to 0.02.
            rms_norm_eps (float, optional): RMS Norm 中避免分母为 0 的附加值. Defaults to 1e-6.
        """
        
        
        super().__init__(**kwargs)
        
        self.vocab_size = vocab_size
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        self.rope_scaling = rope_scaling
        self.rope_method = rope_method
        self.rope_theta = rope_theta
        self.rope_scale_factor = rope_scale_factor
        self.rope_low_freq_factor = rope_low_freq_factor
        self.rope_high_freq_factor = rope_high_freq_factor
        self.rope_old_context_len = rope_old_context_len

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
