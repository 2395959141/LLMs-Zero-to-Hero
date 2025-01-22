from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 512   # 这里其实应该是文本的最大长度（max_seq_len）
    batch_size: int = 12
    n_layer: int = 6
    n_head: int = 12
    n_embd: int = 768    # n_embd 也叫 hidden_dim, hiden_size, 这里同时设置了和 embed_dim 一样
    head_size: int = n_embd // n_head
    dropout: float = 0.1
    vocab_size: int = 50257  # tiktoken 使用的是 GPT-2 的词表 