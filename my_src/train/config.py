from dataclasses import dataclass

@dataclass
class TrainingConfig:
    use_amp: bool = True                # 使用混合精度训练
    gradient_accumulation_steps: int = 2 # 梯度累积步数
    num_workers: int = 1                # 数据加载线程数
    pin_memory: bool = True             # 固定内存
    use_gradient_checkpointing: bool = False  # 梯度检查点
    max_steps: int = 20000              # 最大训练步数
    save_steps: int = 1000              # 每多少步保存一次检查点
    learning_rate: float = 3e-4         # 学习率

@dataclass
class GPTConfig:
    block_size: int = 512   # 这里其实应该是文本的最大长度（max_seq_len）
    batch_size: int = 24
    n_layer: int = 6
    n_head: int = 12
    n_embd: int = 768    # n_embd 也叫 hidden_dim, hiden_size, 这里同时设置了和 embed_dim 一样
    head_size: int = n_embd // n_head
    dropout: float = 0.1
    vocab_size: int = 50257  # tiktoken 使用的是 GPT-2 的词表 
    training: TrainingConfig = TrainingConfig() 