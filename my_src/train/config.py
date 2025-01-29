from dataclasses import dataclass

@dataclass
class TrainConfig:
    # 基础训练参数
    batch_size: int = 12
    epochs: int = 2
    learning_rate: float = 3e-4
    
    # 数据集参数
    dataset: str = "arron666/seq_monkey"  # HuggingFace数据集名称
    data_path: str = None  # 本地数据集路径
    train_val_split: float = 0.9  # 训练集验证集分割比例
    
    # 模型保存相关
    run_name: str = "gpt_training"  # 训练运行的名称
    checkpoint_dir: str = "checkpoints"  # 检查点保存目录
    
    # 训练控制参数
    save_steps: int = 2000  # 每多少步保存一次checkpoint
    max_steps: int = 100000  # 最大训练步数
    patience: int = 3  # 早停耐心值
    min_delta: float = 1e-4  # 最小改善阈值
    
    # 优化器和调度器参数
    optimizer: str = "AdamW"  # 优化器类型
    scheduler: str = "CosineAnnealingLR"  # 学习率调度器类型
    scheduler_t_max: int = 1000  # CosineAnnealingLR的T_max参数
    
    # 验证相关参数
    eval_steps: int = 500  # 每多少步验证一次
    eval_epoch: bool = True  # 是否在每个epoch结束时验证

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
    
    # 训练相关配置
    save_steps: int = 2000  # 每多少步保存一次checkpoint
    max_steps: int = 100000  # 最大训练步数
    patience: int = 3  # 早停耐心值
    min_delta: float = 1e-4  # 最小改善阈值
    
    # 学习率相关配置