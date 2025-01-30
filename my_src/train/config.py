from dataclasses import dataclass

@dataclass
class TrainConfig:
    # 基础训练参数
    batch_size: int = 8  # 每个批次的样本数
    epochs: int = 2
    learning_rate: float = 3e-4
    
    # 数据集参数
    dataset: str = "arron666/seq_monkey"  # HuggingFace数据集名称
    data_path: str = None  # 本地数据集路径
    train_val_split: float = 0.9  # 训练集验证集分割比例
    
    # 模型保存相关
    run_name: str = "gpt_training"  # 训练运行的名称
    checkpoint_dir: str = "checkpoints"  # 检查点保存目录
    
    # # 训练控制参数
    # save_steps: int = 5000  # 每多少步保存一次checkpoint
    # max_steps: int = 50000  # 修改为100000步
    # patience: int = 3  # 早停耐心值
    # min_delta: float = 1e-4  # 最小改善阈值
    
    # # 优化器和调度器参数
    # optimizer: str = "AdamW"  # 优化器类型
    # scheduler: str = "CosineAnnealingLR"  # 学习率调度器类型
    # scheduler_t_max: int = 50000  # 修改为与max_steps相同
    grad_clip: float = 1.0  # 添加梯度裁剪参数
    weight_decay: float = 0.1  # 权重衰减参数
    
    # 验证相关参数
    eval_steps: int = 500  # 每多少步验证一次
    eval_epoch: bool = True  # 是否在每个epoch结束时验证
    
    # 新增梯度累积参数
    gradient_accumulation_steps: int = 4  # 梯度累积步数
    # 实际的有效批次大小 = batch_size * gradient_accumulation_steps = 40
    
    # checkpoint相关
    save_steps: int = 1000  # 每多少步保存一次
    save_best_only: bool = True  # 是否只保存最佳模型
    max_checkpoints: int = 5  # 最多保存多少个checkpoint

    # # 其他训练参数
    # warmup_steps: int = 2000  # 预热步数
    # lr_decay: bool = True  # 是否使用学习率衰减
    num_workers: int = 4  # 数据加载器的工作进程数
    # log_interval: int = 100  # 日志打印间隔

    def __post_init__(self):
        self.gradient_accumulation_steps = 4  # 设置梯度累积步数

@dataclass
class GPTConfig:
    block_size: int = 1024   # 这里其实应该是文本的最大长度（max_seq_len）
    n_layer: int = 6   
    n_head: int = 12  
    n_embd: int = 768    # n_embd 也叫 hidden_dim, hiden_size, 这里同时设置了和 embed_dim 一样
    head_size: int = n_embd // n_head
    dropout: float = 0.1
    vocab_size: int = 50257  # tiktoken 使用的是 GPT-2 的词表
    
    # # 训练相关配置
    # save_steps: int = 5000  # 每多少步保存一次checkpoint
    # max_steps: int = 30000  # 修改为100000步
    # patience: int = 3  # 早停耐心值
    # min_delta: float = 1e-4  # 最小改善阈值
    
    # 学习率相关配置