from dataclasses import dataclass

@dataclass
class TrainConfig:
    # 基础训练参数
    batch_size: int = 16  # 每个批次的样本数
    epochs: int = 2
    learning_rate: float = 3e-4
    
    # 数据集参数
    dataset: str = "arron666/seq_monkey"  # HuggingFace数据集名称
    data_path: str = None  # 本地数据集路径
    train_val_split: float = 0.9  # 训练集验证集分割比例
    
    # 模型保存相关
    run_name: str = "gpt_training"  # 训练运行的名称
    checkpoint_dir: str = "checkpoints"  # 检查点保存目录
    save_steps: int = 50  # 每50步保存一次
    save_best_only: bool = True  # 是否只保存最佳模型
    max_checkpoints: int = 10  # 最多保存多少个checkpoint
    
    # 训练优化参数
    grad_clip: float = 1.0  # 梯度裁剪参数
    weight_decay: float = 0.1  # 权重衰减参数
    gradient_accumulation_steps: int = 8  # 梯度累积步数
    max_steps: int = 20000  # 最大训练步数
    
    # 验证相关参数
    eval_steps: int = 1000  # 每多少步验证一次
    eval_epoch: bool = True  # 是否在每个epoch结束时验证
    
    # 性能优化参数
    use_amp: bool = True  # 使用混合精度训练
    num_workers: int = 4  # 数据加载器的工作进程数
    pin_memory: bool = True  # 固定内存
    use_gradient_checkpointing: bool = False  # 梯度检查点

@dataclass
class GPTConfig:
    block_size: int = 1024   
    n_layer: int = 6   
    n_head: int = 12  
    n_embd: int = 768    
    head_size: int = n_embd // n_head
    dropout: float = 0.1
    vocab_size: int = 21128  # BERT中文词表大小
    batch_size: int = 24
    training: TrainConfig = TrainConfig() 