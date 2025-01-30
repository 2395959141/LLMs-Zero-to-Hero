import argparse
import torch
from torch.utils.data import DataLoader
from config import GPTConfig, TrainConfig
from model import GPT
from dataset import MyDataset, HFDataset
from trainer import train_model
import os
import warnings

# 检查 PyTorch 版本
if torch.__version__ < "2.0.0":
    warnings.warn("torch.compile() 需要 PyTorch 2.0 或更高版本。当前版本将不会启用编译优化。")

def parse_args():
    parser = argparse.ArgumentParser(description='Train a GPT model')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=None,
                        help='learning rate')
    parser.add_argument('--dataset', type=str, default=None,
                        help='dataset name on Hugging Face')
    parser.add_argument('--data-path', type=str, default=None,
                        help='path to local dataset directory')
    parser.add_argument('--run-name', type=str, default=None,
                        help='name for this training run')
    parser.add_argument('--checkpoint-dir', type=str, default="checkpoints",
                        help='directory to save model checkpoints')
    parser.add_argument('--compile-mode', type=str, default="default",
                      choices=["default", "reduce-overhead", "max-autotune"],
                      help='torch.compile的优化模式')
    parser.add_argument('--no-compile', action='store_true',
                      help='禁用模型编译优化')
    return parser.parse_args()

def update_config_from_args(config, args):
    """使用命令行参数更新配置"""
    for key, value in vars(args).items():
        if value is not None:  # 只更新显式指定的参数
            key = key.replace('-', '_')  # 将命令行参数名转换为配置属性名
            if hasattr(config, key):
                setattr(config, key, value)
    return config

def main():
    args = parse_args()
    
    # 初始化配置
    model_config = GPTConfig(vocab_size=50304) ##让大矩阵能够更好地被CUDA加速
    train_config = TrainConfig()
    
    # 使用命令行参数更新配置
    train_config = update_config_from_args(train_config, args)
    
    # 创建checkpoint目录
    os.makedirs(train_config.checkpoint_dir, exist_ok=True)
    
    # 初始化模型
    model_config.batch_size = train_config.batch_size  # 确保两个配置的batch_size一致
    model = GPT(model_config)
    
    # 使用 torch.compile() 编译模型以提升性能
    if not args.no_compile and torch.__version__ >= "2.0.0":
        try:
            print(f"正在使用 {args.compile_mode} 模式编译模型...")
            model = torch.compile(model, mode=args.compile_mode)
            print("模型编译完成！")
        except Exception as e:
            print(f"模型编译失败，将使用原始模型继续训练。错误信息: {str(e)}")
    
    # 加载数据
    if train_config.data_path:
        full_dataset = MyDataset(train_config.data_path, block_size=model_config.block_size)
        print(f"本地数据集大小：{len(full_dataset)}")
    else:
        dataset_dict = HFDataset(train_config.dataset, block_size=model_config.block_size)
        print(f"HuggingFace数据集加载完成")
        print(f"数据集大小：{len(dataset_dict)}")
        if hasattr(dataset_dict, 'column_names'):
            print(f"数据集列：{dataset_dict.column_names}")
        full_dataset = dataset_dict
        
    # 计算分割大小
    train_size = int(len(full_dataset) * train_config.train_val_split)
    val_size = len(full_dataset) - train_size
    
    # 分割数据集
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size]
    )
    
    # 在创建数据加载器之前打印数据集大小
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,  # 添加shuffle=True来打乱训练数据
        num_workers=4,  # 添加多进程加载
        pin_memory=True  # 使用PIN_MEMORY来加速数据传输
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,  # 验证集不需要打乱
        num_workers=4,
        pin_memory=True
    )

    # 训练模型
    model, best_val_loss = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_loader=train_loader,  # 传入DataLoader
        val_loader=val_loader,      # 传入DataLoader
        config=train_config,
        num_epochs=train_config.epochs,
        run_name=args.run_name
    )
    
    print(f'训练完成！最佳验证损失: {best_val_loss:.4f}')

if __name__ == "__main__":
    main() 