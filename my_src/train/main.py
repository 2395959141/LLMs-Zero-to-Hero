import argparse
import torch
from torch.utils.data import DataLoader
from config import GPTConfig, TrainConfig
from model import GPT
from dataset import MyDataset, HFDataset
from trainer import train_model
import os

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
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='directory to save model checkpoints')
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
    model_config = GPTConfig()
    train_config = TrainConfig()
    
    # 使用命令行参数更新配置
    train_config = update_config_from_args(train_config, args)
    
    # 创建checkpoint目录
    os.makedirs(train_config.checkpoint_dir, exist_ok=True)
    
    # 初始化模型
    model_config.batch_size = train_config.batch_size  # 确保两个配置的batch_size一致
    model = GPT(model_config)
    
    # 加载数据
    if train_config.data_path:
        full_dataset = MyDataset(train_config.data_path)
    else:
        dataset_dict = HFDataset(train_config.dataset)
        full_dataset = dataset_dict['train']
        
    # 计算分割大小
    train_size = int(train_config.train_val_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # 分割数据集
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_config.batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=train_config.batch_size, 
        shuffle=False
    )
    
    # 训练模型
    model, best_val_loss = train_model(
        model, 
        train_loader, 
        val_loader, 
        train_config,
        num_epochs=train_config.epochs,
        run_name=train_config.run_name
    )
    
    print(f'训练完成！最佳验证损失: {best_val_loss:.4f}')

if __name__ == "__main__":
    main() 