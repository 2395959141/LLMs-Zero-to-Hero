import argparse
import torch
from torch.utils.data import DataLoader
from config import GPTConfig
from model import GPT
from dataset import MyDataset, load_dataset
from trainer import train_model
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Train a GPT model')
    parser.add_argument('--batch-size', type=int, default=12,
                        help='batch size for training')
    parser.add_argument('--epochs', type=int, default=2,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='learning rate')
    parser.add_argument('--dataset', type=str, default="arron666/seq_monkey",
                        help='dataset name on Hugging Face')
    parser.add_argument('--data-path', type=str, default=None,
                        help='path to local dataset directory')
    parser.add_argument('--run-name', type=str, default="gpt_training",
                        help='name for this training run')
    parser.add_argument('--checkpoint-dir', type=str, default="checkpoints",
                        help='directory to save model checkpoints')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 创建checkpoint目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Initialize model
    config = GPTConfig()
    config.batch_size = args.batch_size
    model = GPT(config)
    
    # Load data
    if args.data_path:
        full_dataset = MyDataset(args.data_path)
    else:
        dataset_dict = load_dataset(args.dataset)
        full_dataset = dataset_dict['train']
        
    # 计算分割大小
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # 分割数据集
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False
    )
    
    # Train model
    train_model(
        model, 
        train_loader, 
        val_loader, 
        num_epochs=args.epochs,
        run_name=args.run_name
    )

if __name__ == "__main__":
    main() 