import argparse
import torch
from torch.utils.data import DataLoader
from config import GPTConfig
from model import GPT
from dataset import MyDataset , load_dataset
from trainer import train_model

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
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize model
    config = GPTConfig()
    config.batch_size = args.batch_size
    model = GPT(config)
    
    # Load data
    if args.data_path:
        dataset = load_dataset(args.data_path)
    else:
        dataset = load_dataset(args.dataset)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset['train'], 
        [0.9, 0.1]
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
        num_epochs=args.epochs
    )

if __name__ == "__main__":
    main() 