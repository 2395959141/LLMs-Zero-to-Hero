import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import swanlab

def train(model, optimizer, scheduler, train_loader, val_loader, device, epoch):
    model.train()
    pbar = tqdm(train_loader, desc=f'Training Epoch {epoch}')
    total_loss = 0
    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits, loss = model(x, targets=y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        # 记录每个batch的loss
        swanlab.log({
            "batch_loss": loss.item(),
            "batch": batch_idx + epoch * len(train_loader)
        })
        
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss

def eval(model, val_loader, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, targets=y)
            val_loss += loss.item()
    return val_loss

def train_model(model, train_loader, val_loader, num_epochs=2, run_name="gpt_training"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    
    # 初始化swanlab
    swanlab.init(
        project="LLM-zero-to-hero_gpt2",
        workspace="2395959141",
        name=run_name,
        config={
            "learning_rate": 3e-4,
            "architecture": "GPT",
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "epochs": num_epochs,
            "device": device
        }
    )
    
    try:
        for epoch in range(num_epochs):
            train_loss = train(model, optimizer, scheduler, train_loader, val_loader, device, epoch)
            val_loss = eval(model, val_loader, device)
            
            avg_train_loss = train_loss/len(train_loader)
            avg_val_loss = val_loss/len(val_loader)
            
            # 记录每个epoch的指标
            swanlab.log({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
            
            print(f'Epoch: {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
            }
            torch.save(checkpoint, f'checkpoints/model_epoch_{epoch}.pt')
    
    finally:
        # 确保运行结束时关闭swanlab
        swanlab.finish() 