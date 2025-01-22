import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import swanlab
import os

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
    
    # 从模型配置中获取训练参数
    training_config = model.config.training
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    
    # 创建checkpoints目录
    os.makedirs('checkpoints', exist_ok=True)
    
    # 初始化swanlab
    swanlab.init(
        project="LLM-zero-to-hero_gpt2",
        workspace="2395959141",
        name=run_name,
        config={
            "learning_rate": training_config.learning_rate,
            "architecture": "GPT",
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "max_steps": training_config.max_steps,
            "epochs": num_epochs,
            "device": device
        }
    )
    
    try:
        total_steps = 0
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            batch_count = 0
            
            pbar = tqdm(train_loader, desc=f'Training Epoch {epoch}', 
                       total=min(len(train_loader), training_config.max_steps-total_steps))
            
            for batch_idx, (x, y) in enumerate(pbar):
                if total_steps >= training_config.max_steps:
                    print(f"Reached maximum steps ({training_config.max_steps}). Stopping training.")
                    # 在达到最大步数时保存最后的检查点
                    checkpoint = {
                        'epoch': epoch,
                        'total_steps': total_steps,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_loss': best_val_loss,
                    }
                    torch.save(checkpoint, f'checkpoints/model_final_step_{total_steps}.pt')
                    break
                    
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits, loss = model(x, targets=y)
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                total_steps += 1
                epoch_loss += loss.item()
                batch_count += 1
                
                # 记录每个batch的loss
                swanlab.log({
                    "batch_loss": loss.item(),
                    "batch": total_steps,
                    "total_steps": total_steps
                })
                
                pbar.set_postfix({'loss': loss.item(), 'steps': total_steps})
                
                # 每save_steps步保存一次检查点
                if total_steps % training_config.save_steps == 0:
                    checkpoint = {
                        'epoch': epoch,
                        'total_steps': total_steps,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    }
                    torch.save(checkpoint, f'checkpoints/model_step_{total_steps}.pt')
            
            if total_steps >= training_config.max_steps:
                break
                
            # 计算验证集loss
            val_loss = eval(model, val_loader, device)
            avg_train_loss = epoch_loss / batch_count
            avg_val_loss = val_loss/len(val_loader)
            
            # 记录每个epoch的指标
            swanlab.log({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "total_steps": total_steps
            })
            
            print(f'Epoch: {epoch}, Steps: {total_steps}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

            # 保存每个epoch的检查点
            checkpoint = {
                'epoch': epoch,
                'total_steps': total_steps,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
            }
            torch.save(checkpoint, f'checkpoints/model_epoch_{epoch}.pt')
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(checkpoint, 'checkpoints/model_best.pt')
    
    finally:
        swanlab.finish() 