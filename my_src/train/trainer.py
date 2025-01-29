import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import swanlab
import os

def train(model, optimizer, scheduler, train_loader, val_loader, device, epoch, config, global_step=0):
    model.train()
    pbar = tqdm(train_loader, total=len(train_loader), desc=f'训练轮次 {epoch+1}')
    total_loss = 0
    best_val_loss = float('inf')
    
    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits, loss = model(x, targets=y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        global_step += 1
        
        # 记录每个batch的loss
        swanlab.log({
            "batch_loss": loss.item(),
            "batch": batch_idx + epoch * len(train_loader),
            "global_step": global_step
        })
        
        # 定期验证
        if global_step % config.eval_steps == 0:
            model.eval()
            val_loss = eval(model, val_loader, device)
            model.train()  # 切回训练模式
            
            # 记录验证结果
            swanlab.log({
                "step_val_loss": val_loss,
                "eval_step": global_step
            })
            
            # 更新进度条信息
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}',
                'step': global_step,
                'val_loss': f'{val_loss:.4f}'
            })
            
            # 检查是否需要保存最佳模型
            if val_loss < best_val_loss - config.min_delta:
                best_val_loss = val_loss
                checkpoint = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': loss.item(),
                    'val_loss': val_loss,
                }
                torch.save(checkpoint, f'checkpoints/best_model_step_{global_step}.pt')
                print(f'\n步数 {global_step}: 保存最佳模型，验证损失: {val_loss:.4f}')
        
        # 每 save_steps 步保存一次checkpoint
        if global_step % config.save_steps == 0:
            checkpoint = {
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss.item(),
            }
            torch.save(checkpoint, f'checkpoints/model_step_{global_step}.pt')
    
    return total_loss / len(train_loader), global_step, best_val_loss

def eval(model, val_loader, device):
    model.eval()
    val_loss = 0
    pbar = tqdm(val_loader, total=len(val_loader), desc='验证中')
    with torch.no_grad():
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, targets=y)
            val_loss += loss.item()
            
            pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
            
    return val_loss / len(val_loader)

def train_model(model, train_loader, val_loader, config, num_epochs=2, run_name="gpt_training"):
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
            "device": device,
            "save_steps": config.save_steps,
            "max_steps": config.max_steps,
            "patience": config.patience,
            "min_delta": config.min_delta
        }
    )
    
    # 添加总体训练进度条
    epoch_pbar = tqdm(range(num_epochs), desc='总体训练进度')
    global_step = 0  # 添加全局步数计数器
    
    # 早停相关变量
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = 'checkpoints/best_model.pt'
    
    try:
        for epoch in epoch_pbar:
            # 检查是否达到最大步数
            if global_step >= config.max_steps:
                print(f'达到最大步数 {config.max_steps}，停止训练')
                break
                
            # 训练阶段
            train_loss, global_step, step_best_val_loss = train(
                model, 
                optimizer, 
                scheduler, 
                train_loader,
                val_loader,  # 添加验证集
                device, 
                epoch,
                config,
                global_step
            )
            
            # 如果配置了epoch结束时验证
            if config.eval_epoch:
                val_loss = eval(model, val_loader, device)
                
                # 早停检查
                if val_loss < best_val_loss - config.min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # 保存最佳模型
                    checkpoint = {
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'best_val_loss': best_val_loss
                    }
                    torch.save(checkpoint, 'checkpoints/best_model.pt')
                    print(f'Epoch {epoch}: 保存最佳模型，验证损失: {val_loss:.4f}')
                else:
                    patience_counter += 1
                    if patience_counter >= config.patience:
                        print(f'早停：验证损失在 {config.patience} 个epoch内没有显著改善')
                        break
            
            # 更新总体进度条信息
            epoch_pbar.set_postfix({
                '训练损失': f'{train_loss:.4f}',
                '验证损失': f'{val_loss:.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}',
                'Step': global_step,
                'Best Val Loss': f'{best_val_loss:.4f}',
                'Patience': patience_counter
            })
            
            # 记录每个epoch的指标
            swanlab.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "global_step": global_step,
                "patience_counter": patience_counter
            })

            # 每个epoch结束后保存checkpoint
            checkpoint = {
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter
            }
            torch.save(checkpoint, f'checkpoints/model_epoch_{epoch}.pt')
    
    finally:
        # 确保运行结束时关闭swanlab
        swanlab.finish()
        
        # 如果训练被中断，确保加载最佳模型
        if os.path.exists(best_model_path):
            print(f'加载最佳模型，最佳验证损失: {best_val_loss:.4f}')
            best_checkpoint = torch.load(best_model_path)
            model.load_state_dict(best_checkpoint['model_state_dict'])
            
        return model, best_val_loss 