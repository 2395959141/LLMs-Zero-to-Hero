import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import swanlab
import os
from torch.amp import autocast, GradScaler

def train(model, optimizer, scheduler, train_loader, val_loader, device, epoch, config, global_step=0):
    model.train()
    total_loss = 0.0
    
    # 计算实际的总步数
    total_steps = len(train_loader) // config.gradient_accumulation_steps
    
    # 更新tqdm描述
    pbar = tqdm(total=total_steps, desc=f'Epoch {epoch+1}/{config.epochs}')
    
    # 只在CUDA设备上使用GradScaler
    device_type = 'cuda' if device == 'cuda' else 'cpu'
    scaler = GradScaler() if device_type == 'cuda' else None
    
    # 用于梯度累积的变量
    accumulated_loss = 0.0
    accumulation_steps = config.gradient_accumulation_steps
    optimizer.zero_grad()
    
    for batch_idx, (x, y) in enumerate(train_loader):
        # 检查是否达到最大步数
        if global_step >= config.max_steps:
            pbar.close()
            print(f"达到最大步数 {config.max_steps}，停止训练")
            break
            
        # 将数据移到设备上
        x, y = x.to(device), y.to(device)
        
        # 使用混合精度训练
        with autocast(device_type='cuda'):
            logits, loss = model(x, targets=y)
            loss = loss / accumulation_steps
        scaler.scale(loss).backward()
        
        # 累积loss（使用未缩放的loss计算平均值）
        accumulated_loss += loss.item() * accumulation_steps
        
        # 在达到累积步数时更新参数
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            optimizer.zero_grad()
            
            # 更新学习率
            scheduler.step()
            
            # 计算平均loss
            avg_loss = accumulated_loss / accumulation_steps
            total_loss += avg_loss
            global_step += 1
            
            # 更新进度条信息
            current_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(ordered_dict={
                'avg_loss': f'{avg_loss:.4f}',
                'lr': f'{current_lr:.2e}',
                'step': f'{global_step}/{config.max_steps}',
                'acc_steps': accumulation_steps
            })
            pbar.update(1)
            
            # 记录训练指标
            swanlab.log({
                "train/avg_loss": avg_loss,
                "train/learning_rate": current_lr,
                "train/global_step": global_step,
                "train/epoch": epoch + (batch_idx + 1) / len(train_loader)
            })
            
            # 重置累积的loss
            accumulated_loss = 0.0
            
            # 检查是否需要进行验证
            if config.eval_steps > 0 and global_step % config.eval_steps == 0:
                val_loss, accuracy = eval(model, val_loader, device)
                print(f'\nStep {global_step}: Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')
                swanlab.log({
                    "val/loss": val_loss,
                    "val/accuracy": accuracy,
                    "val/step": global_step
                })
                model.train()  # 切回训练模式
    
    # 处理最后不完整的累积步数
    if accumulated_loss > 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
        avg_loss = accumulated_loss / (batch_idx % accumulation_steps + 1)
        total_loss += avg_loss
        global_step += 1
    
    pbar.close()
    return total_loss / (len(train_loader) // accumulation_steps), global_step

def eval(model, val_loader, device):
    model.eval()
    val_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad(), autocast(device_type='cuda'):
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, targets=y)
            val_loss += loss.item()
            
            # 计算准确率 - 修改这部分
            predictions = logits.argmax(dim=-1)  # 获取最大值的索引
            predictions = predictions.view(-1)    # 展平预测结果
            y = y.view(-1)                       # 展平目标标签
            
            # 确保维度匹配后再计算正确预测的数量
            assert predictions.shape == y.shape, f"预测形状 {predictions.shape} 与目标形状 {y.shape} 不匹配"
            total_correct += (predictions == y).sum().item()
            total_samples += y.numel()
    
    accuracy = total_correct / total_samples
    return val_loss / len(val_loader), accuracy 



def train_model(
    model,
    train_dataset,
    val_dataset,
    train_loader,
    val_loader,
    config,
    num_epochs,
    run_name=None
):
    """训练模型的主函数"""
    
    # 初始化swanlab
    if run_name:
        swanlab.init(
            experiment_name=run_name,
            config=vars(config)  # 将config对象转换为字典
        )
    
    # 设置设备
    device = torch.device("cuda")
    model = model.to(device)
    
    # 初始化GradScaler
    scaler = GradScaler()
    
    # 设置优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate, #3e-4
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=config.weight_decay, #0.01
        foreach=True  
    )
    
    # 添加学习率调度器
    total_steps = len(train_loader) * num_epochs // config.gradient_accumulation_steps
    warmup_steps = int(total_steps * 0.1)  # 预热步数为总步数的10%
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=total_steps,
        pct_start=warmup_steps/total_steps,
        anneal_strategy='cos',
        cycle_momentum=False
    )
    
    # 初始化最佳验证损失
    best_val_loss = float('inf')
    global_step = 0
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        accumulated_loss = 0.0 
        
        # 在epoch开始时清零梯度
        optimizer.zero_grad()
        # 添加进度条
        pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            # 使用混合精度训练
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / config.gradient_accumulation_steps # 为梯度累积调整loss
            scaler.scale(loss).backward()
            
            # 累积损失
            accumulated_loss += loss.item() * config.gradient_accumulation_steps
            
            # 在达到累积步数时更新参数
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                # 更新参数
                scaler.step(optimizer)
                scaler.update()
                
                # 更新学习率
                scheduler.step()
                
                # 清零梯度
                optimizer.zero_grad()
                
                # 计算平均损失
                avg_loss = accumulated_loss / config.gradient_accumulation_steps
                total_loss += avg_loss
                global_step += 1
                
                # 更新进度条信息
                current_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix(ordered_dict={
                    'avg_loss': f'{avg_loss:.4f}',
                    'lr': f'{current_lr:.2e}',
                    'step': f'{global_step}/{total_steps}'
                })
                pbar.update(config.gradient_accumulation_steps)
                
                # 记录到swanlab
                swanlab.log({
                    "train/avg_loss": avg_loss,
                    "train/learning_rate": current_lr,
                    "train/epoch": epoch + (batch_idx + 1) / len(train_loader),
                    "train/global_step": global_step
                })
                
                # 每5000步进行一次验证和保存检查点
                if global_step % 5000 == 0:
                    model.eval()
                    val_loss, accuracy = eval(model, val_loader, device)
                    print(f'\nStep {global_step}: Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')
                    
                    # 记录验证结果
                    swanlab.log({
                        "val/loss": val_loss,
                        "val/accuracy": accuracy,
                        "val/step": global_step
                    })
                    
                    # 保存检查点
                    if run_name:
                        checkpoint_path = os.path.join(
                            config.checkpoint_dir, 
                            f'{run_name}_step_{global_step}.pt'
                        )
                        torch.save({
                            'global_step': global_step,
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'loss': val_loss,
                            'accuracy': accuracy,
                        }, checkpoint_path)
                        print(f'保存检查点到 {checkpoint_path}')
                    
                    # 如果是最佳模型，额外保存一份
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        if run_name:
                            best_model_path = os.path.join(
                                config.checkpoint_dir, 
                                f'{run_name}_best.pt'
                            )
                            torch.save({
                                'global_step': global_step,
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                                'loss': best_val_loss,
                                'accuracy': accuracy,
                            }, best_model_path)
                            print(f'保存最佳模型到 {best_model_path}')
                    
                    # 恢复训练模式
                    model.train()
                
                # 重置累积的损失
                accumulated_loss = 0.0

        pbar.close()
        
        # 验证模型
        model.eval()
        val_loss, accuracy = eval(model, val_loader, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # 保存最佳模型
            if run_name:
                checkpoint_path = os.path.join(config.checkpoint_dir, f'{run_name}_best.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                }, checkpoint_path)
                print(f'保存最佳模型到 {checkpoint_path}')

        # 记录到swanlab
        swanlab.log({
            "val/loss": val_loss,
            "val/accuracy": accuracy,
            "val/best_loss": best_val_loss,
            "val/epoch": epoch + 1
        })
        
        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {total_loss/len(train_loader):.4f} | '
              f'Val Loss: {val_loss:.4f} | '
              f'Accuracy: {accuracy:.4f}')

    # 记录到swanlab
    swanlab.log({
        "train/total_loss": total_loss,
        "train/best_loss": best_val_loss,
        "train/epochs": num_epochs
    })

    return total_loss 