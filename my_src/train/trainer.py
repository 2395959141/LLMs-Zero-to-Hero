import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import swanlab
import os
from torch.amp import autocast, GradScaler
import glob

def eval(model, val_loader, device):
    model.eval()
    val_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    # 添加验证进度条，并显示总批次数
    total_batches = len(val_loader)
    pbar = tqdm(total=total_batches, desc=f'Validating ({len(val_loader.dataset)} samples)')
    
    with torch.no_grad(), autocast(device_type='cuda'):
        for batch_idx, (x, y) in enumerate(val_loader):
            batch_size = x.size(0)  # 获取当前批次的实际大小
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, targets=y)
            val_loss += loss.item()
            
            # 计算准确率
            predictions = logits.argmax(dim=-1)
            predictions = predictions.view(-1)
            y = y.view(-1)
            
            assert predictions.shape == y.shape, f"预测形状 {predictions.shape} 与目标形状 {y.shape} 不匹配"
            correct = (predictions == y).sum().item()
            total_correct += correct
            total_samples += y.numel()
            
            # 更新进度条，显示更多信息
            current_accuracy = correct / y.numel()
            pbar.set_postfix(ordered_dict={
                'batch': f'{batch_idx+1}/{total_batches}',
                'batch_size': batch_size,
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_accuracy:.2%}',
                'total_samples': total_samples
            })
            pbar.update(1)
    
    pbar.close()
    print(f"\n验证完成: 总样本数 {total_samples} | "
          f"平均损失 {val_loss/len(val_loader):.4f} | "
          f"总体准确率 {total_correct/total_samples:.2%}")
    
    accuracy = total_correct / total_samples
    return val_loss / len(val_loader), accuracy

def safe_log(metrics):
    """安全的记录指标"""
    try:
        swanlab.log(metrics)
    except Exception as e:
        print(f"警告：记录指标失败 - {str(e)}")

def save_checkpoint(model, optimizer, step, loss, checkpoint_dir, run_name, max_checkpoints=10):
    """保存模型检查点,并维护最多max_checkpoints个文件"""
    # 创建检查点目录
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 保存新的检查点
    checkpoint_path = os.path.join(checkpoint_dir, f"{run_name}_step_{step}.pt")
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"模型检查点已保存到: {checkpoint_path}")
    
    # 获取所有检查点文件并按修改时间排序
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, f"{run_name}_step_*.pt"))
    checkpoint_files.sort(key=os.path.getmtime)
    
    # 如果检查点数量超过限制,删除最旧的
    while len(checkpoint_files) > max_checkpoints:
        oldest_checkpoint = checkpoint_files.pop(0)  # 移除并返回最旧的检查点
        try:
            os.remove(oldest_checkpoint)
            print(f"删除旧检查点: {oldest_checkpoint}")
        except Exception as e:
            print(f"删除检查点失败: {e}")

def train_model(
    model,
    train_dataset,
    val_dataset,
    train_loader,
    val_loader,
    test_loader,
    config,
    num_epochs,
    run_name=None,
    use_swanlab=True
):
    """训练模型的主函数"""
    
    if use_swanlab:
        if run_name is None:
            run_name = "default_run"
        swanlab.init(
            experiment_name=run_name,
            config=vars(config)
        )
    
    # 修改所有的 swanlab.log 调用
    def log_metrics(metrics):
        if use_swanlab:
            safe_log(metrics)
    
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
    
    def evaluate_test_samples(model, test_loader, device):
        model.eval()
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad(), autocast(device_type='cuda'):
            # 只取第一个批次（8个样本）
            for batch_idx, (x, y) in enumerate(test_loader):
                if batch_idx >= 1:  # 只评估第一个批次
                    break
                
                x, y = x.to(device), y.to(device)
                logits, loss = model(x, targets=y)
                
                # 计算准确率
                predictions = logits.argmax(dim=-1)
                predictions = predictions.view(-1)
                y = y.view(-1)
                
                total_correct += (predictions == y).sum().item()
                total_samples += y.numel()
        
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        # 添加详细的打印信息
        print(f"\n评估测试样本: 正确数 {total_correct}/{total_samples} | 准确率 {accuracy:.2%}")
        return accuracy
    
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
            
            # 为梯度累积调整loss
            loss = loss / config.gradient_accumulation_steps
            scaler.scale(loss).backward()
            
            # 直接累积原始loss，不需要额外的乘除操作
            accumulated_loss += loss.item()
            
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
                
                # 直接使用accumulated_loss作为平均损失
                avg_loss = accumulated_loss
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
                log_metrics({
                    "train/avg_loss": avg_loss,
                    "train/learning_rate": current_lr,
                    "train/epoch": epoch + (batch_idx + 1) / len(train_loader),
                    "train/global_step": global_step
                })
                
                # 每5000步进行一次验证和保存检查点
                if global_step % config.eval_steps == 0:
                    model.eval()
                    val_loss, accuracy = eval(model, val_loader, device)
                    print(f'\nStep {global_step}: Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')
                    
                    # 记录验证结果
                    log_metrics({
                        "val/loss": val_loss,
                        "val/accuracy": accuracy,
                        "val/step": global_step
                    })
                    
                    # 保存检查点
                    if run_name:
                        save_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            step=global_step,
                            loss=val_loss,
                            checkpoint_dir=config.checkpoint_dir,
                            run_name=run_name,
                            max_checkpoints=config.max_checkpoints
                        )
                    
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
        log_metrics({
            "val/loss": val_loss,
            "val/accuracy": accuracy,
            "val/best_loss": best_val_loss,
            "val/epoch": epoch + 1
        })
        
        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {total_loss/len(train_loader):.4f} | '
              f'Val Loss: {val_loss:.4f} | '
              f'Accuracy: {accuracy:.4f}')

        # 在每个epoch结束后评估测试样本
        if epoch % config.eval_interval == 0:
            test_accuracy = evaluate_test_samples(model, test_loader, device)
            # 添加日志记录
            log_metrics({
                "test/sample_accuracy": test_accuracy,
                "test/step": global_step
            })
            print(f"Epoch {epoch+1}: 测试样本评估完成，当前准确率 {test_accuracy:.2%}\n")

    # 记录到swanlab
    log_metrics({
        "train/total_loss": total_loss,
        "train/best_loss": best_val_loss,
        "train/epochs": num_epochs
    })

    return total_loss 