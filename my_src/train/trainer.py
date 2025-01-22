import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def train(model, optimizer, scheduler, train_loader, val_loader, device):
    model.train()
    pbar = tqdm(train_loader, desc='Training')
    total_loss = 0
    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits, loss = model(x, targets=y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Batch: {batch_idx}, Loss: {loss.item():.4f}')
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

def train_model(model, train_loader, val_loader, num_epochs=2):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    
    for epoch in range(num_epochs):
        train_loss = train(model, optimizer, scheduler, train_loader, val_loader, device)
        val_loss = eval(model, val_loader, device)
        print(f'Epoch: {epoch}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss / len(val_loader),
        }
        torch.save(checkpoint, f'checkpoints/model_epoch_{epoch}.pt') 