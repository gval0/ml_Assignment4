import torch
import torch.optim as optim
import wandb
from tqdm import tqdm
import torch.nn as nn

class FERTrainer:   
    def __init__(self, model, device, config):
        self.model = model.to(device)
        self.device = device
        self.config = config
        if config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'],
                                      weight_decay=config['weight_decay'])
        elif config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'],
                                     momentum=config['momentum'], weight_decay=config['weight_decay'])
        if config['scheduler'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 
                                                     step_size=config['step_size'], 
                                                     gamma=config['gamma'])
        elif config['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 
                                                                T_max=config['epochs'])
        self.criterion = nn.CrossEntropyLoss()
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, target)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            if batch_idx % 100 == 0:
                wandb.log({
                    'batch_loss': loss.item(),
                    'batch_accuracy': 100. * correct / total,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, target)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc, all_preds, all_targets
    
    def train(self, train_loader, val_loader, epochs):
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc, val_preds, val_targets = self.validate(val_loader)
            if hasattr(self, 'scheduler'):
                self.scheduler.step()
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f"New best model saved with validation accuracy: {best_val_acc:.2f}%")
            else:
                patience_counter += 1
            if patience_counter >= self.config['patience']:
                print(f"Early stopping triggered after {patience_counter} epochs without improvement")
                break
        
        return best_val_acc