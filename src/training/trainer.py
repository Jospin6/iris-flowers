import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from src.models.ann import IrisANN

class IrisTrainer:
    """Classe pour entraîner et évaluer le modèle"""
    
    def __init__(self, model, device=None):
        self.model = model
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train(self, train_loader, val_loader, criterion, optimizer, 
              num_epochs=100, patience=10, learning_rate=0.01):
        """Entraîne le modèle avec early stopping"""
        
        optimizer = optimizer(self.model.parameters(), lr=learning_rate)
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            all_preds = []
            all_labels = []
            
            for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            train_loss = running_loss / len(train_loader)
            train_acc = accuracy_score(all_labels, all_preds)
            
            # Validation phase
            val_loss, val_acc = self.evaluate(val_loader, criterion)
            
            # Stockage des métriques
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Sauvegarde du meilleur modèle
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    
    def evaluate(self, data_loader, criterion):
        """Évalue le modèle sur un dataset"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = running_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def get_classification_report(self, data_loader, class_names):
        """Génère un rapport de classification détaillé"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        return classification_report(all_labels, all_preds, target_names=class_names)
    
    def predict_single(self, features):
        """Prédit une seule instance"""
        self.model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).to(self.device)
            output = self.model(features_tensor)
            probabilities = F.softmax(output, dim=0)
            _, predicted = torch.max(output, 0)
            return predicted.item(), probabilities.cpu().numpy()
        
    def save_model(self, path='models/iris_model.pth'):
        """Sauvegarde le modèle et les métadonnées"""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'input_size': 4,
            'output_size': 3
        }
        torch.save(checkpoint, path)
        print(f"Modèle sauvegardé à : {path}")

    @staticmethod
    def load_model(path='models/iris_model.pth', device=None):
        """Charge un modèle sauvegardé"""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(path, map_location=device)
        
        # Créer le modèle
        model = IrisANN(
            input_size=checkpoint.get('input_size', 4),
            output_size=checkpoint.get('output_size', 3)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        # Créer le trainer
        trainer = IrisTrainer(model, device)
        trainer.train_losses = checkpoint.get('train_losses', [])
        trainer.val_losses = checkpoint.get('val_losses', [])
        trainer.train_accuracies = checkpoint.get('train_accuracies', [])
        trainer.val_accuracies = checkpoint.get('val_accuracies', [])
        
        return trainer, model