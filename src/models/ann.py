import torch
import torch.nn as nn
import torch.nn.functional as F

class IrisANN(nn.Module):
    """Réseau de neurones pour la classification des fleurs d'Iris"""
    
    def __init__(self, input_size=4, hidden_size1=16, hidden_size2=8, output_size=3, dropout_rate=0.3):
        super(IrisANN, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size1)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size2)
        
    def forward(self, x):
        x = F.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
    def predict(self, x):
        """Fait une prédiction et retourne la classe"""
        with torch.no_grad():
            outputs = self.forward(x)
            _, predicted = torch.max(outputs, 1)
            return predicted
    
    def predict_proba(self, x):
        """Retourne les probabilités de prédiction"""
        with torch.no_grad():
            outputs = self.forward(x)
            probabilities = F.softmax(outputs, dim=1)
            return probabilities

class IrisModelFactory:
    """Factory pour créer différents modèles"""
    
    @staticmethod
    def create_simple_model(input_size=4, output_size=3):
        """Crée un modèle simple"""
        return nn.Sequential(
            nn.Linear(input_size, 8),
            nn.ReLU(),
            nn.Linear(8, output_size)
        )
    
    @staticmethod
    def create_deep_model(input_size=4, output_size=3):
        """Crée un modèle plus profond"""
        return nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, output_size)
        )