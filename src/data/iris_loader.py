import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader

class IrisDataset(Dataset):
    """Dataset personnalisé pour les fleurs d'Iris"""
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class IrisDataLoader:
    """Classe pour charger et préparer les données Iris"""
    
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def load_data(self):
        """Charge et prépare les données"""
        iris = load_iris()
        X = iris.data
        y = iris.target
        
        # Normalisation des features
        X_scaled = self.scaler.fit_transform(X)
        
        # Encodage des labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y_encoded, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y
        )
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def create_dataloaders(self, batch_size=32):
        """Crée les DataLoaders PyTorch"""
        if self.X_train is None:
            self.load_data()
        
        train_dataset = IrisDataset(self.X_train, self.y_train)
        test_dataset = IrisDataset(self.X_test, self.y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader
    
    def get_class_names(self):
        """Retourne les noms des classes"""
        iris = load_iris()
        return iris.target_names