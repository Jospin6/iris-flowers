import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_training_history(trainer, figsize=(12, 5)):
    """Affiche l'historique d'entraînement"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot losses
    ax1.plot(trainer.train_losses, label='Train Loss')
    ax1.plot(trainer.val_losses, label='Validation Loss')
    ax1.set_title('Loss during training')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(trainer.train_accuracies, label='Train Accuracy')
    ax2.plot(trainer.val_accuracies, label='Validation Accuracy')
    ax2.set_title('Accuracy during training')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(true_labels, predictions, class_names, figsize=(8, 6)):
    """Affiche la matrice de confusion"""
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def save_model(model, path):
    """Sauvegarde le modèle"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': model.__class__.__name__
    }, path)

def load_model(model_class, path, **kwargs):
    """Charge un modèle sauvegardé"""
    checkpoint = torch.load(path)
    model = model_class(**kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model