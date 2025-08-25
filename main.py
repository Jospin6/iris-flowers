import torch
import torch.nn as nn
from src.data.iris_loader import IrisDataLoader
from src.models.ann import IrisANN
from src.training.trainer import IrisTrainer
from src.utils.helpers import plot_training_history, plot_confusion_matrix

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Chargement des données
    data_loader = IrisDataLoader(test_size=0.2, random_state=42)
    train_loader, test_loader = data_loader.create_dataloaders(batch_size=16)
    class_names = data_loader.get_class_names()
    
    # Création du modèle
    model = IrisANN(input_size=4, hidden_size1=16, hidden_size2=8, output_size=3)
    print(f"Model architecture:\n{model}")
    
    # Initialisation du trainer
    trainer = IrisTrainer(model, device=device)
    
    # Définition de la loss et de l'optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Entraînement
    trainer.train(
        train_loader, test_loader, criterion, 
        optimizer=torch.optim.Adam,
        num_epochs=100,
        patience=15,
        learning_rate=0.001
    )
    
    # Visualisation des résultats
    plot_training_history(trainer)
    
    # Évaluation finale
    test_loss, test_acc = trainer.evaluate(test_loader, criterion)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    print(f"Final Test Loss: {test_loss:.4f}")
    
    # Rapport de classification
    report = trainer.get_classification_report(test_loader, class_names)
    print("\nClassification Report:")
    print(report)

    trainer.save_model('models/iris_model.pth')

    # Chargement test (optionnel)
    print("\nTest de chargement du modèle...")
    loaded_trainer, loaded_model = IrisTrainer.load_model('models/iris_model.pth')
    test_loss, test_acc = loaded_trainer.evaluate(test_loader, criterion)
    print(f"Modèle chargé - Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()