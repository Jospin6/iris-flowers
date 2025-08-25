import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data.iris_loader import IrisDataLoader

class TestDataLoader(unittest.TestCase):
    
    def setUp(self):
        self.data_loader = IrisDataLoader(test_size=0.2, random_state=42)
    
    def test_load_data(self):
        X_train, X_test, y_train, y_test = self.data_loader.load_data()
        
        # Vérifie les shapes
        self.assertEqual(X_train.shape[1], 4)  # 4 features
        self.assertEqual(X_test.shape[1], 4)
        
        # Vérifie que les données sont normalisées
        self.assertAlmostEqual(np.mean(X_train), 0, places=1)
        self.assertAlmostEqual(np.std(X_train), 1, places=1)
    
    def test_create_dataloaders(self):
        train_loader, test_loader = self.data_loader.create_dataloaders(batch_size=16)
        
        # Vérifie les batch sizes
        for batch in train_loader:
            self.assertEqual(len(batch[0]), 16)
            break

if __name__ == '__main__':
    unittest.main()