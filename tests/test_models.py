import unittest
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.ann import IrisANN, IrisModelFactory

class TestModels(unittest.TestCase):
    
    def test_iris_ann_forward(self):
        model = IrisANN(input_size=4, output_size=3)
        x = torch.randn(32, 4)  # batch de 32 échantillons
        output = model(x)
        
        self.assertEqual(output.shape, (32, 3))  # 32 prédictions, 3 classes
    
    def test_model_factory(self):
        simple_model = IrisModelFactory.create_simple_model()
        deep_model = IrisModelFactory.create_deep_model()
        
        x = torch.randn(1, 4)
        self.assertEqual(simple_model(x).shape, (1, 3))
        self.assertEqual(deep_model(x).shape, (1, 3))

if __name__ == '__main__':
    unittest.main()