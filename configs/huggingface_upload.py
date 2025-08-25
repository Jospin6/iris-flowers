from huggingface_hub import HfApi, upload_file
import os
import torch

def upload_to_huggingface():
    """Upload model to Hugging Face Hub"""
    api = HfApi()
    
    # Create repository
    repo_id = "jospin6/iris-classification"
    
    # Upload model
    model_path = "models/iris_model.pth"
    if os.path.exists(model_path):
        upload_file(
            path_or_fileobj=model_path,
            path_in_repo="iris_model.pth",
            repo_id=repo_id,
            repo_type="model"
        )
        print("✅ Modèle uploadé sur Hugging Face!")
    
    # Upload API code
    upload_file(
        path_or_fileobj="api/app.py",
        path_in_repo="app.py",
        repo_id=repo_id,
        repo_type="model"
    )

if __name__ == "__main__":
    upload_to_huggingface()