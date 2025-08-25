# Iris Flowers Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.24+-ff4b4b.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow.svg)](https://huggingface.co/)
[![Render](https://img.shields.io/badge/Deployed-Render-5fddc6.svg)](https://render.com/)

A complete machine learning system for classifying iris flowers using a Multi-Layer Perceptron (MLP) neural network implemented with PyTorch. The project includes a trained model, a FastAPI backend, and a Streamlit frontend—all deployed and accessible online.

![screenshot](./screenshot1.png)
![screenshot](./screenshot2.png)

## ✨ Features

- **Machine Learning Model**: MLP classifier trained on the classic Iris dataset
- **RESTful API**: FastAPI backend with comprehensive endpoints
- **Web Interface**: Interactive Streamlit UI for real-time predictions
- **Cloud Deployment**: Fully deployed on modern platforms:
  - Model hosted on Hugging Face
  - API deployed on Render
  - UI deployed on Streamlit Cloud

## 🚀 Live Demos

- **Web Interface**: [Streamlit App](https://iris-flowers-project.streamlit.app/)
- **Model Repository**: [Hugging Face Model](https://huggingface.co/jospin6/iris-classification)

## 🛠️ Tech Stack

- **Machine Learning**: PyTorch, Scikit-learn, Pandas, NumPy
- **Backend**: FastAPI, Uvicorn, Pydantic
- **Frontend**: Streamlit
- **Deployment**: Hugging Face Hub, Render, Streamlit Cloud
- **Environment Management**: Pipenv

## 📦 Getting Started

### Prerequisites

- Python 3.8 or higher
- Pipenv (recommended) or pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/iris-flowers.git
   cd iris-flowers

2. **Install dependencies with Pipenv**
    ```bash
    pip install -r requirements.txt

### Running Locally

1. **Start the API server**
    ```bash
    cd api
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

2. **Start the Streamlit app**
    ```bash
    cd frontend
    streamlit run app.py