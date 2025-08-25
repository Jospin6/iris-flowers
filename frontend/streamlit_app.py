import streamlit as st
import torch
import numpy as np
import requests
import json
import sys
import os
import matplotlib.pyplot as plt

# Configuration de la page
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="wide"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF4B4B;
        text-align: center;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .prediction-text {
        color: #000000 !important; /* Noir */
        font-weight: bold;
        font-size: 1.5rem;
        margin: 0;
    }
    .confidence-high {
        color: green;
        font-weight: bold;
    }
    .confidence-medium {
        color: orange;
        font-weight: bold;
    }
    .confidence-low {
        color: red;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def predict_with_api(features):
    """Utilise l'API FastAPI pour la prédiction"""
    try:
        url = "http://localhost:8000/predict"
        data = {
            "sepal_length": float(features[0]),
            "sepal_width": float(features[1]),
            "petal_length": float(features[2]),
            "petal_width": float(features[3])
        }
        
        response = requests.post(url, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur API: {response.text}")
            return None
    except Exception as e:
        st.error(f"Erreur de connexion à l'API: {e}")
        return None

def predict_locally(features):
    """Prédiction locale sans API"""
    try:
        # Ajouter le chemin src
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(current_dir)
        
        from src.models.ann import IrisANN
        from src.training.trainer import IrisTrainer
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        class_names = ['setosa', 'versicolor', 'virginica']
        
        # Charger le modèle
        model_path = os.path.join(current_dir, 'models', 'iris_model.pth')
        if not os.path.exists(model_path):
            st.error("Modèle non trouvé. Veuillez d'abord entraîner le modèle.")
            return None
        
        trainer, model = IrisTrainer.load_model(model_path, device)
        model.eval()
        
        # Préparer les données d'entrée
        input_data = np.array(features).astype(np.float32)
        input_tensor = torch.FloatTensor(input_data).unsqueeze(0).to(device)  # Ajouter dimension batch
        
        # Prédiction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            return {
                'prediction': class_names[predicted.item()],
                'confidence': confidence.item(),
                'probabilities': probabilities.cpu().numpy()[0].tolist()
            }
            
    except Exception as e:
        st.error(f"Erreur lors de la prédiction locale: {e}")
        return None

def display_results(result, features):
    """Affiche les résultats de la prédiction"""
    st.markdown("---")
    st.header("Résultats de la prédiction")
    
    # Box de prédiction
    col1, col2 = st.columns(2)
    
    with col1:
        # Déterminer la couleur de confiance
        confidence_color = "confidence-high"
        if result['confidence'] < 0.7:
            confidence_color = "confidence-medium"
        if result['confidence'] < 0.5:
            confidence_color = "confidence-low"
        
        st.markdown(f"""
        <div class="prediction-box">
            <h3 class="prediction-text">🌺 Prédiction: {result['prediction'].upper()}</h3>
            <p class="prediction-text">Confiance: <span class="{confidence_color}">{result['confidence']*100:.2f}%</span></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Graphique des probabilités
        fig, ax = plt.subplots(figsize=(8, 4))
        classes = ['setosa', 'versicolor', 'virginica']
        probabilities = result['probabilities']
        
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        bars = ax.bar(classes, probabilities, color=colors)
        ax.set_ylabel('Probabilité')
        ax.set_title('Probabilités par classe')
        ax.set_ylim(0, 1)
        
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
        
        st.pyplot(fig)
    
    with col2:
        st.subheader("Caractéristiques saisies:")
        st.info(f"**Longueur du sépale:** {features[0]} cm")
        st.info(f"**Largeur du sépale:** {features[1]} cm")
        st.info(f"**Longueur du pétale:** {features[2]} cm")
        st.info(f"**Largeur du pétale:** {features[3]} cm")
        
        st.subheader("Probabilités détaillées:")
        for i, (cls, prob) in enumerate(zip(['setosa', 'versicolor', 'virginica'], result['probabilities'])):
            progress_value = min(prob, 1.0)  # S'assurer que c'est entre 0 et 1
            st.write(f"**{cls}:** {prob*100:.2f}%")
            st.progress(float(progress_value))

def main():
    st.title("🌸 Iris Flower Classification App")
    st.markdown("---")
    
    # Sidebar pour la configuration
    st.sidebar.header("Configuration")
    use_api = st.sidebar.checkbox("Utiliser l'API FastAPI", value=False)
    
    if use_api:
        st.sidebar.info("🌐 Mode API activé")
    else:
        st.sidebar.info("💻 Mode local activé")
    
    # Section de saisie des caractéristiques
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Caractéristiques de la fleur")
        sepal_length = st.slider("Longueur du sépale (cm)", 4.0, 8.0, 5.8, 0.1)
        sepal_width = st.slider("Largeur du sépale (cm)", 2.0, 4.5, 3.0, 0.1)
        
    with col2:
        st.header("")
        petal_length = st.slider("Longueur du pétale (cm)", 1.0, 7.0, 4.3, 0.1)
        petal_width = st.slider("Largeur du pétale (cm)", 0.1, 2.5, 1.3, 0.1)
    
    features = [sepal_length, sepal_width, petal_length, petal_width]
    
    # Bouton de prédiction
    if st.button("🔮 Prédire l'espèce", use_container_width=True, type="primary"):
        with st.spinner("Analyse en cours..."):
            if use_api:
                result = predict_with_api(features)
            else:
                result = predict_locally(features)
        
        if result:
            display_results(result, features)
    
    # Section d'exemples
    st.markdown("---")
    st.header("Exemples rapides")
    
    examples = {
        "Setosa": [5.1, 3.5, 1.4, 0.2],
        "Versicolor": [6.0, 2.7, 5.1, 1.6],
        "Virginica": [6.3, 3.3, 6.0, 2.5]
    }
    
    cols = st.columns(3)
    for i, (species, example_features) in enumerate(examples.items()):
        with cols[i]:
            if st.button(f"Charger {species}", use_container_width=True):
                # Utiliser session_state pour stocker les valeurs
                st.session_state.sepal_length = example_features[0]
                st.session_state.sepal_width = example_features[1]
                st.session_state.petal_length = example_features[2]
                st.session_state.petal_width = example_features[3]
                st.rerun()

# Initialiser les valeurs par défaut dans session_state
if 'sepal_length' not in st.session_state:
    st.session_state.sepal_length = 5.8
if 'sepal_width' not in st.session_state:
    st.session_state.sepal_width = 3.0
if 'petal_length' not in st.session_state:
    st.session_state.petal_length = 4.3
if 'petal_width' not in st.session_state:
    st.session_state.petal_width = 1.3

if __name__ == "__main__":
    main()