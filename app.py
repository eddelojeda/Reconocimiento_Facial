#--------------------------------------------------------------------------------------------------------------------------- 
# Code: Extracción de embeddings en el dataset.
# Author: Eddel Elí Ojeda Avilés.
# Last update: 10/05/2025.
#--------------------------------------------------------------------------------------------------------------------------- 

# Importación de módulos necesarios.
import cv2
import torch
import pickle
import numpy as np
import streamlit as st
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# Selección de dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Inicialización de los modelos
mtcnn = MTCNN(keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Configuración de la página
st.set_page_config(page_title="Reconocimiento Facial en Tiempo Real", layout="wide")

# Título de la aplicación
st.title("Reconocimiento Facial en Tiempo Real")

#--------------------------------------------------------------------------------------------------------------------------- 

# Carga de dataset_embeddings.pkl
@st.cache_resource
def load_embeddings(path='dataset_embeddings.pkl'):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"No se encontró el archivo '{path}'. Asegúrate de subirlo al repositorio.")
        st.stop()

#--------------------------------------------------------------------------------------------------------------------------- 

# Función para encontrar la persona más similar en el dataset
def find_most_similar(embedding_input, embeddings_db, threshold=0.95):
    highest_similarity = float('-inf')
    identity = "Desconocido"

    for name, embeddings in embeddings_db.items():
        for emb in embeddings:
            emb_tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)
            similarity = torch.nn.functional.cosine_similarity(embedding_input, emb_tensor).item()
            if similarity > highest_similarity:
                highest_similarity = similarity
                if similarity > threshold:
                    identity = name

    return identity, highest_similarity

#--------------------------------------------------------------------------------------------------------------------------- 

# Carga de los embeddings del dataset
embeddings_db = load_embeddings()

# Captura de imagen desde la cámara
picture = st.camera_input("Captura una imagen")

if picture:
    # Convertir la imagen a formato PIL
    img = Image.open(picture).convert('RGB')

    # Detectar rostro
    face = mtcnn(img)
    if face is not None:
        with torch.no_grad():
            embedding = resnet(face.unsqueeze(0).to(device))

        # Buscar la identidad más cercana
        threshold = 0.8
        name, score = find_most_similar(embedding, embeddings_db, threshold)

        # Muestra los resultados
        if round(score,2) > threshold:
            st.success(f"Identidad: {name}")
            st.info(f"Similitud: {score*100:.2f}%")
        else:
            st.error(f"Identidad: Desconocida")
    else:
        st.warning("No se detectó ningún rostro en la imagen.")