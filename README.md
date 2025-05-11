# Reconocimiento Facial en Tiempo Real 🧠📸

Este repositorio contiene una aplicación web para el reconocimiento facial en tiempo real utilizando deep learning. La app permite identificar personas a partir de una imagen capturada con la cámara, comparando sus características faciales contra un conjunto de embeddings previamente registrados.

## 🚀 Características

- Reconocimiento facial usando **InceptionResnetV1** y **MTCNN** (`facenet-pytorch`).
- Aplicación interactiva desarrollada con **Streamlit**.
- Identificación de personas basada en **similitud de embeddings** faciales.
- Funciona en **tiempo real** desde la cámara web del navegador.

## 🧠 Modelo

- Detección de rostros: `MTCNN`.
- Extracción de embeddings: `InceptionResnetV1` preentrenado con **VGGFace2**.
- Cálculo de similitud: **cosine similarity**.
- Identificación basada en un archivo `dataset_embeddings.pkl` con personas previamente registradas.

## 📂 Estructura del Proyecto

```
📁 proyecto/
├── app.py                  # Código principal de la aplicación Streamlit
├── dataset_embeddings.pkl  # Base de datos de embeddings faciales
├── requirements.txt        # Dependencias del proyecto
└── README.md               # Este archivo
```

## ▶️ Uso

1. Clona el repositorio:
   ```bash
   git clone https://github.com/tuusuario/reconocimiento-facial-app.git
   cd reconocimiento-facial-app
   ```

2. Instala las dependencias:

   ```bash
   pip install -r requirements.txt
   ```
   
3. Ejecuta la app localmente:

   ```bash
   streamlit run app.py
   ```

## 🌐 Prueba la app
Puedes probar la aplicación en línea en el siguiente enlace:

[App - Reconocimiente Facial](https://reconocimiento-facial-eo.streamlit.app/)

## ⚠️ Requisitos

- El archivo dataset_embeddings.pkl debe estar disponible en el repositorio. Este archivo contiene los embeddings faciales de las personas a reconocer.
- Se recomienda utilizar un navegador con soporte para cámara web.

## 📚 Dataset

- Los embeddings deben ser generados previamente utilizando imágenes de referencia para cada persona.
- Puedes extender el sistema añadiendo nuevos usuarios mediante extracción manual con facenet-pytorch.

## ✏️ Autor
Desarrollado por [Eddel Ojeda](https://github.com/eddelojeda)