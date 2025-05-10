#--------------------------------------------------------------------------------------------------------------------------- 
# Code: Extracción de embeddings en el dataset.
# Author: Eddel Elí Ojeda Avilés.
# Last update: 10/05/2025.
#--------------------------------------------------------------------------------------------------------------------------- 

# Importación de módulos necesarios.
import os
import torch
import pickle
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError
from facenet_pytorch import MTCNN, InceptionResnetV1
from concurrent.futures import ThreadPoolExecutor, as_completed

# Selección de dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Inicialización de los modelos
mtcnn = MTCNN(keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

#--------------------------------------------------------------------------------------------------------------------------- 

def get_embedding(image_path):
    """
    Carga una imagen desde el disco, detecta una cara en ella, y genera un vector de embedding facial.

    Args:
        image_path (str): Ruta al archivo de imagen.

    Returns:
        np.ndarray or None: Vector de embedding facial de 512 dimensiones si se detecta un rostro,
                            None si no se detecta o hay errores al procesar la imagen.
    """

    # Abre y convierte la imagen.
    try:
        img = Image.open(image_path).convert("RGB")
    except (UnidentifiedImageError, FileNotFoundError, OSError) as e:
        print(f"❌ Error al cargar {image_path}: {e}")
        return None

    # Redimensiona la imagen a 160x160 píxeles si es necesario.
    if img.size != (160, 160):
        img = img.resize((160, 160), Image.Resampling.LANCZOS)

    # Detecta la cara usando MTCNN, regresando un tensor recortado de la cara o None si no se detectó ninguna cara.
    face = mtcnn(img)
    if face is None:
        print(f"⚠️ No se detectó rostro en: {image_path}")
        return None

    # Pasa la cara al modelo de embeddings y mueve el resultado a CPU
    with torch.no_grad():
        emb = resnet(face.unsqueeze(0).to(device)).squeeze().cpu().numpy()

    # Devuelve el vector de embedding como un array de NumPy
    return emb

#--------------------------------------------------------------------------------------------------------------------------- 

def extract_embeddings(dataset_dir="Data", output_file="dataset_embeddings.pkl"):
    """
    Genera vectores de características (embeddings faciales) a partir de un conjunto de imágenes 
    organizadas en carpetas por persona.

    Estructura esperada del dataset:
        Data/
        ├── Persona1/
        │   ├── img1.jpg
        │   ├── img2.jpg
        ├── Persona2/
        │   ├── img1.jpg
        │   ├── img2.jpg

    Args:
        dataset_dir (str): Ruta al directorio del dataset con subcarpetas por persona. Valor por default: 'Data'.
        output_file (str): Nombre del archivo `.pkl` donde se guardarán los embeddings. Valor por default: 'dataset_embeddings.pkl'

    Returns:
        None. Los resultados se guardan en el archivo `output_file` en formato pickle.
    """
    
    # Lista de subdirectorios (personas) dentro del dataset
    personas = [p for p in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, p))]

    # Diccionario donde se almacenarán los embeddings por persona
    embeddings_dir = {}

    
    with tqdm(total=len(personas), desc="Procesando personas", position=0, leave=True) as pbar:
        # Itera sobre cada persona en el dataset
        for persona in personas:
            persona_path = os.path.join(dataset_dir, persona)
            embeddings_dir[persona] = []

            # Lista de las imágenes en la carpeta de la persona
            imagenes = [
                os.path.join(persona_path, f)
                for f in os.listdir(persona_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png")) and not f.startswith(".")
            ]

            # Procesa las imágenes de cada persona en paralelo usando un ThreadPool
            with ThreadPoolExecutor() as executor:
                future_to_path = {executor.submit(get_embedding, img): img for img in imagenes}
                with tqdm(total=len(future_to_path), desc=f"Procesando imágenes de {persona}", position=1, leave=True) as img_pbar:
                    for future in as_completed(future_to_path):
                        emb = future.result()
                        if emb is not None:
                            embeddings_dir[persona].append(emb.tolist())
                        img_pbar.update(1)

            # Actualiza la barra de progreso de personas al finalizar el procesamiento de cada persona
            pbar.update(1)

    # Guardado de los embeddings output_file
    with open(output_file, "wb") as f:
        pickle.dump(embeddings_dir, f, protocol=4)
    print(f"\n✅ Embeddings generados y guardados en '{output_file}'")

#--------------------------------------------------------------------------------------------------------------------------- 

if __name__ == "__main__":
    extract_embeddings()