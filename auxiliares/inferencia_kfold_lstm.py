import os
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from keras import layers, models, optimizers

# --- 1. CONFIGURAÇÕES (DEVEM SER IGUAIS AO TREINAMENTO) ---
CLASSES = [
    "*",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
]
NUM_CLASSES = len(CLASSES)

# Parâmetros usados no treinamento (verifique se batem com seu cfg)
IMG_HEIGHT = 32
IMG_WIDTH = 32
SEQUENCE_LENGTH = 32 
LSTM_UNITS = 2048 # Valor do seu código original

# --- 2. RECONSTRUÇÃO DA ARQUITETURA ---
def build_model_for_inference(sequence_length: int, img_height: int, img_width: int, lstm_units: int):
    """
    Reconstroi a arquitetura exata do modelo para receber os pesos.
    """
    model = models.Sequential()
    
    # Define o Input explicitamente para evitar avisos do Keras 3
    model.add(layers.Input(shape=(sequence_length, img_height, img_width, 1)))
    
    model.add(layers.TimeDistributed(
        layers.Flatten()
    ))

    model.add(layers.LSTM(lstm_units, return_sequences=True))
    model.add(layers.TimeDistributed(layers.Dense(NUM_CLASSES, activation="softmax")))
    
    # Não precisamos compilar para inferência, mas ajuda a evitar avisos se for usar evaluate
    return model

# --- 3. PRE-PROCESSAMENTO DE IMAGEM ---
def load_and_preprocess_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path, flags=0) # Ler em escala de cinza
    
    if img is None:
        # Tenta ler colorido e converter se falhar (algumas libs salvam jpg diferente)
        img = cv2.imread(image_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Não foi possível ler a imagem: {image_path}")

    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = np.expand_dims(img, axis=-1)
    img = img.astype(np.float32) / 255.0
    return img

def prepare_video_sequences(frame_folder: Path) -> Tuple[np.ndarray, List[str]]:
    extensions = {".jpg", ".jpeg", ".png"}
    frame_files = sorted([
        f for f in frame_folder.iterdir() 
        if f.suffix.lower() in extensions
    ], key=lambda x: int(''.join(filter(str.isdigit, x.name))) if any(char.isdigit() for char in x.name) else x.name)

    if not frame_files:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {frame_folder}")

    print(f"Encontrados {len(frame_files)} frames em '{frame_folder.name}'")

    loaded_frames = []
    valid_files = []
    
    for f_path in frame_files:
        try:
            img = load_and_preprocess_image(str(f_path))
            loaded_frames.append(img)
            valid_files.append(f_path.name)
        except Exception as e:
            print(f"Erro ao carregar {f_path}: {e}")

    # Padding
    num_frames = len(loaded_frames)
    remainder = num_frames % SEQUENCE_LENGTH
    
    if remainder != 0:
        padding_needed = SEQUENCE_LENGTH - remainder
        print(f"Padding: Adicionando {padding_needed} frames vazios.")
        empty_frame = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)
        for _ in range(padding_needed):
            loaded_frames.append(empty_frame)
            valid_files.append("PADDING")
            
    data_array = np.array(loaded_frames)
    num_sequences = len(data_array) // SEQUENCE_LENGTH
    
    batch_input = data_array.reshape((num_sequences, SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH, 1))
    
    return batch_input, valid_files

# --- 4. FUNÇÃO PRINCIPAL DE PREDIÇÃO ---
def predict_folder(model_path: str, video_folder_path: str, output_csv: str):
    if not os.path.exists(model_path):
        print(f"ERRO: Modelo não encontrado em {model_path}")
        return

    print("Construindo modelo e carregando pesos...")
    
    try:
        # 1. Instancia a arquitetura limpa
        model = build_model_for_inference(SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH, LSTM_UNITS)
        
        # 2. Carrega APENAS os pesos (evita o erro de configuração do LSTM)
        model.load_weights(model_path)
        print("Modelo carregado com sucesso!")
        
    except Exception as e:
        print(f"Erro fatal ao carregar pesos: {e}")
        print("Dica: Verifique se LSTM_UNITS no código é igual ao usado no treino (4096).")
        return

    video_folder = Path(video_folder_path)
    print("Processando imagens...")
    
    try:
        X_input, file_names = prepare_video_sequences(video_folder)
    except Exception as e:
        print(e)
        return
    
    print(f"Iniciando predição em {X_input.shape[0]} sequências...")
    # Inference
    predictions = model.predict(X_input, verbose=1)
    
    # Flatten results
    flat_predictions = predictions.reshape(-1, NUM_CLASSES)
    
    results = []
    for i, file_name in enumerate(file_names):
        if file_name == "PADDING":
            continue
            
        probs = flat_predictions[i]
        predicted_index = np.argmax(probs)
        predicted_label = CLASSES[predicted_index]
        confidence = probs[predicted_index]
        
        results.append({
            "frame": file_name,
            "prediction": predicted_label,
            "confidence": f"{confidence:.4f}"
        })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nSucesso! Salvo em: {output_csv}")
    print(df[["frame", "prediction", "confidence"]].head())

# --- EXECUÇÃO ---
if __name__ == "__main__":
    # AJUSTE SEUS CAMINHOS AQUI
    MODEL_PATH = "/mnt/d/resultados/lstm/breno/kfold/checkpoints/fold_1/best_model.h5" 
    INPUT_VIDEO_FOLDER = "/mnt/d/videos_alfabeto_cropped/breno/2"
    OUTPUT_FILE = "/mnt/d/resultado_predicao.csv"

    # Verificação rápida se estamos rodando na CPU (devido ao erro CUDA anterior)
    if not tf.config.list_physical_devices('GPU'):
        print("AVISO: GPU não detectada ou erro CUDA. Rodando em CPU (será mais lento, mas funcionará).")

    predict_folder(MODEL_PATH, INPUT_VIDEO_FOLDER, OUTPUT_FILE)
