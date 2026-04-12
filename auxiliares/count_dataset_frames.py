import cv2
import pandas as pd
import glob
import os
import re
from pathlib import Path

pessoa = 'pedro'
diretorio = f"/mnt/d/videos_alfabeto_cropped/{pessoa}"

def contar_frames(caminho_video):
    # Verifica se o arquivo existe
    if not os.path.exists(caminho_video):
        return None

    video = cv2.VideoCapture(str(caminho_video))

    if not video.isOpened():
        return None

    # Método Rápido
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Método Manual (Fallback)
    if total_frames <= 0:
        total_frames = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break
            total_frames += 1

    video.release()
    return total_frames

# Função para extrair o número do caminho (ajuda na ordenação numérica)
def extrair_numero(caminho):
    numeros = re.findall(r'\d+', caminho)
    return int(numeros[0]) if numeros else 0

base_dir = Path(f"/mnt/d/videos_alfabeto/{pessoa}")

lista_dados = []

# processa todos os .mp4 do diretorio
for video_path in base_dir.rglob("*.mp4"):
    n_frames = contar_frames(video_path)
    
    # Adiciona os dados à lista
    lista_dados.append({
        'frames': n_frames,
        'caminho': str(video_path)
    })

lista_dados.sort(key=lambda x: extrair_numero(x['caminho']))
lista_dados = lista_dados[:10]

# Cria o DataFrame
df_frames = pd.DataFrame(lista_dados)
print(df_frames)

# Exibe o resultado
print("Quantidade de frames originais:")
print(df_frames['frames'].sum())

# Busca todos os arquivos CSV
padrao_arquivos = os.path.join(diretorio, '**', '*.csv')
csv_files = glob.glob(padrao_arquivos, recursive=True)

# 1. Ordena os arquivos numericamente com base no nome do diretório/arquivo
csv_files.sort(key=extrair_numero)

# 2. Seleciona apenas os 10 primeiros
csv_files = csv_files[:10]

all_dfs = []
for filename in csv_files:
    df = pd.read_csv(filename)
    all_dfs.append(df)

# Concatenar e processar
df_final = pd.concat(all_dfs, ignore_index=True)

print("Quantidade de frames após rotulação")
print(df_final['frame'].count())

print(f"Quantidade de frames excluídas:\n{df_frames['frames'].sum() - df_final['frame'].count()}")
