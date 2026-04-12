import cv2
import os

def contar_frames(caminho_video):
    # Verifica se o arquivo existe
    if not os.path.exists(caminho_video):
        print("Erro: Arquivo não encontrado.")
        return None

    # Abre o arquivo de vídeo
    video = cv2.VideoCapture(caminho_video)

    if not video.isOpened():
        print("Erro ao abrir o vídeo.")
        return None

    # Método 1: Através das propriedades do arquivo (Rápido)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Método 2: Verificação manual (Caso o metadado falhe ou retorne 0)
    if total_frames <= 0:
        print("Metadados inválidos. Contando frames manualmente...")
        total_frames = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break
            total_frames += 1

    video.release()
    return total_frames

# Exemplo de uso
caminho = "/mnt/d/videos_alfabeto/fluente/1.mp4"
resultado = contar_frames(caminho)

if resultado is not None:
    print(f"O vídeo '{caminho}' possui {resultado} frames.")