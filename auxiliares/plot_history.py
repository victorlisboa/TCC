import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

log_file = Path("/mnt/d/resultados/lstm/breno/experimentos/5K_epocas/checkpoints_32x32_2048/training_log.csv")
output_file = f'/mnt/d/resultados/training_history_breno_250_epochs_32_2048.pdf'

if not log_file.exists():
    print(f"Aviso: Arquivo de log 'training_log.csv' não encontrado. Pulando o plot do histórico.")

history = pd.read_csv(log_file).iloc[:251] # dá pra customizar pra pegar só uma parcela do treinamento

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['loss'], label='Treino')
plt.plot(history['val_loss'], label='Validação')
plt.title('Modelo Loss')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['weighted_acc'], label='Treino')
plt.plot(history['val_weighted_acc'], label='Validação')
plt.title('Modelo Acurácia')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()

plt.tight_layout()
plt.savefig(output_file)
plt.close()

print(f"Gráficos de treinamento salvos.")
