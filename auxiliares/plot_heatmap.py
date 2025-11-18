import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

img_size_list = [32, 64, 128, 256]
lstm_unit_list = [256, 512, 1024, 2048, 4096]
base_dir = "/mnt/d/resultados/lstm/experimentos/paciencia_20"

"""
Gera heatmaps de acurácia e perda a partir de logs de experimentos,
baseado em listas pré-definidas de parâmetros.
"""
    
results = []
base_dir_path = Path(base_dir)

print(f"Iniciando varredura em: {base_dir}")
print(f"Tamanhos de Imagem a procurar: {img_size_list}")
print(f"Unidades LSTM a procurar: {lstm_unit_list}")

for img_size in img_size_list:
    for lstm_unit in lstm_unit_list:
        
        dir_name = f'checkpoints_{img_size}x{img_size}_{lstm_unit}'
        log_file = base_dir_path / dir_name / 'training_log.csv'
        
        try:
            df = pd.read_csv(log_file)
                        
            best_val_acc = df['val_acc'].max()
            best_val_loss = df['val_loss'].min()
                
            results.append({
                'img_size': img_size,
                'lstm_units': lstm_unit,
                'accuracy': best_val_acc,
                'loss': best_val_loss
            })
            print(f"Processado: {img_size}x{lstm_unit} | Melhor Acc: {best_val_acc:.4f}, Menor Loss: {best_val_loss:.4f}")
        except Exception as e:
            print(f"Erro ao processar o arquivo {log_file}: {e}")

results_df = pd.DataFrame(results)

# Pivotar para Acurácia
# index = linhas (Y), columns = colunas (X), values = células

acc_pivot = results_df.pivot_table(
    index='lstm_units', 
    columns='img_size', 
    values='accuracy'
)
# Assegura que a ordem dos eixos seja a mesma das listas de entrada
acc_pivot = acc_pivot.reindex(index=sorted(lstm_unit_list), columns=sorted(img_size_list))

# Pivotar para Perda
loss_pivot = results_df.pivot_table(
    index='lstm_units', 
    columns='img_size', 
    values='loss'
)
loss_pivot = loss_pivot.reindex(index=sorted(lstm_unit_list), columns=sorted(img_size_list))

# Heatmap de Acurácia
plt.figure(figsize=(12, 8))
sns.heatmap(
    acc_pivot, 
    annot=True,     # Mostrar os valores nas células
    fmt=".4f",      # Formatar com 4 casas decimais
    cmap="Reds", # 'viridis': Alto = Amarelo (bom), Baixo = Roxo (ruim)
    linecolor='white',
    linewidths=0.5
)
plt.title('Heatmap de Acurácia Máxima de Validação')
plt.xlabel('Tamanho da Imagem')
plt.ylabel('Unidades LSTM')
plt.savefig('heatmap_acuracia.png')
plt.close()
print("Salvo: heatmap_acuracia.png")

# Heatmap de Perda
plt.figure(figsize=(12, 8))
sns.heatmap(
    loss_pivot, 
    annot=True, 
    fmt=".4f",
    cmap="Blues_r", # '_r' inverte o mapa. 'plasma_r': Baixo = Amarelo (bom), Alto = Roxo (ruim)
    linecolor='white',
    linewidths=0.5
)
plt.title('Heatmap de Perda Mínima de Validação')
plt.xlabel('Tamanho da Imagem')
plt.ylabel('Unidades LSTM')
plt.savefig('heatmap_perda.png')
plt.close()
print("Salvo: heatmap_perda.png")
