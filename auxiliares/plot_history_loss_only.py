import pandas as pd
import matplotlib.pyplot as plt

input_dir = '/mnt/d/resultados/lstm/breno/experimentos/5K_epocas/checkpoints_32x32_2048/' 

# Carregar os dados do arquivo CSV
df = pd.read_csv(input_dir + 'training_log.csv')

# Encontrar a época e o valor da menor loss de validação
min_loss_row = df.loc[df['val_loss'].idxmin()]
min_epoch = min_loss_row['epoch']
min_loss = min_loss_row['val_loss']

# Criar o gráfico
plt.figure(figsize=(10, 6))

# Plotar a perda de treinamento e a perda de validação
plt.plot(df['epoch'], df['loss'], label='Training Loss')
plt.plot(df['epoch'], df['val_loss'], label='Validation Loss')

# Adicionar um ponto no valor mínimo da validação
plt.scatter(min_epoch, min_loss, color='red', zorder=5, label='Min Val Loss')

# Adicionar uma linha horizontal no nível do menor loss
# linestyle='--' cria o efeito tracejado
plt.axhline(y=min_loss, color='red', linestyle='--', alpha=1)

# Adicionar rótulos e título
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Histórico de Perda (Loss) durante o Treinamento')
plt.legend()
plt.grid(True)

# Salvar o gráfico
plt.savefig(input_dir + 'training_history_loss.png', transparent=True)