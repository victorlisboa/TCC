import pandas as pd
from pathlib import Path

dir_base = '/mnt/d/resultados/cnn_lstm/pedro/experimentos/paciencia_20/checkpoints'

img_sizes = [32, 64, 128, 256]
lstm_units = [256, 512, 1024, 2048, 4096]
menor_val_loss = float('inf')
melhor_tamanho = None
melhor_unidade = None
epoch = None

for lstm_unit in lstm_units:
    for img_size in img_sizes:
        nome_do_arquivo = Path(dir_base) / f'checkpoints_{img_size}x{img_size}_{lstm_unit}/training_log.csv'
        
        try:
            # Lê o arquivo CSV para um DataFrame
            df = pd.read_csv(nome_do_arquivo)
            
            # Verifica se a coluna 'val_loss' existe
            if 'val_loss' in df.columns:
                # Encontra o menor valor na coluna 'val_loss'
                val_loss = df['val_loss'].min()
                print(f"O menor val_loss para imagem {img_size} e {lstm_unit} unidades é: {val_loss}")
                if val_loss < menor_val_loss:
                    menor_val_loss = val_loss
                    melhor_tamanho = img_size
                    melhor_unidade = lstm_unit
                    epoch = df['val_loss'].idxmin()
            else:
                print(f"Erro: A coluna 'val_loss' não foi encontrada no arquivo '{nome_do_arquivo}'.")
                print(f"Colunas disponíveis: {df.columns.tolist()}")

        except FileNotFoundError:
            print(f"Erro: O arquivo '{nome_do_arquivo}' não foi encontrado.")
        except pd.errors.EmptyDataError:
            print(f"Erro: O arquivo '{nome_do_arquivo}' está vazio.")
        except Exception as e:
            print(f"Ocorreu um erro inesperado: {e}")

print(f"\nO menor val_loss foi com a imagem tamanho {melhor_tamanho} e LSTM tamanho {melhor_unidade}: {menor_val_loss} na epoca {epoch}")
