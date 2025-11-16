import pandas as pd
from pathlib import Path

# Substitua 'seu_arquivo.csv' pelo nome real do seu arquivo
dir_base = '/home/victor/desktop/unb/tcc/resultados/lstm/experimentos_img_units/paciencia_20/checkpoints/'

tamanhos = [256, 512, 1024, 2048, 4096]
menor_val_loss = float('inf')
melhor_tamanho = None
epoch = None

for tamanho in tamanhos:
    nome_do_arquivo = Path(dir_base) / f'checkpoints_32x32_{tamanho}/training_log.csv'
    
    try:
        # Lê o arquivo CSV para um DataFrame
        df = pd.read_csv(nome_do_arquivo)
        
        # Verifica se a coluna 'val_loss' existe
        if 'val_loss' in df.columns:
            # Encontra o menor valor na coluna 'val_loss'
            val_loss = df['val_loss'].min()
            print(f"O menor val_loss para o tamanho {tamanho} é: {val_loss}")
            if val_loss < menor_val_loss:
                menor_val_loss = val_loss
                melhor_tamanho = tamanho
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

print(f"\nO menor val_loss foi com o tamanho {melhor_tamanho}: {menor_val_loss} na epoca {epoch}")
