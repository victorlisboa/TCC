import pandas as pd
import glob
import matplotlib.pyplot as plt
import os

pessoa = ''
diretorio = f"/mnt/d/videos_alfabeto_cropped/{pessoa}"
padrao_arquivos = os.path.join(diretorio, '**', '*.csv')

csv_files = glob.glob(padrao_arquivos, recursive=True)

print(f"Arquivos CSV encontrados: {len(csv_files)}")

all_dfs = []
for filename in csv_files:
    df = pd.read_csv(filename)
    all_dfs.append(df)

df = pd.concat(all_dfs, ignore_index=True)

label_counts = df['label'].value_counts()

counts_df = label_counts.reset_index()
counts_df.columns = ['label', 'count']

print("\nContagem de valores para a coluna 'label':")
soma_total = counts_df['count'].sum()
counts_df['percentage'] = ((counts_df['count'] / soma_total) * 100).round(2)

print(counts_df.to_string())

plt.figure(figsize=(12, 7))

label_counts.plot(kind='bar', color='seagreen', edgecolor='black')

plt.title('Histograma de rótulos do conjunto de dados', fontsize=16)
plt.xlabel('Rótulo', fontsize=12)
plt.ylabel('Frequência', fontsize=12)

plt.xticks(rotation=45, ha='right')

plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()

output_filename = f'histograma_rotulos_{pessoa}.pdf'
plt.savefig(output_filename)

print(f"\nGráfico salvo como: {output_filename}")
