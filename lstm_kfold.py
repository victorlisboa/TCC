# %% [markdown]
#  # Importa bibliotecas

# %%
import pandas as pd
import glob
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Masking, Dropout
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import KFold
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import datetime
import pickle
import os
import random
import tensorflow as tf


# %% [markdown]
#  ## Definições

# %%
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


# %% [markdown]
#  # Funções

# %%
def save_history(history, timestamp):
    with open(f'training_history/pkl/{timestamp}.pkl', 'wb') as f:
        pickle.dump(history.history, f)


# %% [markdown]
#  # Preparando dados

# %%
csv_files = glob.glob('/mnt/d/dados_surdos/CSVs/*.csv')

dfs = []
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
df


# %%
# Normaliza features com valores entre 0 e 1
scaler = MinMaxScaler()
landmark_cols = list(df.columns[:63])
df[landmark_cols] = scaler.fit_transform(df[landmark_cols])


# %%
# Separa features e labels por vídeo
grouped = df.groupby(['word', 'repetition'])
X_raw = []
y_raw = []

for (word, rep), group in grouped:
    sequence = group[landmark_cols].values
    X_raw.append(sequence)
    y_raw.append(word)


# %% [markdown]
#  # KFold Cross Validation

# %%
max_len = max(len(seq) for seq in X_raw)
N_SPLITS = 5

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
all_fold_accuracies = []
all_fold_losses = []
histories = []

for fold, (train_index, val_index) in enumerate(kf.split(X_raw), 1):
    print(f"\n========== FOLD {fold} ==========")

    X_train_raw = [X_raw[i] for i in train_index]
    y_train_raw = [y_raw[i] for i in train_index]
    X_val_raw   = [X_raw[i] for i in val_index]
    y_val_raw   = [y_raw[i] for i in val_index]

    X_train = pad_sequences(X_train_raw, maxlen=max_len, padding='post', dtype='float32')
    X_val   = pad_sequences(X_val_raw,   maxlen=max_len, padding='post', dtype='float32')

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    y_val = label_encoder.transform(y_val_raw)

    num_classes = len(label_encoder.classes_)

    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=(max_len, 63)))
    model.add(LSTM(256))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_filepath = f'models/fold_{fold}_checkpoint.model.keras'

    early_stop = EarlyStopping(patience=30, restore_best_weights=True)
    csv_logger = CSVLogger(f'training_log_fold_{fold}.csv')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, min_lr=0.00001, verbose=1)
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    )

    history = model.fit(
        X_train, y_train,
        batch_size=16,
        epochs=1000,
        verbose=1,
        validation_data=(X_val, y_val),
        shuffle=True,
        callbacks=[csv_logger, reduce_lr, model_checkpoint_callback]
    )

    histories.append(history)

    best_model = load_model(checkpoint_filepath)
    val_loss, val_accuracy = best_model.evaluate(X_val, y_val, verbose=0)
    print(f"FOLD {fold} - Accuracy: {val_accuracy:.4f}, Loss: {val_loss:.4f}")

    all_fold_accuracies.append(val_accuracy)
    all_fold_losses.append(val_loss)


# %%
resultados = []
resultados.append("\n========== RESULTADOS FINAIS ==========")
for i, (acc, loss) in enumerate(zip(all_fold_accuracies, all_fold_losses), 1):
    resultados.append(f"Fold {i}: Accuracy = {acc:.4f}, Loss = {loss:.4f}")
media = f"Média de acurácia: {np.mean(all_fold_accuracies):.4f}"
resultados.append(media)

# printa no terminal
for linha in resultados:
    print(linha)

# escreve no arquivo
with open('results.txt', 'w') as f:
    for linha in resultados:
        print(linha, file=f)

# %% [markdown]
#  # Gráficos de Acurácia e Loss por Fold

# %%
for i, history in enumerate(histories, 1):
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Treino')
    plt.plot(history.history['val_accuracy'], linestyle='--', label='Validação')
    plt.title(f'Fold {i} - Acurácia')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], linestyle='--', label='Validação')
    plt.title(f'Fold {i} - Loss')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()

    fig_file_path = f"training_history/img/{timestamp}.png"
    plt.savefig(fig_file_path) # se quiser salvar a imagem, descomentar a linha

    plt.show()




