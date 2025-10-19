# %% [markdown]
#   # Importa bibliotecas

# %%
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Masking, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
import matplotlib.pyplot as plt
import pickle
import os
import random
import tensorflow as tf
import cv2
import glob



# %% [markdown]
#   ## Definições

# %%
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# Diretórios
BASE_DIR = "/mnt/d/videos_fatiados"
TARGET_SIZE = (64, 64)

# Arquivos salvos fora da pasta videos_fatiados
X_FILE = "/mnt/d/X.npy"
Y_FILE = "/mnt/d/y.npy"



# %% [markdown]
#   # Funções

# %%
def save_history(history, timestamp):
    with open(f'training_history/pkl/{timestamp}.pkl', 'wb') as f:
        pickle.dump(history.history, f)


def load_video(path, max_frames=None, target_size=(64, 64)):
    """Carrega vídeo como sequência de frames grayscale normalizados"""
    cap = cv2.VideoCapture(path)
    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # preto e branco
        frame = cv2.resize(frame, target_size)           # redimensiona
        frame = frame.astype("float32") / 255.0          # normaliza
        frames.append(frame)
        count += 1
        if max_frames and count >= max_frames:
            break
    cap.release()
    return np.array(frames)  # shape: (n_frames, H, W)


def pad_and_flatten_sequences(X_raw, max_len, target_size=(64, 64)):
    """Padroniza quantidade de frames e aplica flatten (H*W)"""
    H, W = target_size
    flattened_dim = H * W
    X_padded = np.zeros((len(X_raw), max_len, flattened_dim), dtype="float32")

    for i, seq in enumerate(X_raw):
        seq_len = len(seq)
        flat_seq = seq.reshape(seq_len, -1)  # (frames, H*W)
        if seq_len <= max_len:
            X_padded[i, :seq_len, :] = flat_seq
        else:
            X_padded[i, :, :] = flat_seq[:max_len, :]
    return X_padded



# %% [markdown]
#   # Carregando ou processando dados

# %%
if os.path.exists(X_FILE) and os.path.exists(Y_FILE):
    print("Carregando dados pré-processados...")
    X = np.load(X_FILE)
    y_raw = np.load(Y_FILE, allow_pickle=True)
    max_len = X.shape[1]
else:
    print("Processando vídeos brutos...")

    X_raw = []
    y_raw = []

    # percorre os diretórios de 1 a 26
    for label_dir in sorted(os.listdir(BASE_DIR), key=lambda x: int(x)):
        full_dir = os.path.join(BASE_DIR, label_dir)
        if not os.path.isdir(full_dir):
            continue

        # pega todos os vídeos dentro da pasta
        video_files = glob.glob(os.path.join(full_dir, "*.mp4"))

        for vf in video_files:
            frames = load_video(vf, target_size=TARGET_SIZE)
            X_raw.append(frames)
            y_raw.append(label_dir)  # usa o nome da pasta como rótulo

    max_len = max(len(seq) for seq in X_raw)
    print("Número de vídeos:", len(X_raw))
    print("Maior quantidade de frames:", max_len)

    X = pad_and_flatten_sequences(X_raw, max_len, target_size=TARGET_SIZE)

    # Salva fora da pasta videos_fatiados
    np.save(X_FILE, X)
    np.save(Y_FILE, np.array(y_raw, dtype=object))
    print(f"Dados salvos em {X_FILE} e {Y_FILE}")



# %% [markdown]
#   # KFold Cross Validation

# %%
N_SPLITS = 2
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
all_fold_accuracies = []
all_fold_losses = []
histories = []

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)
num_classes = len(label_encoder.classes_)

for fold, (train_index, val_index) in enumerate(kf.split(X), 1):
    print(f"\n========== FOLD {fold} ==========")

    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y_encoded[train_index], y_encoded[val_index]

    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=(max_len, X.shape[2])))
    model.add(LSTM(128))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    checkpoint_filepath = f'models/fold_{fold}_checkpoint.model.keras'
    early_stop = EarlyStopping(patience=20, restore_best_weights=True)
    csv_logger = CSVLogger(f'training_log_fold_{fold}.csv')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.00001, verbose=1)
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    )

    history = model.fit(
        X_train, y_train,
        batch_size=8,
        epochs=50,
        verbose=1,
        validation_data=(X_val, y_val),
        shuffle=True,
        callbacks=[csv_logger, reduce_lr, model_checkpoint_callback, early_stop]
    )

    histories.append(history)

    best_model = load_model(checkpoint_filepath)
    val_loss, val_accuracy = best_model.evaluate(X_val, y_val, verbose=0)
    print(f"FOLD {fold} - Accuracy: {val_accuracy:.4f}, Loss: {val_loss:.4f}")

    all_fold_accuracies.append(val_accuracy)
    all_fold_losses.append(val_loss)



# %%
print("\n========== RESULTADOS FINAIS ==========")
for i, (acc, loss) in enumerate(zip(all_fold_accuracies, all_fold_losses), 1):
    print(f"Fold {i}: Accuracy = {acc:.4f}, Loss = {loss:.4f}")
print(f"Média de acurácia: {np.mean(all_fold_accuracies):.4f}")


# %% [markdown]
#   # Gráficos de Acurácia e Loss por Fold

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
    plt.show()



