# %% [markdown]
# # Importa bibliotecas

# %%
import pandas as pd
import glob
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import SimpleRNN, Dense, Masking
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.layers import Dropout
from sklearn.preprocessing import LabelEncoder
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import datetime
import pickle

# %% [markdown]
# ## Definições

# %%
import os
import random
import tensorflow as tf

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# %% [markdown]
# # Funções

# %%
def recover_model(checkpoint_filepath):
    initial_epoch = pd.read_csv('training_log.csv')['epoch'].iloc[-1] # pega a ultima epoch registrada no log
    model = load_model(checkpoint_filepath, compile=True) # 
    
    return model, initial_epoch

def save_history(history, timestamp):
    with open(f'training_history/pkl/{timestamp}.pkl', 'wb') as f:
        pickle.dump(history.history, f)


# %% [markdown]
# # Preparando dados

# %%
csv_files = glob.glob('/mnt/d/dados_surdos/CSVs/dados_pessoa2_*.csv')

dfs = []
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
df

# %%
# Separa features e label
grouped = df.groupby(['word', 'repetition'])
X_raw = []
y_raw = []

# %%
# Normaliza features com valores entre 0 e 1
scaler = MinMaxScaler()
landmark_cols = list(df.columns[:63])
df[landmark_cols] = scaler.fit_transform(df[landmark_cols])

# %%
# prepara lista com frames agrupadas por video
for (word, rep), group in grouped:
    sequence = group[landmark_cols].values
    X_raw.append(sequence)
    y_raw.append(word)

# %% [markdown]
# X_raw é uma lista de sequências de frames, onde:
# 
# - cada item da lista representa um vídeo.
# 
# - cada vídeo é representado como um array 2D de shape (T, 63), onde:
# 
#     - T = número de frames (time steps) do vídeo (varia de vídeo para vídeo)
# 
#     - 63 = número de features por frame (21 pontos da mão × 3 coordenadas)

# %%
# Separa em treino, teste e validação de forma estratificada

X_temp, X_test, y_temp, y_test = train_test_split(
    X_raw, y_raw, test_size=0.3, stratify=y_raw)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.285, stratify=y_temp)

# %%
print(f"Treino - {(len(X_train)/len(X_raw))*100}%")
print(f"Teste - {(len(X_test)/len(X_raw))*100}%")
print(f"Validacao - {(len(X_val)/len(X_raw))*100}%")

# %%
# Padding
max_len = max(len(seq) for seq in X_train) # define tamanho maximo das sequencias

X_train = pad_sequences(X_train, maxlen=max_len, padding='post', dtype='float32')
X_val = pad_sequences(X_val,   maxlen=max_len, padding='post', dtype='float32')
X_test = pad_sequences(X_test,  maxlen=max_len, padding='post', dtype='float32')

X_train.shape,X_val.shape,X_test.shape

# %%
# Encode das labels - OneHotEncoder
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(np.array(y_train))
y_val = label_encoder.transform(np.array(y_val))
y_test = label_encoder.transform(np.array(y_test))

y_train.shape,y_test.shape,y_val.shape

# %% [markdown]
# # Criando o modelo

# %%
num_classes = 26 # palavras/labels

# %%
mask = Masking(mask_value=0.0, input_shape=(max_len, 63))
SRNN = SimpleRNN(64)
dense = Dense(num_classes, activation='softmax')

model = Sequential()
model.add(mask)
model.add(SRNN)
model.add(Dropout(0.3))
model.add(dense)

optimizer = Adam(learning_rate=0.001)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
    # metrics=['mse'] # verificar por que ta dando erro
)
model.summary()

# %% [markdown]
# # Treinamento

# %%
# callbacks
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
checkpoint_filepath = f'models/checkpoint.model.keras'

early_stop = EarlyStopping(patience=100, restore_best_weights=True)
csv_logger = CSVLogger('training_log.csv', append=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=100, min_lr=0.00001, verbose=1)
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True
)

# %%
epochs = 20000
initial_epoch = 0

# utilizar para recuperar um checkpoint perdido
# model, initial_epoch = recover_model(checkpoint_filepath) # descomentar linha para recuperar modelo

# %%
history = model.fit(
    X_train, y_train,
    batch_size=16,
    epochs=epochs,
    verbose=1,
    validation_data=(X_val, y_val),
    shuffle=True,
    callbacks=[model_checkpoint_callback,
               csv_logger,
               reduce_lr],
    initial_epoch=initial_epoch
)

# %% [markdown]
# # Avaliação do modelo

# %%
best_model = load_model(checkpoint_filepath) # o checkpoint salva apenas o melhor modelo
test_loss, test_accuracy = best_model.evaluate(X_test, y_test, verbose=0)
print(f"best model - accuracy: {test_accuracy:.4f}")
print(f"best model - loss: {test_loss:.4f}")

# %% [markdown]
# # Historico de treinamento

# %%
# historico de treinamento
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()

fig_file_path = f"training_history/img/{timestamp}.png"
# plt.savefig(fig_file_path) # se quiser salvar a imagem, descomentar a linha
plt.show()

# %%
np.max(history.history['accuracy'])

# %%
np.min(history.history['loss'])

# %%
# salva historico de treinamento e melhor modelo

# save_history(history, timestamp) # descomentar linha para salvar pkl do historico
# best_model.save(f'models/{int(test_accuracy*100)}_{timestamp}.model.keras') # descomentar linha para salvar melhor modelo

# %% [markdown]
# # Previsao do modelo

# %%
y_pred = model.predict(X_test)

y_pred = np.argmax(y_pred, axis=1)
y_true = y_test

# %%
labels = df['word'].unique()

cr = classification_report(y_true, y_pred, target_names=labels)
print(cr)

# %%
# decodifica os nomes das classes
y_pred_labels = label_encoder.inverse_transform(y_pred)
y_true_labels = label_encoder.inverse_transform(y_true)

# visualiza os pares
for i in range(y_pred_labels.shape[0]):
    print(f"{i+1}. Verdadeiro: {y_true_labels[i]} | Previsto: {y_pred_labels[i]}")



