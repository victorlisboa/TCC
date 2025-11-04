from dataclasses import dataclass
from sklearn.utils.class_weight import compute_class_weight
import os
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple, Generator
import cv2
import numpy as np
import tensorflow as tf
from keras import layers, models, optimizers
import matplotlib.pyplot as plt

CLASSES = [
    "*",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
]
CLASS_TO_INDEX: Dict[str, int] = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

@dataclass
class TrainConfig:
    data_dir: Path
    epochs: int
    sequence_length: int
    batch_size: int
    seed: int
    device: str
    checkpoint_dir: str
    split_ratios: Tuple[float, float, float]

def build_model() -> tf.keras.Sequential:
    
    cnn = models.Sequential([
        layers.Input(shape=(256, 256, 1)),
        
        layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)), # -> (128, 128, 32)
        
        layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)), # -> (64, 64, 64)

        layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)), # -> (32, 32, 128)

        layers.GlobalAveragePooling2D(),    # -> (128,)
        # layers.Flatten(),

        layers.Dense(1024, activation='relu'),

    ], name="cnn_feature_extractor")

    model = models.Sequential()
    
    model.add(layers.TimeDistributed(cnn, input_shape=(None, 256, 256, 1)))
    
    model.add(layers.LSTM(512,
                          return_sequences=True,
                          dropout=0.3,
    ))

    model.add(layers.TimeDistributed(layers.Dropout(0.7)))
    
    model.add(layers.TimeDistributed(layers.Dense(NUM_CLASSES, activation="softmax")))

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
        weighted_metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    return model

def get_video_metadata(data_dir: Path) -> List[Tuple[List[str], List[int]]]:
    """Coleta caminhos de frames e labels sem carregar imagens.
    Retorna uma lista de tuplas (frame_paths, labels) para cada vídeo.
    """
    all_video_data = []
    video_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    print(f"Encontrou {len(video_dirs)} diretórios de vídeo.")
    
    for video_dir in video_dirs:
        video_id = video_dir.name
        csv_file = data_dir / f"{video_id}.csv"
        
        frame_labels_map = {}
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame_num = int(row['frame'])
                label = row['label']
                frame_labels_map[frame_num] = label
        # frame_labels_map mapeia número do frame para o label correspondente
        
        frame_files = sorted([f for f in video_dir.iterdir() if f.suffix.lower() == '.jpg'])
        
        video_frame_paths = []
        video_labels = []
        
        for frame_file in frame_files:
            frame_num = int(frame_file.stem.split('_')[1])
            label_str = frame_labels_map.get(frame_num, '*')
            label_idx = CLASS_TO_INDEX.get(label_str, CLASS_TO_INDEX['*'])  # usa label '*' caso nao encontre a label mapeada
            video_frame_paths.append(str(frame_file))
            video_labels.append(label_idx)
        
        if not video_frame_paths:
            print(f"Aviso: Nenhum frame .jpg encontrado em {video_dir}. Pulando.")
            continue
            
        all_video_data.append((video_frame_paths, video_labels))
        
    print(f"Metadados de {len(all_video_data)} vídeos coletados.")
    return all_video_data

def data_generator(video_metadata: List[Tuple[List[str], List[int]]], 
                   sequence_length: int, 
                   class_weights: Dict[int, float]) -> Generator:
    """
    Gera sequências de tamanho fixo de vídeo.
    Descarta os frames restantes no final de um vídeo se não formarem uma sequência completa.
    """
    for (frame_paths, labels) in video_metadata:
        video_frames = []
        video_labels = []
        for frame_path, label in zip(frame_paths, labels):
            frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            if frame is None:
                print(f"Aviso: Falha ao carregar frame {frame_path}. Pulando.")
                continue
            frame = frame.astype(np.float32) / 255.0    # normaliza para [0, 1]
            frame = np.expand_dims(frame, axis=-1)      # (256, 256, 1)
            video_frames.append(frame)
            video_labels.append(label)

        if not video_frames:
            continue

        X_video = np.array(video_frames, dtype=np.float32)
        y_video = np.array(video_labels, dtype=np.int32)
        
        # fatia o vídeo em sequências de tamanho 'sequence_length'
        num_frames = len(X_video)
        num_sequences = num_frames // sequence_length  # divisão inteira descarta o final

        for i in range(num_sequences):
            start = i * sequence_length
            end = start + sequence_length

            X_sequence = X_video[start:end]
            y_sequence = y_video[start:end]

            # usa os pesos da classe para CADA frame na sequência
            sw_sequence = np.array([class_weights[label] for label in y_sequence], dtype=np.float32)

            if X_sequence.shape[0] == sequence_length:
                yield X_sequence, y_sequence, sw_sequence

def create_dataset(
    metadata: List[Tuple[List[str], List[int]]],
    sequence_length: int,
    batch_size: int,
    seed: int,
    class_weights: Dict[int, float],
    is_training: bool = True
) -> tf.data.Dataset:
    
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(metadata, sequence_length, class_weights),
        output_signature=(
            tf.TensorSpec(shape=(sequence_length, 256, 256, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(sequence_length,), dtype=tf.int32),
            tf.TensorSpec(shape=(sequence_length,), dtype=tf.float32)
        )
    )
    
    if is_training:
        total_sequences = 0
        for video_paths, _ in metadata:
            total_sequences += len(video_paths) // sequence_length
        
        # devolve um dataset vazio se não houver sequências
        if total_sequences == 0:
            return tf.data.Dataset.from_tensor_slices((
                tf.zeros([0, sequence_length, 256, 256, 1], dtype=tf.float32),
                tf.zeros([0, sequence_length], dtype=tf.int32),
                tf.zeros([0, sequence_length], dtype=tf.float32)
            )).batch(batch_size)

        dataset = dataset.repeat()
        # embaralha no nível dos chunks
        dataset = dataset.shuffle(buffer_size=total_sequences, seed=seed)
    
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def main():
    cfg = TrainConfig(
        data_dir=Path("/home/vitorlisboa/datasets/videos_alfabeto_cropped/breno"),
        epochs=50,
        sequence_length=32,
        batch_size=2,
        seed=42,
        device="gpu",
        checkpoint_dir="./checkpoints",
        split_ratios=(0.6, 0.2, 0.2) # 60% treino, 20% val, 20% teste
    )

    # settando as seeds
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    tf.random.set_seed(cfg.seed)

    if not cfg.data_dir.exists():
        raise FileNotFoundError(f"diretório de dados não encontrado: {cfg.data_dir}")

    gpus = tf.config.list_physical_devices('GPU')
    if gpus: print('GPUs encontradas:\n', gpus)
    else: print('Nenhuma GPU encontrada')
    
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = build_model()
        model.summary()

    optimizer = model.optimizer

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(checkpoint, cfg.checkpoint_dir, max_to_keep=3)

    start_epoch = 0
    latest_ckpt = manager.latest_checkpoint
    if latest_ckpt:
        checkpoint.restore(latest_ckpt).expect_partial()
        print(f"Restored checkpoint from {latest_ckpt}")
        prog_file = Path(cfg.checkpoint_dir) / "training_progress.json"
        if prog_file.exists():
            try:
                with open(prog_file, 'r') as f:
                    progress = json.load(f)
                start_epoch = progress.get('epoch', 0) + 1
                print(f"Resuming from epoch {start_epoch}")
            except Exception:
                start_epoch = 0
    
    all_video_data = get_video_metadata(cfg.data_dir)
    num_videos = len(all_video_data)

    # randomiza os dados antes de dividir
    random.shuffle(all_video_data)

    # divide os dados em treino, validação e teste
    train_split = cfg.split_ratios[0]
    val_split = cfg.split_ratios[1]
    test_split = cfg.split_ratios[2]
    
    num_train = int(num_videos * train_split)
    num_val = int(num_videos * val_split)
    val_end_idx = num_train + num_val
    
    train_data = all_video_data[:num_train]
    val_data = all_video_data[num_train : val_end_idx]
    test_data = all_video_data[val_end_idx :]

    print("-" * 30)
    print(f"Dados divididos:")
    print(f"Total: {num_videos}")
    print(f"Treino: {len(train_data)} ({int(train_split * 100)}%)")
    print(f"Validação: {len(val_data)} ({int(val_split * 100)}%)")
    print(f"Teste: {len(test_data)} ({int(test_split * 100)}%)")
    print("-" * 30)
    
    if not train_data or not val_data or not test_data:
        raise ValueError("Divisão de dados resultou em um conjunto vazio. Verifique 'num_videos'.")
    
    print("-" * 30)
    print("Calculando pesos das classes (class weights)...")
    
    # Coleta todos os labels do conjunto de treino
    all_train_labels = []
    for _, labels in train_data:
        all_train_labels.extend(labels)

    class_indices = np.array(range(NUM_CLASSES))
    
    # Calcula os pesos balanceados
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=class_indices,
        y=all_train_labels
    )
    
    # Converte para um dicionário para consulta rápida
    train_class_weights = {i: weight for i, weight in enumerate(class_weights_array)}
    print("Pesos de treino calculados.")
    
    # Cria pesos padrão (todos 1.0) para validação e teste
    default_class_weights = {i: 1.0 for i in class_indices}

    def calculate_total_chunks(metadata, sequence_length):
        total_chunks = 0
        if not metadata:
            return 0
        for video_paths, _ in metadata:
            total_chunks += len(video_paths) // sequence_length
        return total_chunks
    
    total_train_chunks = calculate_total_chunks(train_data, cfg.sequence_length)
    total_val_chunks = calculate_total_chunks(val_data, cfg.sequence_length)
    total_test_chunks = calculate_total_chunks(test_data, cfg.sequence_length)

    if total_train_chunks == 0:
        raise ValueError(
            f"Nenhum 'chunk' de treino foi gerado. "
            f"Verifique se seus vídeos de treino são mais longos que 'sequence_length' ({cfg.sequence_length})."
        )
    
    train_dataset = create_dataset(train_data, cfg.sequence_length, cfg.batch_size, cfg.seed, train_class_weights, is_training=True)
    val_dataset = create_dataset(val_data, cfg.sequence_length, cfg.batch_size, cfg.seed, default_class_weights, is_training=False)
    test_dataset = create_dataset(test_data, cfg.sequence_length, cfg.batch_size, cfg.seed, default_class_weights, is_training=False)
    
    steps_per_epoch = total_train_chunks // cfg.batch_size
    validation_steps = total_val_chunks // cfg.batch_size
    test_steps = total_test_chunks // cfg.batch_size

    # (Você pode adicionar avisos se validation_steps ou test_steps forem 0)
    if total_val_chunks == 0:
        print("Aviso: Nenhum 'chunk' de validação gerado.")
    if total_test_chunks == 0:
         print("Aviso: Nenhum 'chunk' de teste gerado.")
    
    class CheckpointCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            path = manager.save()
            with open(Path(cfg.checkpoint_dir) / "training_progress.json", "w") as f:
                json.dump({"epoch": epoch}, f)
            print(f"\nCheckpoint salvo: {path}")

    best_model_path = os.path.join(cfg.checkpoint_dir, "best_model.h5")
    
    save_best_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=best_model_path,
        save_weights_only=False, # salva o modelo inteiro
        monitor='val_loss',
        mode='min',
        save_best_only=True,     # salva somente se for melhor
        verbose=1
    )

    callbacks = [
        CheckpointCallback(),  # Para resumir o treino
        save_best_callback     # Para salvar o melhor modelo
    ]

    # treinamento
    history = model.fit(
        train_dataset,
        epochs=cfg.epochs,
        initial_epoch=start_epoch,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1,
    )

    print("\n" + "=" * 30)
    print("TREINAMENTO CONCLUÍDO")
    print("=" * 30 + "\n")

    
    # Avaliação final

    print(f"Carregando o MELHOR modelo salvo de: {best_model_path}")
    best_model = tf.keras.models.load_model(best_model_path)
    
    print("Avaliando o MELHOR modelo (baseado em val_loss) no conjunto de teste...")
    best_model_results = best_model.evaluate(test_dataset, steps=test_steps, verbose=1)
    print(f"Resultados (Melhor Modelo): Loss={best_model_results[0]:.4f}, Accuracy={best_model_results[1]:.4f}")

    # Plotagem dos gráficos de treinamento
    plt.figure(figsize=(12, 4))
    
    # Gráfico de Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title('Modelo Loss')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    
    # Gráfico de Acurácia
    plt.subplot(1, 2, 2)
    plt.plot(history.history['acc'], label='Treino')
    plt.plot(history.history['val_acc'], label='Validação')
    plt.title('Modelo Acurácia')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    print("Gráficos de treinamento salvos em 'training_history.png'")


if __name__ == "__main__":
    main()
