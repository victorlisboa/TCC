import functools
import json
import math
import os
import random
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models, optimizers
from sklearn.utils.class_weight import compute_class_weight
import dataclasses
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GroupKFold
import argparse

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
    batch_size: int
    sequence_length: int
    image_height: int
    image_width: int
    lstm_units: int
    patience: int
    seed: int
    device: str
    checkpoint_dir: str

def build_model(sequence_length: int, img_height: int, img_width: int, lstm_units: int) -> tf.keras.Sequential:
    
    # Definição CNN
    cnn = models.Sequential([
        layers.Input(shape=(img_height, img_width, 1)),
        
        layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.SpatialDropout2D(0.2),
        
        layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.SpatialDropout2D(0.2),

        layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.SpatialDropout2D(0.2),
        
    ], name="cnn_feature_extractor")

    model = models.Sequential()
    
    model.add(layers.TimeDistributed(
        cnn, 
        input_shape=(sequence_length, img_height, img_width, 1)
    ))

    # model.add(layers.TimeDistributed(layers.GlobalAveragePooling2D()))
    model.add(layers.TimeDistributed(layers.Flatten()))
    model.add(layers.TimeDistributed(layers.Dense(lstm_units, activation='relu')))  # o tamanho da dense é o mesmo da LSTM
    model.add(layers.TimeDistributed(layers.Dropout(0.3)))

    model.add(layers.LSTM(
        lstm_units,
        return_sequences=True,
        dropout=0.3,
    ))

    model.add(layers.TimeDistributed(layers.Dropout(0.3)))
    
    model.add(layers.TimeDistributed(layers.Dense(NUM_CLASSES, activation="softmax")))

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
        weighted_metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    return model

def get_video_metadata(data_dir: Path) -> List[Tuple[List[str], List[int]]]:
    """Coleta caminhos de frames e labels sem carregar imagens.
       Retorna uma lista de tuplas (lista_de_caminhos_dos_frames, lista_de_labels) para cada vídeo.
    """
    all_video_data = []
    # procura por diretórios cujo nome é um dígito em todo os subdiretórios de data_dir
    video_dirs = sorted([d for d in data_dir.rglob('*') if d.is_dir() and d.name.isdigit()])
    print(f"Encontrou {len(video_dirs)} diretórios de vídeo.")
    
    for video_dir in video_dirs:
        video_id = video_dir.name
        # assume que o csv está no mesmo diretório que os diretorios de cada vídeo
        csv_file = video_dir.parent / f"{video_id}.csv"
        
        frame_labels_map = {}
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame_num = int(row['frame'])
                label = row['label']
                frame_labels_map[frame_num] = label
        
        frame_files = sorted([f for f in video_dir.iterdir() if f.suffix.lower() == '.jpg'])
        
        video_frame_paths = []
        video_labels = []
        
        for frame_file in frame_files:
            frame_num = int(frame_file.stem.split('_')[1])
            label_str = frame_labels_map.get(frame_num, '*')
            label_idx = CLASS_TO_INDEX.get(label_str, CLASS_TO_INDEX['*'])
            video_frame_paths.append(str(frame_file))
            video_labels.append(label_idx)
        
        if not video_frame_paths:
            print(f"Aviso: Nenhum frame .jpg encontrado em {video_dir}. Pulando.")
            continue
            
        all_video_data.append((video_frame_paths, video_labels))
        
    print(f"Metadados de {len(all_video_data)} vídeos coletados.")
    return all_video_data

def create_sequence_metadata(
    video_metadata: List[Tuple[List[str], List[int]]],
    sequence_length: int,
    stride: int
) -> List[Tuple[List[str], List[int]]]:
    """Cria "chunks" de metadados (sequências) a partir de metadados de vídeos inteiros."""
    all_sequences = []
    for (frame_paths, labels) in video_metadata:
        video_len = len(frame_paths)
        
        for i in range(0, video_len - sequence_length + 1, stride):
            chunk_paths = frame_paths[i : i + sequence_length]
            chunk_labels = labels[i : i + sequence_length]
            
            if len(chunk_paths) == sequence_length:
                all_sequences.append((chunk_paths, chunk_labels))
                
    return all_sequences

@tf.function
def load_image(path_tensor: tf.Tensor, img_height: int, img_width: int) -> tf.Tensor:
    """Lê, decodifica, redimensiona e normaliza uma imagem usando ops do TF."""
    img_bytes = tf.io.read_file(path_tensor)
    img = tf.io.decode_jpeg(img_bytes, channels=1) 
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.image.convert_image_dtype(img, tf.float32)
    img.set_shape([img_height, img_width, 1]) 
    return img

@tf.function
def load_sequence(paths_tensor: tf.Tensor, 
                  labels_tensor: tf.Tensor, 
                  img_height: int, 
                  img_width: int,
                  class_weights_tensor: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Carrega todas as imagens em uma sequência e calcula os pesos."""
    images = tf.map_fn(
        lambda p: load_image(p, img_height, img_width), 
        paths_tensor, 
        dtype=tf.float32
    )
    sample_weights = tf.gather(class_weights_tensor, labels_tensor)
    return images, labels_tensor, sample_weights

def create_dataset(
    metadata: List[Tuple[List[str], List[int]]], 
    batch_size: int, 
    seed: int,
    class_weights: Dict[int, float],
    sequence_length: int,
    img_height: int,
    img_width: int,
    is_training: bool,
) -> tf.data.Dataset:
    """Cria um tf.data.Dataset a partir dos metadados da sequência usando tf.data."""
    all_paths = [paths for paths, labels in metadata]
    all_labels = [labels for paths, labels in metadata]
    
    if not all_paths:
        print("Aviso: create_dataset recebeu metadados vazios.")
        return tf.data.Dataset.from_tensor_slices((
            tf.zeros([0, sequence_length], dtype=tf.string),
            tf.zeros([0, sequence_length], dtype=tf.int32)
        )).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    weights_list = [class_weights[i] for i in range(NUM_CLASSES)]
    class_weights_tensor = tf.constant(weights_list, dtype=tf.float32)

    dataset = tf.data.Dataset.from_tensor_slices((all_paths, all_labels))
    
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    dataset = dataset.with_options(options)

    if is_training:
        dataset = dataset.shuffle(buffer_size=len(metadata), seed=seed)
        dataset = dataset.repeat()

    load_fn = functools.partial(
        load_sequence, 
        img_height=img_height, 
        img_width=img_width, 
        class_weights_tensor=class_weights_tensor
    )
    
    dataset = dataset.map(load_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def setup_environment(cfg: TrainConfig) -> tf.distribute.Strategy:
    """Configura seeds, GPUs e estratégia de distribuição."""
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    tf.random.set_seed(cfg.seed)

    if not cfg.data_dir.exists():
        raise FileNotFoundError(f"Diretório de dados não encontrado: {cfg.data_dir}")

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print('GPUs encontradas:\n', gpus)
    else:
        print('Nenhuma GPU encontrada')
    
    # Retorna a estratégia para ser usada no 'scope'
    return tf.distribute.MirroredStrategy()

def prepare_fold_datasets(
    cfg: TrainConfig,
    all_video_data: List[Tuple[List[str], List[int]]],
    train_val_indices: np.ndarray,
    test_indices: np.ndarray
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, int, int, int, List, Dict]:
    """
    Prepara os datasets para um ÚNICO FOLD da validação cruzada.
    """
    
    # separa os dados do fold
    test_data = [all_video_data[i] for i in test_indices]
    train_val_list = [all_video_data[i] for i in train_val_indices]

    # embaralha e divide no nível do vídeo
    random.shuffle(train_val_list)
    val_data = train_val_list[:1]
    train_data = train_val_list[1:]

    num_videos = len(all_video_data)

    print("-" * 30)
    print(f"Divisão de vídeos:")
    print(f"Total: {num_videos}")
    print(f"Treino: {len(train_data)} ({int(len(train_data) / num_videos * 100)}%)")
    print(f"Validação: {len(val_data)} ({int(len(val_data) / num_videos * 100)}%)")
    print(f"Teste: {len(test_data)} ({int(len(test_data) / num_videos * 100)}%)")
    print("-" * 30)
    
    if not train_data or not val_data or not test_data:
        raise ValueError("Divisão de dados (vídeos) resultou em um conjunto vazio.")

    # cria de sequências (chunks)
    print(f"Criando sequências (chunks) de {cfg.sequence_length} frames...")
    train_stride = cfg.sequence_length // 2 # gera sobreposição para dados de treino
    val_test_stride = cfg.sequence_length

    train_sequences = create_sequence_metadata(train_data, cfg.sequence_length, train_stride)
    val_sequences = create_sequence_metadata(val_data, cfg.sequence_length, val_test_stride)
    test_sequences = create_sequence_metadata(test_data, cfg.sequence_length, val_test_stride)

    print("-" * 30)
    print(f"Divisão de Sequências:")
    print(f"Sequências de Treino:    {len(train_sequences)}")
    print(f"Sequências de Validação: {len(val_sequences)}")
    print(f"Sequências de Teste:     {len(test_sequences)}")
    print("-" * 30)
    
    if not train_sequences or not val_sequences or not test_sequences:
        raise ValueError("Criação de sequência resultou em um conjunto vazio.")

    # cálculo de pesos (baseado apenas nos vídeos de treino)
    print("Calculando pesos das classes (class weights)...")
    all_train_labels = []
    for _, labels in train_data:
        all_train_labels.extend(labels)
    
    if not all_train_labels:
        raise ValueError("Nenhum label encontrado nos dados de treino.")

    class_indices = np.array(range(NUM_CLASSES))
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=class_indices,
        y=all_train_labels
    )

    train_class_weights = {i: weight for i, weight in enumerate(class_weights_array)}
    default_class_weights = {i: 1.0 for i in class_indices}

    # cria datasets
    train_dataset = create_dataset(
        train_sequences, cfg.batch_size, cfg.seed, 
        train_class_weights, cfg.sequence_length,
        cfg.image_height, cfg.image_width,
        is_training=True
    )
    val_dataset = create_dataset(
        val_sequences, cfg.batch_size, cfg.seed, 
        default_class_weights, cfg.sequence_length, 
        cfg.image_height, cfg.image_width,
        is_training=False
    )
    test_dataset = create_dataset(
        test_sequences, cfg.batch_size, cfg.seed, 
        default_class_weights, cfg.sequence_length, 
        cfg.image_height, cfg.image_width,
        is_training=False
    )
    
    # calcula steps
    steps_per_epoch = math.ceil(len(train_sequences) / cfg.batch_size)
    validation_steps = len(val_sequences) // cfg.batch_size
    test_steps = len(test_sequences) // cfg.batch_size
    
    return (
        train_dataset, val_dataset, test_dataset, 
        steps_per_epoch, validation_steps, test_steps,
        test_sequences, default_class_weights # Retorna para re-avaliação
    )


def create_callbacks(cfg: TrainConfig, manager: tf.train.CheckpointManager) -> List[tf.keras.callbacks.Callback]:
    """Cria e retorna a lista de callbacks para o treinamento."""
    
    # Define a classe do callback internamente
    class CheckpointCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            path = manager.save()
            with open(Path(cfg.checkpoint_dir) / "training_progress.json", "w") as f:
                json.dump({"epoch": epoch}, f)
            print(f"\nCheckpoint salvo: {path}")

    # salva logs de treino
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)  # necessário criar diretório para CSVLogger
    csv_logger_callback = tf.keras.callbacks.CSVLogger(filename=os.path.join(cfg.checkpoint_dir, "training_log.csv"))

    best_model_path = os.path.join(cfg.checkpoint_dir, f"best_model.h5")
    
    save_best_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=best_model_path,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=cfg.patience,
        restore_best_weights=True
    )

    return [CheckpointCallback(), save_best_callback, early_stopping_callback, csv_logger_callback]


def plot_training_history(history, cfg: TrainConfig, fold_num: int):
    """Salva os gráficos de perda e acurácia do treinamento para um fold."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title(f'Fold {fold_num} - Modelo Loss')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['acc'], label='Treino')
    plt.plot(history.history['val_acc'], label='Validação')
    plt.title(f'Fold {fold_num} - Modelo Acurácia')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()
    
    plt.tight_layout()
    
    plot_path = os.path.join(cfg.checkpoint_dir, f'fold_{fold_num}_training_history.pdf')
    plt.savefig(plot_path)
    plt.close()

    print(f"Gráficos de treinamento do Fold {fold_num} salvos em '{plot_path}'")


def evaluate_fold(
    history: tf.keras.callbacks.History, 
    best_model_path: str,
    test_steps: int,
    test_sequences: List,
    default_class_weights: Dict,
    cfg: TrainConfig,
    fold_num: int
) -> Tuple[List[float], Dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Avalia o melhor modelo, gera relatórios detalhados, salva e plota os resultados."""
    
    # Plota o histórico de treinamento para este fold
    plot_training_history(history, cfg, fold_num)
    
    print(f"\nAvaliação do FOLD {fold_num}")

    # carrega o melhor modelo
    best_model = tf.keras.models.load_model(best_model_path)

    # cria o dataset de teste
    test_dataset = create_dataset(
        test_sequences, cfg.batch_size, cfg.seed, 
        default_class_weights, cfg.sequence_length, 
        cfg.image_height, cfg.image_width,
        is_training=False
    )

    # avalicação do melhor modelo
    best_model_results = best_model.evaluate(test_dataset, steps=test_steps, verbose=1)
    
    results_filename = os.path.join(cfg.checkpoint_dir, f"fold_{fold_num}_results.txt")
    print(f"Resultados (Melhor Modelo): Loss={best_model_results[0]:.4f}, Accuracy={best_model_results[1]:.4f}")
    
    with open(results_filename, "w") as f:
        f.write(f"Fold: {fold_num}\n")
        f.write(f"Loss: {best_model_results[0]:.4f}\n")
        f.write(f"Accuracy: {best_model_results[1]:.4f}\n")
        f.write("\n" + "="*30 + "\n")

    all_true_labels = []
    all_pred_labels = []

    # Recriamos o dataset para garantir um novo iterador
    pred_dataset = create_dataset(
        test_sequences, cfg.batch_size, cfg.seed, 
        default_class_weights, cfg.sequence_length, 
        cfg.image_height, cfg.image_width,
        is_training=False
    )

    for images, labels, _ in pred_dataset.take(test_steps):
        predictions = best_model.predict(images, verbose=0)
        predicted_indices = np.argmax(predictions, axis=-1)
        
        all_true_labels.extend(labels.numpy().flatten())
        all_pred_labels.extend(predicted_indices.flatten())

    # gera relatório de classificação
    report_dict = classification_report(
        all_true_labels, 
        all_pred_labels, 
        labels=range(NUM_CLASSES), 
        target_names=CLASSES,
        zero_division=0,
        output_dict=True  # Retorna um dicionário!
    )
    report_str = classification_report(
        all_true_labels, 
        all_pred_labels, 
        labels=range(NUM_CLASSES), 
        target_names=CLASSES,
        zero_division=0
    )
    print(report_str)
    
    # Adicionar relatório ao arquivo de resultados
    with open(results_filename, "a") as f:
        f.write("Relatório de Classificação:\n")
        f.write(report_str)

    # 6. gera e salva matriz de confusão
    print("Gerando Matriz de Confusão...")
    cm = confusion_matrix(
        all_true_labels, 
        all_pred_labels, 
        labels=range(NUM_CLASSES)
    )
    
    # plotar a matriz de confusão
    fig, ax = plt.subplots(figsize=(18, 18))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    display.plot(ax=ax, xticks_rotation='vertical', cmap='viridis', values_format='d')
    cm_filename = os.path.join(cfg.checkpoint_dir, f'fold_{fold_num}_confusion_matrix.pdf')
    plt.savefig(cm_filename)
    plt.close(fig)

    print(f"Matriz de confusão salva em '{cm_filename}'")

    return best_model_results, report_dict, np.array(all_true_labels), np.array(all_pred_labels), cm


def save_final_results(
    all_fold_metrics: List[List[float]],
    all_true: np.ndarray,
    all_pred: np.ndarray,
    total_cm: np.ndarray,
    img_size: int,
    lstm_units: int,
    base_dir: str
):
    """ Salva os resultados agregados de todos os folds."""
    print("\n" + "=" * 50)
    print("Validação Cruzada Concluída.")
    print("=" * 50 + "\n")
    
    metrics_array = np.array(all_fold_metrics)
    mean_loss = np.mean(metrics_array[:, 0])
    std_loss = np.std(metrics_array[:, 0])
    mean_acc = np.mean(metrics_array[:, 1])
    std_acc = np.std(metrics_array[:, 1])

    print(f"Resultados dos {len(all_fold_metrics)} Folds:")
    print(f"Loss Média:    {mean_loss:.4f} +/- {std_loss:.4f}")
    print(f"Acc Média:     {mean_acc:.4f} +/- {std_acc:.4f}")

    # salva o resultado final
    results_filename = Path(base_dir) / f"kfold_results.txt"
    with open(results_filename, "w") as f:
        f.write("\nResultados da Validação Cruzada\n")
        f.write(f"ImgSize: {img_size}, LSTM Units: {lstm_units}\n")
        f.write(f"Loss Média:    {mean_loss:.4f} +/- {std_loss:.4f}\n")
        f.write(f"Acc Média:     {mean_acc:.4f} +/- {std_acc:.4f}\n")
        f.write("\n" + "="*30 + "\n")
        
        report_str = classification_report(
            all_true, 
            all_pred, 
            labels=range(NUM_CLASSES), 
            target_names=CLASSES,
            zero_division=0
        )
        f.write("Relatório de Classificação Agregado:\n")
        f.write(report_str)

    # gera e salva matriz de confusão agregada
    fig, ax = plt.subplots(figsize=(18, 18))
    display = ConfusionMatrixDisplay(confusion_matrix=total_cm, display_labels=CLASSES)
    display.plot(ax=ax, xticks_rotation='vertical', cmap='viridis', values_format='d')
    plt.title('Matriz de Confusão Agregada')
    plt.tight_layout()
    cm_filename = Path(base_dir) / f'kfold_confusion_matrix.pdf'
    plt.savefig(cm_filename)
    plt.close(fig)
    print(f"Matriz de confusão agregada salva em '{cm_filename}'")


def main():
    '''K-Fold Cross-Validation para CNN+LSTM em vídeos do alfabeto de libras.'''
    
    # configuração k-fold
    cfg = TrainConfig(
        data_dir=Path("/home/vitorlisboa/datasets/videos_alfabeto_cropped/pedro"),
        epochs=10000,
        batch_size=2,
        sequence_length=32,
        image_height=256,
        image_width=256,
        lstm_units=1024,
        patience=20,
        seed=42,
        device="auto",
        checkpoint_dir=f"./checkpoints_cnn_lstm_kfold",
    )

    # setup do ambiente
    strategy = setup_environment(cfg)

    # coleta metadados de todos os vídeos
    all_video_data = get_video_metadata(cfg.data_dir)
    num_videos = len(all_video_data)
    print(f"Total de vídeos coletados: {num_videos}")

    video_indices = np.arange(num_videos)

    # deixa sempre 1 vídeo para teste
    # e os outros para treino/validação
    group_kfold = GroupKFold(n_splits=num_videos)

    all_fold_metrics = []
    all_true_labels_list = []
    all_pred_labels_list = []
    total_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

    print(f"\n\nIniciando K-Fold com tamanho de imagem {cfg.image_height}x{cfg.image_height} e {cfg.lstm_units} unidades LSTM.\n")

    for fold, (train_val_indices, test_indices) in enumerate(group_kfold.split(video_indices, groups=video_indices)):
        fold_num = fold + 1
        print("\n" + "="*50)
        print(f"--- Iniciando fold {fold_num} / {num_videos} ---")
        print(f"Índice de vídeo de teste: {test_indices[0]}")
        print(f"Índices de vídeo de treino/val: {train_val_indices}")
        print("="*50 + "\n")

        # cria um diretório de checkpoint específico para este fold
        fold_checkpoint_dir = os.path.join(cfg.checkpoint_dir, f"fold_{fold_num}")
        os.makedirs(fold_checkpoint_dir, exist_ok=True)
        
        # Atualiza o cfg para este fold
        fold_cfg = dataclasses.replace(cfg, checkpoint_dir=fold_checkpoint_dir)

        # construção do modelo
        with strategy.scope():
            model = build_model(fold_cfg.sequence_length, fold_cfg.image_height, fold_cfg.image_width, fold_cfg.lstm_units)
            optimizer = model.optimizer
            checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
            manager = tf.train.CheckpointManager(checkpoint, Path(fold_cfg.checkpoint_dir), max_to_keep=3)

        # preparação dos datasets
        (
            train_dataset, val_dataset, test_dataset, 
            steps_per_epoch, validation_steps, test_steps,
            test_sequences, default_class_weights
        ) = prepare_fold_datasets(fold_cfg, all_video_data, train_val_indices, test_indices)


        # criação dos callbacks
        callbacks = create_callbacks(fold_cfg, manager)

        # treinamento
        print(f"\nIniciando treinamento do fold {fold_num}...")
        history = model.fit(
            train_dataset,
            epochs=fold_cfg.epochs,
            initial_epoch=0,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_dataset,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1,
        )

        # avaliacao do fold
        fold_metrics, _, fold_true, fold_pred, fold_cm = evaluate_fold(
            history,
            os.path.join(fold_cfg.checkpoint_dir, "best_model.h5"),
            test_steps,
            test_sequences, default_class_weights, fold_cfg, fold_num
        )
        
        all_fold_metrics.append(fold_metrics)
        all_true_labels_list.append(fold_true)
        all_pred_labels_list.append(fold_pred)
        total_cm += fold_cm

    all_true_agg = np.concatenate(all_true_labels_list)
    all_pred_agg = np.concatenate(all_pred_labels_list)
        
    save_final_results(
        all_fold_metrics, 
        all_true_agg, 
        all_pred_agg,
        total_cm, 
        cfg.image_width, 
        cfg.lstm_units,
        cfg.checkpoint_dir
    )

if __name__ == "__main__":
    main()
