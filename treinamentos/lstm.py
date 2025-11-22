import functools
import json
import math
import os
import random
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models, optimizers
from sklearn.utils.class_weight import compute_class_weight
import argparse
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

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
    split_ratios: Tuple[float, float, float]

def build_model(sequence_length: int, img_height: int, img_width: int, lstm_units: int) -> tf.keras.Sequential:
    model = models.Sequential()
    
    model.add(layers.TimeDistributed(
        layers.Flatten(), 
        input_shape=(sequence_length, img_height, img_width, 1)
    ))

    model.add(layers.LSTM(lstm_units, return_sequences=True))
    model.add(layers.TimeDistributed(layers.Dense(NUM_CLASSES, activation="softmax")))
    
    model.compile(
        optimizer=optimizers.Adam(1e-5),
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
    video_dirs = sorted([d for d in data_dir.rglob('*') if d.is_dir() and d.name.isdigit()])
    print(f"Encontrou {len(video_dirs)} diretórios de vídeo.")
    
    for video_dir in video_dirs:
        video_id = video_dir.name
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

def build_and_restore_model(cfg: TrainConfig, strategy: tf.distribute.Strategy) -> Tuple[tf.keras.Model, tf.train.CheckpointManager, int]:
    """Constrói o modelo dentro do escopo da estratégia e restaura o checkpoint."""
    checkpoint_dir_path = Path(cfg.checkpoint_dir)

    with strategy.scope():
        model = build_model(cfg.sequence_length, cfg.image_height, cfg.image_width, cfg.lstm_units)
        optimizer = model.optimizer

        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir_path, max_to_keep=3)

        start_epoch = 0
        latest_ckpt = manager.latest_checkpoint
        if latest_ckpt:
            checkpoint.restore(latest_ckpt).expect_partial()
            print(f"Restaurou checkpoint do ponto {latest_ckpt}")
            prog_file = checkpoint_dir_path / "training_progress.json"
            if prog_file.exists():
                try:
                    with open(prog_file, 'r') as f:
                        progress = json.load(f)
                    start_epoch = progress.get('epoch', 0) + 1
                    print(f"Continuando da época {start_epoch}")
                except Exception:
                    start_epoch = 0
    
    model.summary()
    
    return model, manager, start_epoch

def prepare_datasets(cfg: TrainConfig) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, int, int, int, List, Dict]:
    """
    Encapsula todo o pipeline de preparação de dados.
    Retorna os datasets de treino, validação e teste, além dos passos por época e pesos padrão.
    """
    
    # obtem metadados no nível do vídeo
    all_video_data = get_video_metadata(cfg.data_dir)
    num_videos = len(all_video_data)

    # embaralha e divide no nível do vídeo
    random.shuffle(all_video_data)
    train_split, val_split, test_split = cfg.split_ratios
    
    num_train = int(num_videos * train_split)
    num_val = int(num_videos * val_split)
    val_end_idx = num_train + num_val
    
    train_data = all_video_data[:num_train]
    val_data = all_video_data[num_train : val_end_idx]
    test_data = all_video_data[val_end_idx :]

    print("-" * 30)
    print(f"Divisão de vídeos:")
    print(f"Total: {num_videos}")
    print(f"Treino: {len(train_data)} ({int(train_split * 100)}%)")
    print(f"Validação: {len(val_data)} ({int(val_split * 100)}%)")
    print(f"Teste: {len(test_data)} ({int(test_split * 100)}%)")
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
    print("Pesos de treino calculados.")
    
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
    csv_logger_callback = tf.keras.callbacks.CSVLogger(
        filename=os.path.join(cfg.checkpoint_dir, "training_log.csv"),
        append=True
    )

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

    reduce_lr_callback = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=cfg.patience // 2,
        min_lr=1e-6,
        verbose=1
    )

    return [CheckpointCallback(), save_best_callback, early_stopping_callback, csv_logger_callback]

def plot_training_history(cfg: TrainConfig):
    """Salva os gráficos de perda e acurácia do treinamento."""
    
    log_file = Path(cfg.checkpoint_dir) / "training_log.csv"
    if not log_file.exists():
        print(f"Aviso: Arquivo de log 'training_log.csv' não encontrado em {cfg.checkpoint_dir}. Pulando o plot do histórico.")
        return

    history = pd.read_csv(log_file)

    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Treino')
    plt.plot(history['val_loss'], label='Validação')
    plt.title('Modelo Loss')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['weighted_acc'], label='Treino')
    plt.plot(history['val_weighted_acc'], label='Validação')
    plt.title('Modelo Acurácia')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'training_history_{cfg.image_height}x{cfg.image_width}_{cfg.lstm_units}.pdf')
    plt.close()

    print(f"Gráficos de treinamento salvos em 'training_history_{cfg.image_height}x{cfg.image_width}_{cfg.lstm_units}.pdf'")

def evaluate_and_save(
    best_model_path: str,
    test_steps: int,
    test_sequences: List,
    default_class_weights: Dict,
    cfg: TrainConfig
):
    """
    Avalia o melhor modelo, gera relatórios detalhados, salva e plota os resultados.
    """
    print("\n" + "=" * 30)
    print("TREINAMENTO CONCLUÍDO")
    print("=" * 30 + "\n")

    # 1. Carregar o melhor modelo
    print(f"Carregando o melhor modelo salvo de: {best_model_path}")
    best_model = tf.keras.models.load_model(best_model_path)

    # 2. Criar o dataset de teste (para avaliação padrão)
    print("Preparando dataset de teste para avaliação (Loss/Accuracy)...")
    test_dataset = create_dataset(
        test_sequences, cfg.batch_size, cfg.seed, 
        default_class_weights, cfg.sequence_length, 
        cfg.image_height, cfg.image_width,
        is_training=False
    )

    # 3. Avaliação padrão (Loss & Accuracy)
    print("Avaliando o melhor modelo (baseado em val_loss) no conjunto de teste...")
    best_model_results = best_model.evaluate(test_dataset, steps=test_steps, verbose=1)
    
    print(f"Resultados (Melhor Modelo): Loss={best_model_results[0]:.4f}, Accuracy={best_model_results[1]:.4f}")
    
    results_filename = f"best_model_results_{cfg.image_height}x{cfg.image_width}_{cfg.lstm_units}.txt"
    with open(results_filename, "w") as f:
        f.write(f"Modelo:  LSTM\n")
        f.write(f"Tamanho da Imagem: {cfg.image_height}x{cfg.image_width}\n")
        f.write(f"Número de Unidades LSTM: {cfg.lstm_units}\n")
        f.write(f"Loss: {best_model_results[0]:.4f}\n")
        f.write(f"Accuracy: {best_model_results[1]:.4f}\n")
        f.write("\n" + "="*30 + "\n")

    

    print("Gerando predições para o relatório de classificação e matriz de confusão...")
    
    # Recriamos o dataset para garantir um novo iterador
    pred_dataset = create_dataset(
        test_sequences, cfg.batch_size, cfg.seed, 
        default_class_weights, cfg.sequence_length, 
        cfg.image_height, cfg.image_width,
        is_training=False
    )

    all_true_labels = []
    all_pred_labels = []

    for images, labels, _ in pred_dataset.take(test_steps):
        predictions = best_model.predict(images, verbose=0)
        predicted_indices = np.argmax(predictions, axis=-1)
        
        all_true_labels.extend(labels.numpy().flatten())
        all_pred_labels.extend(predicted_indices.flatten())

    # relatório de classificação
    print("Gerando Relatório de Classificação...")
    report = classification_report(
        all_true_labels, 
        all_pred_labels, 
        labels=range(NUM_CLASSES),
        target_names=CLASSES,
        zero_division=0
    )
    print(report)
    
    # adiciona relatorio ao arquivo de resultados
    with open(results_filename, "a") as f:
        f.write("Relatório de Classificação:\n")
        f.write(report)

    print(f"Relatório de Classificação salvo em '{results_filename}'")

    # gera e salva matriz de confusão
    print("Gerando Matriz de Confusão...")
    cm = confusion_matrix(
        all_true_labels, 
        all_pred_labels, 
        labels=range(NUM_CLASSES)
    )
    
    # plota a matriz
    fig, ax = plt.subplots(figsize=(18, 18))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    display.plot(ax=ax, xticks_rotation='vertical', cmap='viridis', values_format='d')
    
    plt.title('Matriz de Confusão')
    plt.tight_layout()
    cm_filename = f'confusion_matrix_{cfg.image_height}x{cfg.image_width}_{cfg.lstm_units}.pdf'
    plt.savefig(cm_filename)
    plt.close(fig)
    print(f"Matriz de confusão salva em '{cm_filename}'")


    # plota histórico de treinamento
    plot_training_history(cfg)

def run_experiment(img_size: int, lstm_units: int):
    """Executa um experimento completo com os parâmetros fornecidos."""
    # 1. Configuração
    cfg = TrainConfig(
        data_dir=Path("/home/vitorlisboa/datasets/videos_alfabeto_cropped/fluente"),
        epochs=1000,
        batch_size=2,
        sequence_length=32,
        image_height=img_size,
        image_width=img_size,
        lstm_units=lstm_units,
        patience=20,
        seed=42,
        device="auto",
        checkpoint_dir=f"./checkpoints_lstm_experimentos/checkpoints_{img_size}x{img_size}_{lstm_units}",
        split_ratios=(0.6, 0.2, 0.2)
    )

    # 2. Setup do Ambiente (Seeds, GPU, Estratégia)
    strategy = setup_environment(cfg)

    # 3. Construção do Modelo e Restauração de Checkpoint
    model, manager, start_epoch = build_and_restore_model(cfg, strategy)

    # 4. Preparação dos Datasets
    (
        train_dataset, val_dataset, test_dataset, 
        steps_per_epoch, validation_steps, test_steps,
        test_sequences, default_class_weights
    ) = prepare_datasets(cfg)

    # 5. Criação dos Callbacks
    callbacks = create_callbacks(cfg, manager)

    # 6. Treinamento
    print("\nIniciando o treinamento...")
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

    # 7. Avaliação e Salvamento
    evaluate_and_save(
        os.path.join(cfg.checkpoint_dir, "best_model.h5"),
        test_steps,
        test_sequences,
        default_class_weights,
        cfg
    )

def main():
    """
    Função principal que recebe parametros de tamanho da imagem e unidades da LSTM e executa o experimento.
    Args:
        --img_size: Tamanho da imagem (altura e largura)
        --lstm_units: Número de unidades LSTM
    """

    # 1. Configura o parser para ler os argumentos
    parser = argparse.ArgumentParser(description="Executa um experimento LSTM.")
    parser.add_argument("--img_size", type=int, required=True, 
                        help="Tamanho da imagem (altura e largura)")
    parser.add_argument("--lstm_units", type=int, required=True,
                        help="Número de unidades LSTM")

    args = parser.parse_args()

    img_size = args.img_size
    lstm_units = args.lstm_units

    print(f"\n\nIniciando experimento com tamanho de imagem {img_size}x{img_size} e {lstm_units} unidades LSTM.\n")
    try:
        run_experiment(img_size, lstm_units)

    except Exception as e:
        print(f"Erro durante o experimento (Size: {img_size}, Units: {lstm_units}): {e}")
        
    print(f"Experimento (Size: {img_size}, Units: {lstm_units}) concluído.")


if __name__ == "__main__":
    main()
