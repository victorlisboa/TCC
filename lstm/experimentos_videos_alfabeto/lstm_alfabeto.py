from dataclasses import dataclass
import os
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable
import cv2
import numpy as np
import tensorflow as tf
from keras import layers, models, optimizers

@dataclass
class TrainConfig:
    data_dir: Path
    epochs: int = 5
    batch_size: int = 1
    steps_per_epoch: int = 0
    val_steps: int = 0
    seed: int = 42
    device: str = "gpu"
    checkpoint_dir: str = "./checkpoints"

def build_model() -> tf.keras.Model:
    inp = layers.Input(shape=(None, 256, 256, 1), name="frames")
    # Raw images only: flatten each frame, then LSTM over time
    x = layers.TimeDistributed(layers.Flatten())(inp)
    x = layers.LayerNormalization()(x)
    x = layers.LSTM(256, return_sequences=True)(x)
    out = layers.TimeDistributed(layers.Dense(NUM_CLASSES, activation="softmax"))(x)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    return model

def main():
    cfg = TrainConfig(
        data_dir=Path("/mnt/d/videos_alfabeto_cropped/breno"),
        # data_dir=Path("/home/vitorlisboa/datasets/videos_alfabeto_cropped/breno"),
        epochs=5,
        batch_size=1,
        seed=42,
        device="gpu",
        checkpoint_dir="./checkpoints",
    )

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    tf.random.set_seed(cfg.seed)

    if not cfg.data_dir.exists():
        raise FileNotFoundError(f"diretório de dados não encontrado: {cfg.data_dir}")

    # find GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print('GPUs encontradas:\n',gpus)
    else:
        print('Nenhuma GPU encontrada')

    # train_stems, val_stems = split_stems(cfg.data_dir)
    # if len(train_stems) == 0:
    #     raise RuntimeError(f"No trainable items in {cfg.data_dir}. Expect directories 1-10 with frame_*.jpg files and corresponding CSV labels.")

    model = build_model()
    model.summary()

    # Load checkpoint if exists
    optimizer = model.optimizer
    start_epoch = load_training_state(model, optimizer, cfg.checkpoint_dir)

    train_gen = make_generator(cfg.data_dir, train_stems, batch_size=cfg.batch_size, shuffle=True)
    val_gen = make_generator(cfg.data_dir, val_stems, batch_size=cfg.batch_size, shuffle=False) if len(val_stems) > 0 else None

    steps_per_epoch = math.ceil(len(train_stems) / cfg.batch_size)
    val_steps = math.ceil(len(val_stems) / cfg.batch_size) if len(val_stems) > 0 else None

    # Custom callback to save training state after each epoch
    class CheckpointCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            save_training_state(self.model, self.model.optimizer, epoch, cfg.checkpoint_dir)

    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath="./lstm_alfabeto_best.keras",
        monitor="val_acc" if val_gen is not None else "acc",
        mode="max",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )

    callbacks = [ckpt_cb, CheckpointCallback()]

    history = model.fit(
        ,
        steps_per_epoch=steps_per_epoch,
        epochs=cfg.epochs,
        initial_epoch=start_epoch,
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1,
    )

    model.save("./lstm_alfabeto_final.keras")
    print("Saved final model to ./lstm_alfabeto_final.keras")


if __name__ == "__main__":
    main()
