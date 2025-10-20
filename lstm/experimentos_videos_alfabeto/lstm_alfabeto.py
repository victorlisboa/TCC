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
from tensorflow.keras import layers, models, optimizers


CLASSES = [
    "*",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
]
CLASS_TO_INDEX: Dict[str, int] = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)


def _expand_path(p: Path) -> Path:
    return p.expanduser().resolve()


def read_labels(csv_path: Path) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    if not csv_path.exists():
        return mapping
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame_idx = int(row.get("frame", "0"))
            except Exception:
                continue
            label_raw = (row.get("label", "*") or "*").strip().lower()
            label_idx = CLASS_TO_INDEX.get(label_raw, CLASS_TO_INDEX["*"])
            mapping[frame_idx] = label_idx
    return mapping


def read_dir_frames_gray256(dir_path: Path) -> List[np.ndarray]:
    """Read frame_*.jpg files from directory, sorted by frame number, as grayscale 256x256 float32 in [0,1]."""
    if not dir_path.exists() or not dir_path.is_dir():
        raise FileNotFoundError(f"Frames directory not found: {dir_path}")
    # Get all frame_*.jpg files and sort by frame number
    jpgs = sorted(dir_path.glob("frame_*.jpg"), key=lambda p: int(p.stem.split('_')[1]))
    if len(jpgs) == 0:
        raise FileNotFoundError(f"No frame_*.jpg files found in: {dir_path}")
    frames: List[np.ndarray] = []
    for img_path in jpgs:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if img.shape != (256, 256):
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        frames.append(img)
    return frames


def load_sequence(base_dir: Path, stem: str) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (X, y) where
    - X: [T, 256, 256, 1] float32 in [0,1]
    - y: [T] int labels (sparse)
    Frames are read from base_dir/<stem>/*.jpg and labels from base_dir/<stem>.csv
    """
    frames_dir = base_dir / stem
    csv_path = base_dir / f"{stem}.csv"
    frames = read_dir_frames_gray256(frames_dir)
    labels_map = read_labels(csv_path)
    T = len(frames)
    X = np.stack(frames, axis=0)[..., None]  # [T, 256, 256, 1]
    y = np.array([labels_map.get(i + 1, CLASS_TO_INDEX["*"]) for i in range(T)], dtype=np.int32)
    return X, y


def pad_batch(seqs: List[np.ndarray], labels: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Unused in frame-level training; kept for compatibility if needed later
    lengths = [s.shape[0] for s in seqs]
    B = len(seqs)
    T = max(lengths)
    X_pad = np.zeros((B, T, 256, 256, 1), dtype=np.float32)
    y_pad = np.zeros((B, T), dtype=np.int32)
    sw = np.zeros((B, T), dtype=np.float32)
    for i, (x, y) in enumerate(zip(seqs, labels)):
        t = x.shape[0]
        X_pad[i, :t] = x
        y_pad[i, :t] = y
        sw[i, :t] = 1.0
    return X_pad, y_pad, sw


def stems_from_dir(base_dir: Path) -> List[str]:
    """Return directory names 1-10 that have both directory and CSV file."""
    base_dir = base_dir.expanduser().resolve()
    stems = []
    for i in range(1, 11):
        stem = str(i)
        dir_path = base_dir / stem
        csv_path = base_dir / f"{stem}.csv"
        if dir_path.exists() and dir_path.is_dir() and csv_path.exists():
            stems.append(stem)
    return stems


def split_stems(base_dir: Path) -> Tuple[List[str], List[str]]:
    """Split directories 1-10: train=1-8, val=9-10."""
    stems = stems_from_dir(base_dir)
    train = [str(i) for i in range(1, 8 + 1) if str(i) in stems]
    val = [str(i) for i in range(9, 10 + 1) if str(i) in stems]
    return train, val


def make_generator(base_dir: Path, stems: List[str], batch_size: int = 1, shuffle: bool = True) -> Iterable:
    """Yield (X_pad, y_pad, sample_weight) batches of sequences with padding.
    - X_pad: [B, T, 256, 256, 1], y_pad: [B, T], sample_weight: [B, T]
    """
    stems = list(stems)
    while True:
        if shuffle:
            random.shuffle(stems)
        for i in range(0, len(stems), batch_size):
            batch_stems = stems[i:i + batch_size]
            seqs: List[np.ndarray] = []
            labels: List[np.ndarray] = []
            for s in batch_stems:
                X, y = load_sequence(base_dir, s)
                seqs.append(X)
                labels.append(y)
            X_pad, y_pad, sw = pad_batch(seqs, labels)
            yield X_pad, y_pad, sw


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


@dataclass
class TrainConfig:
    data_dir: Path
    epochs: int = 5
    batch_size: int = 1  # variable-length friendly
    steps_per_epoch: int = 0  # 0 => auto from dataset size
    val_steps: int = 0       # 0 => auto
    seed: int = 42
    device: str = "gpu"  # "cpu" | "gpu" | "auto"



def main():
    cfg = TrainConfig(
        data_dir=Path("/mnt/d/videos_alfabeto_cropped/breno"),
        # data_dir=_expand_path(Path("~/datasets/videos_alfabeto_cropped/breno")),
        epochs=5,
        batch_size=64,
        seed=42,
        device="auto",
    )

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    tf.random.set_seed(cfg.seed)

    if not cfg.data_dir.exists():
        raise FileNotFoundError(f"diretório de dados não encontrado: {cfg.data_dir}")

    train_stems, val_stems = split_stems(cfg.data_dir)
    if len(train_stems) == 0:
        raise RuntimeError(f"No trainable items in {cfg.data_dir}. Expect directories 1-10 with frame_*.jpg files and corresponding CSV labels.")

    model = build_model()
    model.summary()

    train_gen = make_generator(cfg.data_dir, train_stems, batch_size=cfg.batch_size, shuffle=True)
    val_gen = make_generator(cfg.data_dir, val_stems, batch_size=cfg.batch_size, shuffle=False) if len(val_stems) > 0 else None

    steps_per_epoch = math.ceil(len(train_stems) / cfg.batch_size)
    val_steps = math.ceil(len(val_stems) / cfg.batch_size) if len(val_stems) > 0 else None

    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath="./lstm_alfabeto_best.keras",
        monitor="val_acc" if val_gen is not None else "acc",
        mode="max",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=cfg.epochs,
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=[ckpt_cb],
        verbose=1,
    )

    model.save("./lstm_alfabeto_final.keras")
    print("Saved final model to ./lstm_alfabeto_final.keras")


if __name__ == "__main__":
    main()
