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
    """Read .jpg frames from a directory, sorted, as grayscale 256x256 float32 in [0,1]."""
    if not dir_path.exists() or not dir_path.is_dir():
        raise FileNotFoundError(f"Frames directory not found: {dir_path}")
    jpgs = sorted(dir_path.glob("*.jpg"))
    if len(jpgs) == 0:
        raise FileNotFoundError(f"No .jpg frames found in: {dir_path}")
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
    """Pad sequences in a batch to the same length and build sample weights mask.
    Returns (X_pad, y_pad, sw) with shapes [B, T, 256,256,1], [B, T], [B, T].
    """
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
    base_dir = base_dir.expanduser().resolve()
    """Return directory names that have matching CSV files."""
    dirs = {p.name for p in base_dir.iterdir() if p.is_dir()}
    csvs = {p.stem for p in base_dir.glob("*.csv")}
    candidates = sorted(list(dirs & csvs), key=lambda s: (len(s), s))
    numeric = [s for s in candidates if s.isdigit()]
    if len(numeric) == len(candidates) and len(numeric) > 0:
        # If all are numeric, ensure ordering and optionally restrict to 1..10 if present
        ordered = sorted(numeric, key=lambda s: int(s))
        # If 1..10 exist, keep only those
        wanted = {str(i) for i in range(1, 11)}
        subset = [s for s in ordered if s in wanted]
        return subset if len(subset) > 0 else ordered
    return candidates


def split_stems(base_dir: Path) -> Tuple[List[str], List[str]]:
    stems = stems_from_dir(base_dir)
    numeric = [s for s in stems if s.isdigit()]
    if len(numeric) == len(stems):
        nums = sorted([int(s) for s in stems])
        train = [str(n) for n in nums if n <= 8]
        val = [str(n) for n in nums if n >= 9]
        return train, val
    rng = random.Random(42)
    rng.shuffle(stems)
    k = max(1, int(0.2 * len(stems)))
    return stems[k:], stems[:k]


def make_generator(base_dir: Path, stems: List[str], batch_size: int = 1, shuffle: bool = True) -> Iterable:
    """Python generator yielding (X, y, sample_weight) for Keras fit.
    - Supports variable sequence lengths with padding (batch_size >= 1).
    - Uses sparse labels per timestep.
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
    # Use raw images only: flatten each frame and feed to LSTM
    x = layers.TimeDistributed(layers.Flatten())(inp)  # [B, T, 256*256]
    # Optional normalization to stabilize training on raw pixel ranges
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
    device: str = "auto"  # "cpu" | "gpu" | "auto"


def setup_devices(device: str) -> None:
    if device == "auto":
        return
    if device == "cpu":
        tf.config.set_visible_devices([], "GPU")
    elif device == "gpu":
        # leave defaults; error if no GPU will just fallback
        pass


def main():
    # Hardcoded configuration (no CLI args)
    cfg = TrainConfig(
        # data_dir=_expand_path(Path("mnt/d/videos_alfabeto_cropped/breno")),
        data_dir=_expand_path(Path("~/datasets/videos_alfabeto_cropped/breno")),
        epochs=5,
        batch_size=1,
        seed=42,
        device="auto",
    )

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    tf.random.set_seed(cfg.seed)
    setup_devices(cfg.device)

    if not cfg.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {cfg.data_dir}")

    train_stems, val_stems = split_stems(cfg.data_dir)
    if len(train_stems) == 0:
        raise RuntimeError(f"No trainable items in {cfg.data_dir}. Expect numbered dirs with .jpg frames and <n>.csv labels.")

    model = build_model()
    model.summary()

    train_gen = make_generator(cfg.data_dir, train_stems, batch_size=cfg.batch_size, shuffle=True)
    if len(val_stems) > 0:
        val_gen = make_generator(cfg.data_dir, val_stems, batch_size=cfg.batch_size, shuffle=False)
    else:
        val_gen = None

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


