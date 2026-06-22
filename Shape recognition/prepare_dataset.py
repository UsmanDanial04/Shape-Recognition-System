# prepare_dataset.py
"""
Dataset download and preparation for the shape classifier CNN.

Downloads Quick Draw .npy files for 5 classes, preprocesses them, and saves
train / val / test splits to the data/ folder.

Prompt 6 implementation.

Usage:
    python prepare_dataset.py
"""

import os
import urllib.request
import io
import numpy as np
import numpy.lib.format as fmt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CLASSES = ["circle", "square", "triangle", "star"]
SAMPLES_PER_CLASS = 10_000
DATA_DIR = "data"
BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_class(class_name: str) -> np.ndarray:
    """Download Quick Draw .npy file and return raw flat array."""
    out_path = os.path.join(DATA_DIR, f"{class_name}.npy")

    if os.path.exists(out_path):
        try:
            data = np.load(out_path)
            # check if it has the correct shape, if not, re-download
            if data.ndim == 2 and data.shape[0] >= SAMPLES_PER_CLASS:
                print(f"  [cache] {class_name}.npy already exists and is valid, skipping download.")
                return data
        except Exception:
            pass

    print(f"  Downloading first {SAMPLES_PER_CLASS} samples of {class_name} using Range request …")
    url = f"{BASE_URL}/{class_name}.npy"
    
    # We need SAMPLES_PER_CLASS * 784 bytes of raw data.
    # The header is typically < 128 bytes, but we request a bit more to be safe.
    range_end = 1024 + SAMPLES_PER_CLASS * 784
    req = urllib.request.Request(url, headers={'Range': f'bytes=0-{range_end}'})
    
    try:
        with urllib.request.urlopen(req) as response:
            content = response.read()
    except Exception as e:
        print(f"  Range request failed: {e}. Falling back to full download …")
        urllib.request.urlretrieve(url, out_path)
        data = np.load(out_path)
        return data

    stream = io.BytesIO(content)
    version = fmt.read_magic(stream)
    if version == (1, 0):
        shape, fortran_order, dtype = fmt.read_array_header_1_0(stream)
    else:
        shape, fortran_order, dtype = fmt.read_array_header_2_0(stream)
    header_len = stream.tell()

    raw_bytes = content[header_len : header_len + SAMPLES_PER_CLASS * 784]
    data_flat = np.frombuffer(raw_bytes, dtype=np.uint8)
    
    # Reshape to 2D (N, 784)
    data = data_flat.reshape(-1, 784)
    
    # Save as a standard, complete .npy file
    np.save(out_path, data)
    print(f"  Saved {data.shape[0]} samples to {out_path}")
    return data


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    X_all = []
    y_all = []

    for label, name in enumerate(CLASSES):
        raw  = download_class(name)               # (N, 784)  uint8 [0-255]
        raw  = raw[:SAMPLES_PER_CLASS]            # take first 10 k
        imgs = raw.reshape(-1, 28, 28)            # → (N, 28, 28)
        imgs = imgs.astype(np.float32) / 255.0   # → [0, 1]
        X_all.append(imgs)
        y_all.append(np.full(len(imgs), label, dtype=np.int32))

    X = np.concatenate(X_all, axis=0)            # (50 000, 28, 28)
    y = np.concatenate(y_all, axis=0)            # (50 000,)

    # Shuffle + split: 80 / 10 / 10
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test   = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)

    # Add channel dimension for CNN: (N, 28, 28, 1)
    X_train = X_train[..., np.newaxis]
    X_val   = X_val[...,   np.newaxis]
    X_test  = X_test[...,  np.newaxis]

    print(f"\nSplit sizes — train: {len(X_train):,} | "
          f"val: {len(X_val):,} | test: {len(X_test):,}")

    # Save
    np.save(os.path.join(DATA_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(DATA_DIR, "X_val.npy"),   X_val)
    np.save(os.path.join(DATA_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(DATA_DIR, "y_val.npy"),   y_val)
    np.save(os.path.join(DATA_DIR, "y_test.npy"),  y_test)
    print("Arrays saved to data/\n")

    # --- Visual verification: 5×5 sample grid per class -------------------
    fig, axes = plt.subplots(len(CLASSES), 5, figsize=(10, len(CLASSES) * 2))
    fig.suptitle("Quick Draw — sample grid (5 per class)", fontsize=14)

    for row, name in enumerate(CLASSES):
        # Grab 5 samples from the training split for this class
        idx      = np.where(y_train == row)[0][:5]
        samples  = X_train[idx].squeeze()          # (5, 28, 28)
        for col, img in enumerate(samples):
            ax = axes[row, col]
            ax.imshow(img, cmap="gray_r")
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(name, fontsize=11, rotation=0,
                              labelpad=40, va="center")

    plt.tight_layout()
    grid_path = os.path.join(DATA_DIR, "sample_grid.png")
    plt.savefig(grid_path, dpi=100)
    plt.show()
    print(f"Sample grid saved to {grid_path}")


if __name__ == "__main__":
    main()
