#!/usr/bin/env python3
# E01 — Head-only baseline

import os, json, time, socket, subprocess
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.metrics import AUC
from sklearn.model_selection import train_test_split

# -----------------------
# Config
# -----------------------
SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5
NUM_LABELS = 5
EXP_ID = "E01"
BACKBONE = "densenet121"
NPZ_PATH = os.path.expanduser("~/chexpert_project/data/processed/npz/chexpert_ruleA_30000.npz")
OUT_DIR  = Path.home() / "chexpert_project" / "outputs"

# -----------------------
# Utils
# -----------------------
class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epoch_times = []
    def on_epoch_begin(self, epoch, logs=None):
        self._start = time.time()
    def on_epoch_end(self, epoch, logs=None):
        self.epoch_times.append(time.time() - self._start)

def gpu_info():
    try:
        q = subprocess.check_output(
            ["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"], text=True
        ).strip().splitlines()
        return q[0] if q else "unknown"
    except Exception:
        return "unknown"

# Good GPU manners
for g in tf.config.list_physical_devices("GPU"):
    try: tf.config.experimental.set_memory_growth(g, True)
    except: pass

tf.random.set_seed(SEED); np.random.seed(SEED)

# -----------------------
# Data
# -----------------------
data = np.load(NPZ_PATH)
X = data["X"]                              # uint8 (N,H,W,3)
Y = data["Y"].astype("float32")            # (N,5)
Xtr, Xval, Ytr, Yval = train_test_split(X, Y, test_size=0.2, random_state=SEED, shuffle=True)

AUTOTUNE = tf.data.AUTOTUNE
def norm(x,y): return tf.cast(x, tf.float32)/255.0, y
train_ds = (tf.data.Dataset.from_tensor_slices((Xtr,Ytr))
            .shuffle(min(8192,len(Xtr)), seed=SEED, reshuffle_each_iteration=True)
            .map(norm, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE))
val_ds   = (tf.data.Dataset.from_tensor_slices((Xval,Yval))
            .map(norm, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE))

# -----------------------
# Bundle folder
# -----------------------
stamp = time.strftime("%Y%m%d_%H%M%S")
bundle_name = f"{BACKBONE}_{EXP_ID}_{IMG_SIZE[0]}x{IMG_SIZE[1]}_{stamp}"
BUNDLE = OUT_DIR / "bundles" / bundle_name
BUNDLE.mkdir(parents=True, exist_ok=True)

MODEL_OUT = BUNDLE / "model_best.keras"
CSV_OUT   = BUNDLE / "train_log.csv"
META_OUT  = BUNDLE / "run_meta.json"
ENV_OUT   = BUNDLE / "env.txt"

# -----------------------
# Model (head-only)
# -----------------------
base = DenseNet121(include_top=False, weights="imagenet", input_shape=(*IMG_SIZE,3))
for l in base.layers: l.trainable = False

inp = layers.Input(shape=(*IMG_SIZE,3))
x = base(inp, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
out = layers.Dense(NUM_LABELS, activation="sigmoid")(x)
model = models.Model(inp, out)

opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=opt, loss="binary_crossentropy",
              metrics=[AUC(multi_label=True, num_labels=NUM_LABELS, name="AUC")])

time_cb = TimeHistory()
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(str(MODEL_OUT), save_best_only=True,
                                       monitor="val_loss", mode="min", verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2,
                                     restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2,
                                         patience=1, min_lr=1e-6, verbose=1),
    tf.keras.callbacks.CSVLogger(str(CSV_OUT), append=False),
    time_cb,
]

# -----------------------
# Train
# -----------------------
hist = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=2)

# Final val
val_loss, val_auc = model.evaluate(val_ds, verbose=0)

# -----------------------
# Save meta & env
# -----------------------
meta = {
    "bundle": str(BUNDLE),
    "exp_id": EXP_ID,
    "backbone": BACKBONE,
    "img_size": IMG_SIZE,
    "batch": BATCH_SIZE,
    "epochs": EPOCHS,
    "optimizer": "adam",
    "lr": 1e-4,
    "dropout": 0.5,
    "dense_units": 256,
    "policy": "Rule A (NaN->0,-1->0); CheXpert-5",
    "npz": NPZ_PATH,
    "n_train": int(len(Xtr)), "n_val": int(len(Xval)),
    "val_loss": float(val_loss), "val_auc": float(val_auc),
    "epoch_times_sec": time_cb.epoch_times,
    "host": socket.gethostname(),
    "gpu": gpu_info(),
    "seed": SEED,
    "stamp": stamp,
    "notes": "E01 head-only baseline; bundle contains model_best.keras and train_log.csv."
}
with open(META_OUT, "w") as f: json.dump(meta, f, indent=2)

with open(ENV_OUT, "w") as f:
    f.write(f"tensorflow: {tf.__version__}\n")
    try:
        import numpy as np, pandas as pd
        f.write(f"numpy: {np.__version__}\n")
        f.write(f"pandas: {pd.__version__}\n")
    except Exception: pass

print("\n[done] Bundle →", BUNDLE)
print("      Model  →", MODEL_OUT)
print("      CSV    →", CSV_OUT)
print("      Meta   →", META_OUT)
