#!/usr/bin/env python3
# E05 — Full fine-tune like E03 but gentler LR (3e-5). No aug

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
BATCH = 32
WARMUP_EPOCHS = 3
FT_EPOCHS = 10
PATIENCE = 2
NUM_LABELS = 5
LR_WARMUP = 1e-4
LR_FINETUNE = 3e-5

NPZ_PATH = os.path.expanduser("~/chexpert_project/data/processed/npz/chexpert_ruleA_30000.npz")
OUT_DIR  = Path.home() / "chexpert_project" / "outputs"
BUNDLE   = OUT_DIR / "bundles" / "model_E05"

MODEL_OUT = BUNDLE / "model_E05.keras"
CSV_OUT   = BUNDLE / "train_log.csv"
META_OUT  = BUNDLE / "run_E05_meta.json"
ENV_OUT   = BUNDLE / "env.txt"

# -----------------------
# System niceties
# -----------------------
cpu_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "0") or 0)
if cpu_threads:
    tf.config.threading.set_intra_op_parallelism_threads(cpu_threads)
    tf.config.threading.set_inter_op_parallelism_threads(max(1, cpu_threads // 2))

for g in tf.config.list_physical_devices("GPU"):
    try: tf.config.experimental.set_memory_growth(g, True)
    except: pass

tf.random.set_seed(SEED); np.random.seed(SEED)

def gpu_info():
    try:
        q = subprocess.check_output(
            ["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"], text=True
        ).strip().splitlines()
        return q[0] if q else "unknown"
    except Exception:
        return "unknown"

class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None): self.epoch_times=[]
    def on_epoch_begin(self, epoch, logs=None): self._t=time.time()
    def on_epoch_end(self, epoch, logs=None): self.epoch_times.append(time.time()-self._t)

class LrLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            logs["learning_rate"] = lr

# -----------------------
# Data
# -----------------------
data = np.load(NPZ_PATH)
X = data["X"]                              
Y = data["Y"].astype("float32")            

Xtr, Xval, Ytr, Yval = train_test_split(
    X, Y, test_size=0.2, random_state=SEED, shuffle=True
)

AUTOTUNE = tf.data.AUTOTUNE
def norm(x,y): return tf.cast(x, tf.float32)/255.0, y

train_ds = (tf.data.Dataset.from_tensor_slices((Xtr,Ytr))
            .shuffle(min(8192,len(Xtr)), seed=SEED, reshuffle_each_iteration=True)
            .map(norm, num_parallel_calls=AUTOTUNE)
            .batch(BATCH).prefetch(AUTOTUNE))

val_ds   = (tf.data.Dataset.from_tensor_slices((Xval,Yval))
            .map(norm, num_parallel_calls=AUTOTUNE)
            .batch(BATCH).prefetch(AUTOTUNE))

# -----------------------
# Bundle
# -----------------------
BUNDLE.mkdir(parents=True, exist_ok=True)

# -----------------------
# Model
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

# Phase 1 — warmup head only
opt1 = tf.keras.optimizers.Adam(learning_rate=LR_WARMUP)
model.compile(optimizer=opt1, loss="binary_crossentropy",
              metrics=[AUC(multi_label=True, num_labels=NUM_LABELS, name="AUC")])

t1 = TimeHistory()
cbs1 = [
    tf.keras.callbacks.ModelCheckpoint(str(MODEL_OUT), save_best_only=True,
                                       monitor="val_loss", mode="min", verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                     restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2,
                                         patience=1, min_lr=1e-6, verbose=1),
    tf.keras.callbacks.CSVLogger(str(CSV_OUT), append=False),
    LrLogger(), t1,
]
print("[E05] Warmup head-only for", WARMUP_EPOCHS, "epochs @", LR_WARMUP)
model.fit(train_ds, validation_data=val_ds, epochs=WARMUP_EPOCHS, callbacks=cbs1, verbose=2)

# Phase 2 — unfreeze ALL, use gentler LR
print("[E05] Unfreezing ALL layers, fine-tuning @", LR_FINETUNE, "(clipnorm=1.0)")
for l in base.layers: l.trainable = True

opt2 = tf.keras.optimizers.Adam(learning_rate=LR_FINETUNE, clipnorm=1.0)
model.compile(optimizer=opt2, loss="binary_crossentropy",
              metrics=[AUC(multi_label=True, num_labels=NUM_LABELS, name="AUC")])

t2 = TimeHistory()
cbs2 = [
    tf.keras.callbacks.ModelCheckpoint(str(MODEL_OUT), save_best_only=True,
                                       monitor="val_loss", mode="min", verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                     restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2,
                                         patience=1, min_lr=1e-6, verbose=1),
    tf.keras.callbacks.CSVLogger(str(CSV_OUT), append=True),
    LrLogger(), t2,
]
model.fit(train_ds, validation_data=val_ds, epochs=FT_EPOCHS, callbacks=cbs2, verbose=2)

# Final val
val_loss, val_auc = model.evaluate(val_ds, verbose=0)

# -----------------------
# Meta & env
# -----------------------
meta = {
    "bundle": str(BUNDLE),
    "exp_id": "E05",
    "img_size": IMG_SIZE,
    "batch": BATCH,
    "warmup_epochs": WARMUP_EPOCHS,
    "finetune_epochs": FT_EPOCHS,
    "optimizer_phase1": f"adam@{LR_WARMUP}",
    "optimizer_phase2": f"adam@{LR_FINETUNE}_clipnorm1.0",
    "head": {"units": 256, "dropout": 0.5},
    "augmentation": "none",
    "policy": "Rule A (NaN->0,-1->0); CheXpert-5",
    "npz": NPZ_PATH,
    "n_train": int(len(Xtr)), "n_val": int(len(Xval)),
    "val_loss": float(val_loss), "val_auc": float(val_auc),
    "epoch_times_sec_phase1": t1.epoch_times,
    "epoch_times_sec_phase2": t2.epoch_times,
    "host": socket.gethostname(),
    "gpu": gpu_info(),
    "seed": SEED,
    "stamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "notes": "E05: E03 recipe with gentler FT LR (3e-5); no augmentation."
}
with open(META_OUT, "w") as f: json.dump(meta, f, indent=2)

with open(ENV_OUT, "w") as f:
    f.write(f"tensorflow: {tf.__version__}\n")
    try:
        import numpy as np, pandas as pd
        f.write(f"numpy: {np.__version__}\n")
        f.write(f"pandas: {pd.__version__}\n")
    except Exception:
        pass

print("\n[done] Bundle  →", BUNDLE)
print("      Model   →", MODEL_OUT)
print("      CSV     →", CSV_OUT)
print("      Meta    →", META_OUT)
