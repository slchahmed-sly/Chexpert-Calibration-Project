#!/usr/bin/env python3
# E06 — E03 recipe + CosineDecay FT LR and AdamW (fallback to Adam). No aug.

import os, json, time, socket, subprocess, math
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
LR_START  = 5e-5     # cosine start
LR_FLOOR  = 1e-6     # cosine floor approx via alpha
WD        = 1e-4     # AdamW weight decay

NPZ_PATH = os.path.expanduser("~/chexpert_project/data/processed/npz/chexpert_ruleA_30000.npz")
OUT_DIR  = Path.home() / "chexpert_project" / "outputs"
BUNDLE   = OUT_DIR / "bundles" / "model_E06"

MODEL_OUT = BUNDLE / "model_E06.keras"
CSV_OUT   = BUNDLE / "train_log.csv"
META_OUT  = BUNDLE / "run_E06_meta.json"
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
    def __init__(self, steps_per_epoch=None): super().__init__(); self.spe = steps_per_epoch
    def on_epoch_end(self, epoch, logs=None):
        if logs is None: return
        lr = self.model.optimizer.learning_rate
        try:
            # handle constant LR or schedule
            if hasattr(lr, '__call__'):
                step = tf.cast(self.model.optimizer.iterations, tf.float32)
                logs["learning_rate"] = float(tf.keras.backend.get_value(lr(step)))
            else:
                logs["learning_rate"] = float(tf.keras.backend.get_value(lr))
        except Exception:
            pass

def try_adamw(learning_rate, weight_decay, clipnorm=1.0):
    # Prefer built-in AdamW if present; otherwise fallback to Adam
    try:
        return tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay, clipnorm=clipnorm)
    except Exception:
        try:
            # Older Keras may expose legacy
            return tf.keras.optimizers.legacy.AdamW(learning_rate=learning_rate, weight_decay=weight_decay, clipnorm=clipnorm)
        except Exception:
            return tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=clipnorm)

# -----------------------
# Data
# -----------------------
data = np.load(NPZ_PATH)
X = data["X"]                              # uint8
Y = data["Y"].astype("float32")            # (N,5)

Xtr, Xval, Ytr, Yval = train_test_split(X, Y, test_size=0.2, random_state=SEED, shuffle=True)

AUTOTUNE = tf.data.AUTOTUNE
def norm(x,y): return tf.cast(x, tf.float32)/255.0, y

train_ds = (tf.data.Dataset.from_tensor_slices((Xtr,Ytr))
            .shuffle(min(8192,len(Xtr)), seed=SEED, reshuffle_each_iteration=True)
            .map(norm, num_parallel_calls=AUTOTUNE)
            .batch(BATCH).prefetch(AUTOTUNE))
val_ds   = (tf.data.Dataset.from_tensor_slices((Xval,Yval))
            .map(norm, num_parallel_calls=AUTOTUNE)
            .batch(BATCH).prefetch(AUTOTUNE))

steps_per_epoch = math.ceil(len(Xtr) / BATCH)

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

# Phase 1 — warmup head only (constant LR)
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
    LrLogger(steps_per_epoch), t1,
]
print("[E06] Warmup head-only for", WARMUP_EPOCHS, "epochs @", LR_WARMUP)
model.fit(train_ds, validation_data=val_ds, epochs=WARMUP_EPOCHS, callbacks=cbs1, verbose=2)

# Phase 2 — unfreeze ALL, AdamW + CosineDecay
for l in base.layers: l.trainable = True
alpha = LR_FLOOR / LR_START  # final lr fraction
decay_steps = steps_per_epoch * FT_EPOCHS
schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=LR_START,
    decay_steps=decay_steps,
    alpha=alpha  # final lr = alpha * initial
)
opt2 = try_adamw(learning_rate=schedule, weight_decay=WD, clipnorm=1.0)

model.compile(optimizer=opt2, loss="binary_crossentropy",
              metrics=[AUC(multi_label=True, num_labels=NUM_LABELS, name="AUC")])

t2 = TimeHistory()
cbs2 = [
    tf.keras.callbacks.ModelCheckpoint(str(MODEL_OUT), save_best_only=True,
                                       monitor="val_loss", mode="min", verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                     restore_best_weights=True, verbose=1),
    # Note: ReduceLROnPlateau isn’t used with an explicit schedule.
    tf.keras.callbacks.CSVLogger(str(CSV_OUT), append=True),
    LrLogger(steps_per_epoch), t2,
]
print(f"[E06] Full FT with AdamW (wd={WD}) + CosineDecay({LR_START}→~{LR_FLOOR}) for {FT_EPOCHS} epochs")
model.fit(train_ds, validation_data=val_ds, epochs=FT_EPOCHS, callbacks=cbs2, verbose=2)

# Final val
val_loss, val_auc = model.evaluate(val_ds, verbose=0)

# -----------------------
# Meta & env
# -----------------------
meta = {
    "bundle": str(BUNDLE),
    "exp_id": "E06",
    "img_size": IMG_SIZE,
    "batch": BATCH,
    "warmup_epochs": WARMUP_EPOCHS,
    "finetune_epochs": FT_EPOCHS,
    "optimizer_phase1": f"adam@{LR_WARMUP}",
    "optimizer_phase2": f"adamw@cosine({LR_START}->{LR_FLOOR})_clipnorm1.0_wd{WD}",
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
    "steps_per_epoch": steps_per_epoch,
    "cosine_decay_steps": decay_steps,
    "stamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "notes": "E06: full FT with AdamW + cosine decay; no augmentation."
}
BUNDLE.mkdir(parents=True, exist_ok=True)
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
