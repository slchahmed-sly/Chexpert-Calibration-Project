#!/usr/bin/env python3
# E02 — Warmup 2 epochs head-only, then unfreeze last 50 w/ fresh optimizer.

import os, json, time, socket, subprocess
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.metrics import AUC
from sklearn.model_selection import train_test_split

SEED = 42
IMG_SIZE = (224,224)
BATCH = 32
WARMUP_EPOCHS = 2
FT_EPOCHS = 8
NUM_LABELS = 5
UNFREEZE_LAST = 50
EXP_ID = "E02"
BACKBONE = "densenet121"
NPZ_PATH = os.path.expanduser("~/chexpert_project/data/processed/npz/chexpert_ruleA_30000.npz")
OUT_DIR  = Path.home() / "chexpert_project" / "outputs"

class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None): self.epoch_times=[]
    def on_epoch_begin(self, epoch, logs=None): self._t=time.time()
    def on_epoch_end(self, epoch, logs=None): self.epoch_times.append(time.time()-self._t)

def gpu_info():
    try:
        q = subprocess.check_output(
            ["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"], text=True
        ).strip().splitlines()
        return q[0] if q else "unknown"
    except Exception:
        return "unknown"

for g in tf.config.list_physical_devices("GPU"):
    try: tf.config.experimental.set_memory_growth(g, True)
    except: pass
tf.random.set_seed(SEED); np.random.seed(SEED)

data = np.load(NPZ_PATH)
X = data["X"]; Y = data["Y"].astype("float32")
Xtr, Xval, Ytr, Yval = train_test_split(X,Y,test_size=0.2,random_state=SEED,shuffle=True)

AUTOTUNE = tf.data.AUTOTUNE
def norm(x,y): return tf.cast(x,tf.float32)/255.0, y
train_ds = (tf.data.Dataset.from_tensor_slices((Xtr,Ytr))
            .shuffle(min(8192,len(Xtr)), seed=SEED, reshuffle_each_iteration=True)
            .map(norm, num_parallel_calls=AUTOTUNE).batch(BATCH).prefetch(AUTOTUNE))
val_ds   = (tf.data.Dataset.from_tensor_slices((Xval,Yval))
            .map(norm, num_parallel_calls=AUTOTUNE).batch(BATCH).prefetch(AUTOTUNE))

stamp = time.strftime("%Y%m%d_%H%M%S")
bundle_name = f"{BACKBONE}_{EXP_ID}_{IMG_SIZE[0]}x{IMG_SIZE[1]}_{stamp}"
BUNDLE = OUT_DIR / "bundles" / bundle_name
BUNDLE.mkdir(parents=True, exist_ok=True)
MODEL_OUT = BUNDLE / "model_best.keras"
CSV_OUT   = BUNDLE / "train_log.csv"
META_OUT  = BUNDLE / "run_meta.json"
ENV_OUT   = BUNDLE / "env.txt"

base = DenseNet121(include_top=False, weights="imagenet", input_shape=(*IMG_SIZE,3))
for l in base.layers: l.trainable = False

inp = layers.Input(shape=(*IMG_SIZE,3))
x = base(inp, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
out = layers.Dense(NUM_LABELS, activation="sigmoid")(x)
model = models.Model(inp, out)

# Phase 1: head-only
opt1 = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=opt1, loss="binary_crossentropy",
              metrics=[AUC(multi_label=True, num_labels=NUM_LABELS, name="AUC")])

time_cb1 = TimeHistory()
cbs1 = [
    tf.keras.callbacks.ModelCheckpoint(str(MODEL_OUT), save_best_only=True,
                                       monitor="val_loss", mode="min", verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2,
                                     restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2,
                                         patience=1, min_lr=1e-6, verbose=1),
    tf.keras.callbacks.CSVLogger(str(CSV_OUT), append=False),
    time_cb1,
]
print(f"[E02] Warmup {WARMUP_EPOCHS} epoch(s), head-only...")
model.fit(train_ds, validation_data=val_ds, epochs=WARMUP_EPOCHS, callbacks=cbs1, verbose=2)

# Phase 2: unfreeze last 50 + fresh optimizer
print(f"[E02] Unfreezing last {UNFREEZE_LAST} layers and lr=5e-5")
for l in base.layers[:-UNFREEZE_LAST]: l.trainable = False
for l in base.layers[-UNFREEZE_LAST:]: l.trainable = True

opt2 = tf.keras.optimizers.Adam(learning_rate=5e-5)  # fresh optimizer!
model.compile(optimizer=opt2, loss="binary_crossentropy",
              metrics=[AUC(multi_label=True, num_labels=NUM_LABELS, name="AUC")])

time_cb2 = TimeHistory()
cbs2 = [
    tf.keras.callbacks.ModelCheckpoint(str(MODEL_OUT), save_best_only=True,
                                       monitor="val_loss", mode="min", verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2,
                                     restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2,
                                         patience=1, min_lr=1e-6, verbose=1),
    tf.keras.callbacks.CSVLogger(str(CSV_OUT), append=True),  # append to the same CSV
    time_cb2,
]
model.fit(train_ds, validation_data=val_ds, epochs=FT_EPOCHS, callbacks=cbs2, verbose=2)

val_loss, val_auc = model.evaluate(val_ds, verbose=0)

meta = {
    "bundle": str(BUNDLE),
    "exp_id": EXP_ID,
    "backbone": BACKBONE,
    "img_size": IMG_SIZE,
    "batch": BATCH,
    "warmup_epochs": WARMUP_EPOCHS,
    "finetune_epochs": FT_EPOCHS,
    "unfreeze_last": UNFREEZE_LAST,
    "optimizers": {"phase1":"adam@1e-4","phase2":"adam@5e-5"},
    "policy": "Rule A (NaN->0,-1->0); CheXpert-5",
    "npz": NPZ_PATH,
    "n_train": int(len(Xtr)), "n_val": int(len(Xval)),
    "val_loss": float(val_loss), "val_auc": float(val_auc),
    "epoch_times_sec_phase1": time_cb1.epoch_times,
    "epoch_times_sec_phase2": time_cb2.epoch_times,
    "host": socket.gethostname(),
    "gpu": gpu_info(),
    "seed": SEED,
    "stamp": stamp,
    "notes": "E02: warmup→unfreeze_last50; CSV concatenates both phases."
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
