#!/usr/bin/env python3
# E08 — Same recipe as E07A, but AdamW + Cosine decay (with 1-epoch warmup) for fine-tuning.

import os, json, time, socket, subprocess, math
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
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

LR_WARMUP_HEAD = 1e-4     
BASE_LR_FT = 1e-4          
WEIGHT_DECAY = 1e-4

EXP_ID = "E08"
NPZ_PATH = os.path.expanduser("~/chexpert_project/data/processed/npz/chexpert_ruleA_30000.npz")
OUT_DIR  = Path.home() / "chexpert_project" / "outputs"
BUNDLE   = OUT_DIR / "bundles" / f"model_{EXP_ID}"
MODEL_OUT = BUNDLE / "model_best.keras"
CKPT_WEIGHTS = BUNDLE / "tmp_best.weights.h5"
CSV_OUT   = BUNDLE / "train_log.csv"
META_OUT  = BUNDLE / f"run_{EXP_ID}_meta.json"
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

class LrEpochLogger(tf.keras.callbacks.Callback):
    """Log the epoch-end LR when using a custom schedule."""
    def __init__(self, schedule, steps_per_epoch):
        super().__init__()
        self.schedule = schedule
        self.spe = steps_per_epoch
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            step = (epoch+1)*self.spe - 1
            lr_t = self.schedule(step)
            lr = float(tf.keras.backend.get_value(lr_t))
            logs["learning_rate"] = lr

# -----------------------
# Data
# -----------------------
data = np.load(NPZ_PATH)
X = data["X"]                              # uint8 (N,H,W,3)
Y = data["Y"].astype("float32")            # (N,5)

Xtr, Xval, Ytr, Yval = train_test_split(
    X, Y, test_size=0.2, random_state=SEED, shuffle=True
)
steps_per_epoch_ft = int(math.ceil(len(Xtr)/BATCH))

AUTOTUNE = tf.data.AUTOTUNE
def prep(x,y):
    x = tf.cast(x, tf.float32)           
    x = preprocess_input(x)               
    return x, y

train_ds = (tf.data.Dataset.from_tensor_slices((Xtr,Ytr))
            .shuffle(min(8192,len(Xtr)), seed=SEED, reshuffle_each_iteration=True)
            .map(prep, num_parallel_calls=AUTOTUNE)
            .batch(BATCH).prefetch(AUTOTUNE))

val_ds   = (tf.data.Dataset.from_tensor_slices((Xval,Yval))
            .map(prep, num_parallel_calls=AUTOTUNE)
            .batch(BATCH).prefetch(AUTOTUNE))

# -----------------------
# Bundle
# -----------------------
BUNDLE.mkdir(parents=True, exist_ok=True)

# -----------------------
# Model
# -----------------------
base = DenseNet121(include_top=False, weights="imagenet", input_shape=(*IMG_SIZE,3))
for l in base.layers: l.trainable = False   # freeze for warmup

inp = layers.Input(shape=(*IMG_SIZE,3))
x = base(inp)                               # BN stays active after unfreeze
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
out = layers.Dense(NUM_LABELS, activation="sigmoid")(x)
model = models.Model(inp, out)

macro_auc = AUC(multi_label=True, num_labels=NUM_LABELS, name="macro_AUC")

# Phase 1 — head-only warmup (constant LR)
opt_head = tf.keras.optimizers.Adam(learning_rate=LR_WARMUP_HEAD)
model.compile(optimizer=opt_head, loss="binary_crossentropy", metrics=[macro_auc])

t1 = TimeHistory()
cbs1 = [
      tf.keras.callbacks.ModelCheckpoint(
    filepath=str(CKPT_WEIGHTS),
    save_best_only=True,
    save_weights_only=True,      # <-- key change
    monitor="val_loss",
    mode="min",
    verbose=1,
),
    tf.keras.callbacks.EarlyStopping(monitor="val_macro_AUC",
        patience=PATIENCE, mode="max", restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_macro_AUC",
        mode="max", factor=0.2, patience=1, min_lr=1e-6, verbose=1),
    tf.keras.callbacks.CSVLogger(str(CSV_OUT), append=False),
]

print(f"[{EXP_ID}] Warmup {WARMUP_EPOCHS} epoch(s), head-only @ {LR_WARMUP_HEAD} ...")
model.fit(train_ds, validation_data=val_ds, epochs=WARMUP_EPOCHS, callbacks=cbs1, verbose=1)

# -----------------------
# Phase 2 — unfreeze ALL + AdamW + Warmup→Cosine
# -----------------------
for l in base.layers: l.trainable = True

# AdamW import (compat)
try:
    from tensorflow.keras.optimizers import AdamW as KAdamW
except Exception:
    from tensorflow.keras.optimizers.experimental import AdamW as KAdamW

class WarmupThenCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup_steps, total_steps):
        super().__init__()
        self.base_lr = tf.convert_to_tensor(base_lr, dtype=tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.total_steps = tf.cast(total_steps, tf.float32)
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        # Linear warmup
        lr = tf.cond(
            step < self.warmup_steps,
            lambda: self.base_lr * (step / tf.maximum(1.0, self.warmup_steps)),
            # Cosine decay to 0
            lambda: self.base_lr * 0.5 * (1.0 + tf.cos(
                math.pi * (step - self.warmup_steps) / tf.maximum(1.0, (self.total_steps - self.warmup_steps))
            ))
        )
        return lr
    def get_config(self):
        return {"base_lr": float(self.base_lr.numpy()),
                "warmup_steps": int(self.warmup_steps.numpy()),
                "total_steps": int(self.total_steps.numpy())}

total_steps = steps_per_epoch_ft * FT_EPOCHS
warmup_steps = steps_per_epoch_ft * 1  # 1 epoch warmup inside FT
lr_schedule = WarmupThenCosine(BASE_LR_FT, warmup_steps, total_steps)

opt_ft = KAdamW(learning_rate=lr_schedule, weight_decay=WEIGHT_DECAY, clipnorm=1.0)
model.compile(optimizer=opt_ft, loss="binary_crossentropy", metrics=[macro_auc])

t2 = TimeHistory()
cbs2 = [
  tf.keras.callbacks.ModelCheckpoint(
    filepath=str(CKPT_WEIGHTS),
    save_best_only=True,
    save_weights_only=True,      # <-- key change
    monitor="val_loss",
    mode="min",
    verbose=1,
),
    tf.keras.callbacks.EarlyStopping(monitor="val_macro_AUC",
        patience=PATIENCE, mode="max", restore_best_weights=True, verbose=1),
    # NOTE: No ReduceLROnPlateau here (would fight with the schedule)
    tf.keras.callbacks.CSVLogger(str(CSV_OUT), append=True),
    LrEpochLogger(schedule=lr_schedule, steps_per_epoch=steps_per_epoch_ft),
]

print(f"[{EXP_ID}] Unfreezing ALL; AdamW(wd={WEIGHT_DECAY}) with Warmup→Cosine (base LR={BASE_LR_FT}) ...")
model.fit(train_ds, validation_data=val_ds, epochs=FT_EPOCHS, callbacks=cbs2, verbose=1)


val_metrics = model.evaluate(val_ds, verbose=0)
val_loss = float(val_metrics[0]); val_auc  = float(val_metrics[1])

# -----------------------
# Meta & env
# -----------------------
BUNDLE.mkdir(parents=True, exist_ok=True)
meta = {
    "bundle": str(BUNDLE),
    "exp_id": EXP_ID,
    "img_size": [IMG_SIZE[0], IMG_SIZE[1], 3],
    "batch": BATCH,
    "warmup_epochs": WARMUP_EPOCHS,
    "finetune_epochs": FT_EPOCHS,
    "preprocessing": "densenet_preprocess_input",
    "label_policy": "RuleA (NaN->0, U->0)",
    "optimizer": "adamw",
    "weight_decay": WEIGHT_DECAY,
    "lr_schedule": "warmup_then_cosine",
    "base_lr_ft": BASE_LR_FT,
    "bn_trainable_after_unfreeze": True,
    "selection_metric": "val_macro_AUC",
    "npz": NPZ_PATH,
    "n_train": int(len(Xtr)), "n_val": int(len(Xval)),
    "val_loss": val_loss, "val_macro_auc": val_auc,
    "epoch_times_sec_phase1": t1.epoch_times,
    "epoch_times_sec_phase2": t2.epoch_times,
    "host": socket.gethostname(),
    "gpu": gpu_info(),
    "seed": SEED,
    "stamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "notes": "E08: E07A + AdamW(wd=1e-4) and Warmup→Cosine FT; select by val_macro_AUC."
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
