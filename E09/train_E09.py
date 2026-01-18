#!/usr/bin/env python3
# E09 — Mask-U (~43k), AdamW + Warmup→Cosine

import os, json, time, socket, subprocess, math
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.metrics import AUC
from sklearn.model_selection import train_test_split

# ----------------------- Config -----------------------
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

EXP_ID = "E09"
NPZ_PATH = os.path.expanduser("~/chexpert_project/data/processed/npz/chexpert_maskU_224_43000.npz")

OUT_DIR  = Path.home() / "chexpert_project" / "outputs"
BUNDLE   = OUT_DIR / "bundles" / f"model_{EXP_ID}"
MODEL_OUT = BUNDLE / "model_best.keras"
CKPT_WEIGHTS = BUNDLE / "tmp_best.weights.h5"
CSV_OUT   = BUNDLE / "train_log.csv"
META_OUT  = BUNDLE / f"run_{EXP_ID}_meta.json"
ENV_OUT   = BUNDLE / "env.txt"

# ----------------------- System niceties -----------------------
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
    def __init__(self, schedule, steps_per_epoch): super().__init__(); self.sch=schedule; self.spe=steps_per_epoch
    def on_epoch_end(self, epoch, logs=None):
        step = (epoch+1)*self.spe - 1
        lr = tf.keras.backend.get_value(self.sch(step))
        if logs is not None: logs["learning_rate"] = float(lr)

# AUC metric that *ignores* sample_weight (to avoid shape issues).
class AUCIgnoreWeights(AUC):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true, y_pred, sample_weight=None)

# Elementwise BCE (no reduction); returns shape (batch, 5)
@tf.function
def bce_elementwise(y_true, y_pred):
    # Keras backend BCE returns mean over last axis;
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    return -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))

# ----------------------- Data (Mask-U) -----------------------
data = np.load(NPZ_PATH)
X = data["X"]                              
Y = data["Y"].astype("float32")            
M = data["M"].astype("float32")          

Xtr, Xval, Ytr, Yval, Mtr, Mval = train_test_split(
    X, Y, M, test_size=0.2, random_state=SEED, shuffle=True
)

AUTOTUNE = tf.data.AUTOTUNE
def prep(x,y,m):
    x = tf.cast(x, tf.float32)
    x = preprocess_input(x)
    return x, y, m

train_ds = (tf.data.Dataset.from_tensor_slices((Xtr,Ytr,Mtr))
            .shuffle(min(8192,len(Xtr)), seed=SEED, reshuffle_each_iteration=True)
            .map(prep, num_parallel_calls=AUTOTUNE)
            .batch(BATCH).prefetch(AUTOTUNE))

val_ds   = (tf.data.Dataset.from_tensor_slices((Xval,Yval,Mval))
            .map(prep, num_parallel_calls=AUTOTUNE)
            .batch(BATCH).prefetch(AUTOTUNE))

steps_per_epoch_ft = int(math.ceil(len(Xtr)/BATCH))

# ----------------------- Model -----------------------
base = DenseNet121(include_top=False, weights="imagenet", input_shape=(*IMG_SIZE,3))
for l in base.layers: l.trainable = False

inp = layers.Input(shape=(*IMG_SIZE,3))
x = base(inp)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
out = layers.Dense(NUM_LABELS, activation="sigmoid")(x)
model = models.Model(inp, out)

auc_metric = AUCIgnoreWeights(multi_label=True, num_labels=NUM_LABELS, name="AUC")

# ----------------------- Phase 1 — head-only -----------------------
opt_head = tf.keras.optimizers.Adam(learning_rate=LR_WARMUP_HEAD)
model.compile(
    optimizer=opt_head,
    loss=bce_elementwise,     # <-- elementwise loss (batch,5)
    metrics=[auc_metric],
    run_eagerly=False,
)

BUNDLE.mkdir(parents=True, exist_ok=True)
t1 = TimeHistory()
cbs1 = [
    tf.keras.callbacks.ModelCheckpoint(
    filepath=str(CKPT_WEIGHTS),
    save_best_only=True,
    save_weights_only=True,      # <-- key change
    monitor="val_loss",
    mode="min",
    verbose=1,
),           # monitor val_loss (masked)
    tf.keras.callbacks.EarlyStopping(monitor="val_loss",
        patience=PATIENCE, mode="min", restore_best_weights=True, verbose=1),
    tf.keras.callbacks.CSVLogger(str(CSV_OUT), append=False),
    t1,
]
print(f"[{EXP_ID}] Warmup {WARMUP_EPOCHS} epoch(s), head-only @ {LR_WARMUP_HEAD} ...")
# Keras will pick sample_weight automatically from the 3rd dataset element (M)
model.fit(train_ds, validation_data=val_ds, epochs=WARMUP_EPOCHS, callbacks=cbs1, verbose=1)

# ----------------------- Phase 2 — unfreeze + AdamW + Warmup→Cosine -----------------------
for l in base.layers: l.trainable = True
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
        return tf.cond(
            step < self.warmup_steps,
            lambda: self.base_lr * (step / tf.maximum(1.0, self.warmup_steps)),
            lambda: self.base_lr * 0.5 * (1.0 + tf.cos(
                math.pi * (step - self.warmup_steps) / tf.maximum(1.0, (self.total_steps - self.warmup_steps))
            ))
        )
    def get_config(self):
        return {"base_lr": float(self.base_lr.numpy()),
                "warmup_steps": int(self.warmup_steps.numpy()),
                "total_steps": int(self.total_steps.numpy())}

total_steps = steps_per_epoch_ft * FT_EPOCHS
warmup_steps = steps_per_epoch_ft * 1
lr_schedule = WarmupThenCosine(BASE_LR_FT, warmup_steps, total_steps)

opt_ft = KAdamW(learning_rate=lr_schedule, weight_decay=WEIGHT_DECAY, clipnorm=1.0)
model.compile(
    optimizer=opt_ft,
    loss=bce_elementwise,     # elementwise loss keeps mask working
    metrics=[auc_metric],
    run_eagerly=False,
)

t2 = TimeHistory()
cbs2 = [
    tf.keras.callbacks.ModelCheckpoint(
    filepath=str(CKPT_WEIGHTS),
    save_best_only=True,
    save_weights_only=True,      # <-- key change
    monitor="val_loss",
    mode="min",
    verbose=1,
),          # keep monitoring val_loss
    tf.keras.callbacks.EarlyStopping(monitor="val_loss",
        patience=PATIENCE, mode="min", restore_best_weights=True, verbose=1),
    tf.keras.callbacks.CSVLogger(str(CSV_OUT), append=True),
    LrEpochLogger(schedule=lr_schedule, steps_per_epoch=steps_per_epoch_ft),
    t2,
]
print(f"[{EXP_ID}] Unfreezing ALL; AdamW(wd={WEIGHT_DECAY}) with Warmup→Cosine (base LR={BASE_LR_FT}) ...")
model.fit(train_ds, validation_data=val_ds, epochs=FT_EPOCHS, callbacks=cbs2, verbose=1)

# ----------------------- Final val & meta -----------------------
val_loss, val_auc = model.evaluate(val_ds, verbose=0)

meta = {
    "bundle": str(BUNDLE),
    "exp_id": EXP_ID,
    "img_size": [IMG_SIZE[0], IMG_SIZE[1], 3],
    "batch": BATCH,
    "warmup_epochs": WARMUP_EPOCHS,
    "finetune_epochs": FT_EPOCHS,
    "preprocessing": "densenet_preprocess_input",
    "label_policy": "MaskU (NaN->0, U ignored via per-label sample_weight mask)",
    "optimizer": "adamw",
    "weight_decay": WEIGHT_DECAY,
    "lr_schedule": "warmup_then_cosine",
    "base_lr_ft": BASE_LR_FT,
    "bn_trainable_after_unfreeze": True,
    "selection_metric": "val_loss",
    "npz": NPZ_PATH,
    "n_train": int(len(Xtr)), "n_val": int(len(Xval)),
    "val_loss": float(val_loss), "val_auc_unmasked": float(val_auc),  # AUC logged but unmasked
    "epoch_times_sec_phase1": getattr(t1, "epoch_times", []),
    "epoch_times_sec_phase2": getattr(t2, "epoch_times", []),
    "host": socket.gethostname(),
    "gpu": gpu_info(),
    "seed": SEED,
    "stamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "notes": "E09: Mask-U + AdamW(wd=1e-4) Warmup→Cosine; elementwise BCE with per-label mask; checkpoint on val_loss."
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
