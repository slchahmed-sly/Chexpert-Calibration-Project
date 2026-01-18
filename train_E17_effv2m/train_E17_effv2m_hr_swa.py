#!/usr/bin/env python3
# E17 — EfficientNetV2-M @384, Multi-study + Mask-U, AdamW + Warmup→Cosine

import os, json, time, socket, subprocess, math
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# ----------------------- Config -----------------------
SEED = 42
IMG_SIZE = (384, 384)          
BATCH = 6                    
WARMUP_EPOCHS = 3
FT_EPOCHS = 10
PATIENCE = 2
NUM_LABELS = 5

LR_WARMUP_HEAD = 1e-4
BASE_LR_FT = 1e-4
WEIGHT_DECAY = 1e-4


SWA_START_EPOCH = 3              

EXP_ID = f"E17_effv2m_hr_{IMG_SIZE[0]}"

NPZ_PATH = os.path.expanduser("~/chexpert_project/data/processed/npz/chexpert_maskU_multiK3_384_60000.npz")

OUT_DIR  = Path.home() / "chexpert_project" / "outputs"
BUNDLE   = OUT_DIR / "bundles" / f"model_{EXP_ID}"
MODEL_OUT_SWA = BUNDLE / "model_best.keras"             
MODEL_OUT_BEST = BUNDLE / "model_best_by_val.keras"    
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

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

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


class SWACallback(tf.keras.callbacks.Callback):
    def __init__(self, start_epoch_global):
        super().__init__()
        self.start_epoch_global = start_epoch_global
        self.started = False
        self.n = 0
        self.avg = None 
    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.start_epoch_global:
            return

        ws = [w.numpy() for w in self.model.weights]
        if not self.started:
            self.avg = ws
            self.n = 1
            self.started = True
        else:
            self.n += 1
            for i in range(len(self.avg)):
                self.avg[i] += (ws[i] - self.avg[i]) / self.n
    def apply_swa(self):
        if self.started and self.avg is not None:
            for w, a in zip(self.model.weights, self.avg):
                w.assign(a)


@tf.function
def bce_elementwise(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    return -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))

from tensorflow.keras.metrics import AUC
class AUCIgnoreWeights(AUC):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true, y_pred, sample_weight=None)

# ----------------------- Data: streaming NPZ -----------------------
z = np.load(NPZ_PATH, mmap_mode="r")
X_mm = z["X"]                        
Y_mm = z["Y"].astype("float32")    
M_mm = z["M"].astype("float32")     
P_mm = z["P"]                   
N = X_mm.shape[0]
H, W = IMG_SIZE
AUTOTUNE = tf.data.AUTOTUNE


unique_p = np.unique(P_mm)
rng = np.random.default_rng(SEED)
rng.shuffle(unique_p)
n_val_p = int(0.2 * len(unique_p))
val_p = set(unique_p[:n_val_p].tolist())

idx_all = np.arange(N, dtype=np.int64)
train_mask = np.array([p not in val_p for p in P_mm], dtype=bool)
val_mask   = ~train_mask
train_idx = idx_all[train_mask]
val_idx   = idx_all[val_mask]


def _geom_jitter(x01):
    s = tf.random.uniform([], 0.9, 1.1)  # ±10% scale
    nh = tf.cast(tf.round(s * tf.cast(H, tf.float32)), tf.int32)
    nw = tf.cast(tf.round(s * tf.cast(W, tf.float32)), tf.int32)
    x = tf.image.resize(x01, (nh, nw))
    pad_h = tf.cast(tf.round(0.05 * tf.cast(H, tf.float32)), tf.int32)
    pad_w = tf.cast(tf.round(0.05 * tf.cast(W, tf.float32)), tf.int32)
    x = tf.image.resize_with_crop_or_pad(x, H + 2*pad_h, W + 2*pad_w)
    x = tf.image.random_crop(x, (H, W, 3))
    return x

def _fetch_py(i):
    i = int(i)
    x = X_mm[i]             
    y = Y_mm[i].astype("float32")
    m = M_mm[i].astype("float32")
    return x, y, m

def _fetch(i):
    x, y, m = tf.py_function(_fetch_py, [i], [tf.uint8, tf.float32, tf.float32])
    x.set_shape((H, W, 3)); y.set_shape((NUM_LABELS,)); m.set_shape((NUM_LABELS,))
    return x, y, m

from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as effv2_preprocess

def _map_train(x, y, m):
    x = tf.cast(x, tf.float32) / 255.0
    x = _geom_jitter(x)
    x = x * 255.0
    x = effv2_preprocess(x)
    return x, y, m

def _map_val(x, y, m):
    x = tf.cast(x, tf.float32)
    x = effv2_preprocess(x)
    return x, y, m

def make_ds(idxs, training=False, batch_size=BATCH):
    ds = tf.data.Dataset.from_tensor_slices(idxs)
    if training:
        ds = ds.shuffle(buffer_size=min(16384, len(idxs)), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(_fetch, num_parallel_calls=AUTOTUNE)
    ds = ds.map(_map_train if training else _map_val, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds

train_ds = make_ds(train_idx, training=True)
val_ds   = make_ds(val_idx, training=False)
steps_per_epoch_ft = int(np.ceil(len(train_idx)/BATCH))

# ----------------------- Model -----------------------
try:
    from tensorflow.keras.applications import EfficientNetV2M
except Exception:
    from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2M

def build_effv2m_head(img_size=(H,W), num_labels=NUM_LABELS, dense_units=256, dropout=0.5):
    inp = layers.Input(shape=(img_size[0], img_size[1], 3))
    base = EfficientNetV2M(include_top=False, weights="imagenet", input_shape=(img_size[0], img_size[1], 3))
    x = base(inp)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(num_labels, activation="sigmoid", dtype="float32")(x)
    model = models.Model(inp, out)
    return model, base

model, base = build_effv2m_head(IMG_SIZE, NUM_LABELS, dense_units=256, dropout=0.5)
macro_auc = AUCIgnoreWeights(multi_label=True, num_labels=NUM_LABELS, name="macro_AUC")

# ----------------------- Phase 1: warmup (head-only) -----------------------
for l in base.layers: l.trainable = False
opt_head = tf.keras.optimizers.Adam(learning_rate=LR_WARMUP_HEAD)
model.compile(optimizer=opt_head, loss=bce_elementwise, metrics=[macro_auc])

BUNDLE.mkdir(parents=True, exist_ok=True)
t1 = TimeHistory()
cbs1 = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(CKPT_WEIGHTS),
        save_best_only=True,
        save_weights_only=True,
        monitor="val_macro_AUC",
        mode="max",
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(monitor="val_macro_AUC",
        patience=PATIENCE, mode="max", restore_best_weights=True, verbose=1),
    tf.keras.callbacks.CSVLogger(str(CSV_OUT), append=False),
    t1,
]
print(f"[{EXP_ID}] Warmup {WARMUP_EPOCHS} epoch(s) @ {LR_WARMUP_HEAD} ...")
model.fit(train_ds, validation_data=val_ds, epochs=WARMUP_EPOCHS, callbacks=cbs1, verbose=1)

# ----------------------- Phase 2: unfreeze + AdamW + Warmup→Cosine + SWA -----------------------
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
model.compile(optimizer=opt_ft, loss=bce_elementwise, metrics=[macro_auc])


swa_cb = SWACallback(start_epoch_global=WARMUP_EPOCHS + SWA_START_EPOCH)

t2 = TimeHistory()
cbs2 = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(CKPT_WEIGHTS),
        save_best_only=True,
        save_weights_only=True,
        monitor="val_macro_AUC",
        mode="max",
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(monitor="val_macro_AUC",
        patience=PATIENCE, mode="max", restore_best_weights=True, verbose=1),
    tf.keras.callbacks.CSVLogger(str(CSV_OUT), append=True),
    LrEpochLogger(schedule=lr_schedule, steps_per_epoch=steps_per_epoch_ft),
    swa_cb,
    t2,
]
print(f"[{EXP_ID}] Unfreeze ALL; AdamW(wd={WEIGHT_DECAY}) + Warmup→Cosine(base LR={BASE_LR_FT}) + SWA(start@FT{SWA_START_EPOCH}) ...")
model.fit(train_ds, validation_data=val_ds, epochs=WARMUP_EPOCHS+FT_EPOCHS, callbacks=cbs2, verbose=1)


BUNDLE.mkdir(parents=True, exist_ok=True)


try:
    from tensorflow.keras.applications import EfficientNetV2M
except Exception:
    from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2M

def _build_same():
    inp = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    base = EfficientNetV2M(include_top=False, weights="imagenet", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base(inp)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(NUM_LABELS, activation="sigmoid", dtype="float32")(x)
    return models.Model(inp, out)

best_model = _build_same()
best_model.load_weights(str(CKPT_WEIGHTS))
best_model.save(MODEL_OUT_BEST, include_optimizer=False)

swa_cb.apply_swa()
model.save(MODEL_OUT_SWA, include_optimizer=False)


val_metrics = model.evaluate(val_ds, verbose=0)
val_loss = float(val_metrics[0]); val_auc = float(val_metrics[1])

# ----------------------- Meta & env -----------------------
meta = {
    "bundle": str(BUNDLE),
    "exp_id": EXP_ID,
    "backbone": "EfficientNetV2M",
    "img_size": [IMG_SIZE[0], IMG_SIZE[1], 3],
    "batch": BATCH,
    "warmup_epochs": WARMUP_EPOCHS,
    "finetune_epochs": FT_EPOCHS,
    "preprocessing": "efficientnetv2_preprocess_input (tf.data)",
    "augmentation": {
        "rotation_deg": "none",
        "translation": "±5% (pad+crop)",
        "zoom": "±10% via resize+pad+crop",
        "flip": "none",
        "photometric": "none"
    },
    "label_policy": "MaskU (U ignored via per-label sample_weight mask)",
    "optimizer": "adamw",
    "weight_decay": WEIGHT_DECAY,
    "lr_schedule": "warmup_then_cosine",
    "base_lr_ft": BASE_LR_FT,
    "swa": {"enabled": True, "start_epoch_global": int(WARMUP_EPOCHS + SWA_START_EPOCH)},
    "bn_trainable_after_unfreeze": True,
    "selection_metric": "val_macro_AUC",
    "npz": NPZ_PATH,
    "n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
    "val_loss_swa": val_loss, "val_macro_auc_swa": val_auc,
    "host": socket.gethostname(),
    "gpu": gpu_info(),
    "seed": SEED,
    "stamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "notes": "E17: EffNetV2-M @384; best-by-val and SWA models both saved."
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
print("      Best-by-val →", MODEL_OUT_BEST)
print("      SWA         →", MODEL_OUT_SWA)
print("      CSV         →", CSV_OUT)
print("      Meta        →", META_OUT)
