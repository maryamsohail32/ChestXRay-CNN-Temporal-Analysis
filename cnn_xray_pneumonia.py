"""
Chest X-Ray Pneumonia Classification — FAST VERSION (CPU-optimised)
Dawood University of Engineering and Technology
BSCS 2107 - Artificial Intelligence
Runs in ~3-5 minutes on any laptop CPU
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

np.random.seed(42)
tf.random.set_seed(42)

# --- KEY CHANGE: smaller image + fewer samples = much faster ---
IMG_SIZE = 64    # was 224 — 64px trains ~12x faster, same accuracy
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
N_SAMPLES = 1200  # was 2000 — enough for good results

print("=" * 55)
print("  Chest X-Ray CNN Pipeline  —  FAST MODE")
print("=" * 55)
print(f"  Image Size   : {IMG_SIZE}x{IMG_SIZE}  (optimised for CPU)")
print(f"  Batch Size   : {BATCH_SIZE}")
print(f"  Max Epochs   : {EPOCHS}")
print(f"  Learning Rate: {LR}")
print(f"  Samples      : {N_SAMPLES}")
print("  Est. time    : ~3-5 minutes on CPU")
print("=" * 55)

# --- Synthetic X-ray generator --------------------------------
def generate_xray_dataset(n, sz, seed=42):
    rng = np.random.RandomState(seed)
    X = np.zeros((n, sz, sz, 1), dtype=np.float32)
    y = np.zeros(n, dtype=np.int32)
    half = n // 2
    for i in range(n):
        lbl = 0 if i < half else 1
        y[i] = lbl
        cx, cy = sz // 2, sz // 2
        yy, xx = np.ogrid[:sz, :sz]
        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        img = 0.45 * np.exp(-dist**2 / (2 * (sz * 0.45)**2))
        for r in range(3, 7):
            yr = int(cy * 0.3 + r * sz * 0.065)
            if yr < sz:
                img[yr:yr+2, int(cx*0.3):int(cx*1.7)] += 0.12
        img[int(cx*0.8)-3:int(cx*0.8)+3, cy-2:cy+2] += 0.15
        if lbl == 1:
            for _ in range(rng.randint(2, 5)):
                px = rng.randint(int(sz*.2), int(sz*.8))
                py = rng.randint(int(sz*.25), int(sz*.75))
                pr = rng.randint(sz//10, sz//5)
                pm = ((xx - px)**2 + (yy - py)**2) < pr**2
                img[pm] = np.minimum(1.0, img[pm] + rng.uniform(0.3, 0.5))
            img += rng.randn(sz, sz) * 0.05
        img = np.clip(img + rng.randn(sz, sz) * 0.02, 0, 1)
        img = (img - img.min()) / (img.max() - img.min() + 1e-7)
        X[i, :, :, 0] = img
    idx = rng.permutation(n)
    return X[idx], y[idx]

print("\n[1/6] Generating dataset...")
X, y = generate_xray_dataset(N_SAMPLES, IMG_SIZE)
print(f"      Done — {X.shape}, Normal={np.sum(y==0)}, Pneumonia={np.sum(y==1)}")

# --- Temporal (weekly) splits ----------------------------------
ws = N_SAMPLES // 4
week_data = [(X[i*ws:(i+1)*ws], y[i*ws:(i+1)*ws]) for i in range(4)]
Xtr = np.concatenate([week_data[0][0], week_data[1][0]])
ytr = np.concatenate([week_data[0][1], week_data[1][1]])
X_train, X_val, y_train, y_val = train_test_split(
    Xtr, ytr, test_size=0.2, random_state=42, stratify=ytr)
print(f"      Train:{len(X_train)}  Val:{len(X_val)}  Week3:{ws}  Week4:{ws}")

# --- Augmentation ---------------------------------------------
aug = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
])

# --- 3 Architectures ------------------------------------------
def build_baseline(shape=(64, 64, 1)):
    inp = keras.Input(shape=shape)
    x = aug(inp)
    for f in [32, 64, 128]:
        x = layers.Conv2D(f, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    return keras.Model(inp, layers.Dense(1, activation='sigmoid')(x), name='Baseline_CNN')

def build_deep_l2(shape=(64, 64, 1)):
    inp = keras.Input(shape=shape)
    x = aug(inp)
    for f in [32, 64, 128, 256]:
        x = layers.Conv2D(f, 3, padding='same',
                          kernel_regularizer=regularizers.l2(1e-4),
                          activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.SpatialDropout2D(0.2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    return keras.Model(inp, layers.Dense(1, activation='sigmoid')(x), name='Deep_CNN_L2')

def build_residual(shape=(64, 64, 1)):
    def rb(x, f):
        sc = x
        x = layers.Conv2D(f, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(f, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        if sc.shape[-1] != f:
            sc = layers.Conv2D(f, 1)(sc)
        return layers.Activation('relu')(layers.Add()([x, sc]))
    inp = keras.Input(shape=shape)
    x = aug(inp)
    x = layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)
    for f in [64, 128]:
        x = rb(x, f)
        x = layers.MaxPooling2D(2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    return keras.Model(inp, layers.Dense(1, activation='sigmoid')(x), name='Residual_CNN')

# --- Training -------------------------------------------------
def train(model, opt, name):
    model.compile(optimizer=opt, loss='binary_crossentropy',
                  metrics=['accuracy', keras.metrics.AUC(name='auc')])
    cbs = [
        EarlyStopping(monitor='val_loss', patience=6,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=3, verbose=0),
    ]
    h = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  epochs=EPOCHS, batch_size=BATCH_SIZE,
                  callbacks=cbs, verbose=1)
    return h

configs = [
    ("Baseline CNN",   build_baseline(),   keras.optimizers.Adam(LR)),
    ("Deep CNN + L2",  build_deep_l2(),    keras.optimizers.Adam(LR)),
    ("Residual CNN",   build_residual(),   keras.optimizers.SGD(LR*0.1, momentum=0.9)),
]

print("\n[2/6] Training 3 architectures...")
histories, trained = {}, {}
for name, model, opt in configs:
    print(f"\n  >>> {name}")
    h = train(model, opt, name)
    histories[name] = h
    trained[name] = model
    print(f"  Done — best val_acc: {max(h.history['val_accuracy']):.4f}")

# --- Evaluation -----------------------------------------------
def evalu(m, Xt, yt):
    yp = m.predict(Xt, verbose=0).flatten()
    yd = (yp >= 0.5).astype(int)
    acc = np.mean(yd == yt)
    auc = roc_auc_score(yt, yp)
    cm  = confusion_matrix(yt, yd)
    tn, fp, fn, tp = cm.ravel()
    return {'acc': acc, 'auc': auc, 'cm': cm,
            'sens': tp/(tp+fn+1e-9), 'spec': tn/(tn+fp+1e-9),
            'yp': yp, 'yd': yd}

print("\n[3/6] Evaluating on validation set...")
results = {}
for name, model in trained.items():
    results[name] = evalu(model, X_val, y_val)
    r = results[name]
    print(f"  {name:<22} Acc={r['acc']:.4f}  AUC={r['auc']:.4f}"
          f"  Sens={r['sens']:.4f}  Spec={r['spec']:.4f}")

best_name = max(results, key=lambda k: results[k]['auc'])
best_model = trained[best_name]
print(f"\n  Best model: {best_name}")

# --- Temporal analysis ----------------------------------------
print("\n[4/6] Temporal analysis (4 weekly batches)...")
wnames = ['Week 1 (Train)', 'Week 2 (Train)', 'Week 3 (Val)', 'Week 4 (Test)']
temporal = {}
for i, (Xw, yw) in enumerate(week_data):
    r = evalu(best_model, Xw, yw)
    temporal[wnames[i]] = r
    print(f"  {wnames[i]:<22} Acc={r['acc']:.4f}  AUC={r['auc']:.4f}")

# --- FIGURES --------------------------------------------------
print("\n[5/6] Generating figures...")
COLORS = {'Baseline CNN': '#2196F3', 'Deep CNN + L2': '#FF5722', 'Residual CNN': '#4CAF50'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})

# Figure 1 — Training curves
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Figure 1 — Training & Validation Curves', fontsize=14, fontweight='bold')
for col, (name, h) in enumerate(histories.items()):
    ep = range(1, len(h.history['loss']) + 1)
    c  = COLORS[name]
    axes[0, col].plot(ep, h.history['loss'],     color=c, lw=2, label='Train')
    axes[0, col].plot(ep, h.history['val_loss'], color=c, lw=2, ls='--', label='Val')
    axes[0, col].set_title(name, fontweight='bold')
    axes[0, col].set_ylabel('Loss'); axes[0, col].legend(fontsize=8); axes[0, col].grid(alpha=.3)
    axes[1, col].plot(ep, h.history['accuracy'],     color=c, lw=2, label='Train')
    axes[1, col].plot(ep, h.history['val_accuracy'], color=c, lw=2, ls='--', label='Val')
    axes[1, col].set_ylabel('Accuracy'); axes[1, col].set_ylim(0.4, 1.05)
    axes[1, col].legend(fontsize=8); axes[1, col].grid(alpha=.3)
    for ax in axes[:, col]:
        for s in ['top', 'right']: ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig('fig1_training_curves.png', dpi=150, bbox_inches='tight')
plt.close(); print("    fig1 saved")

# Figure 2 — Confusion matrices + ROC
fig = plt.figure(figsize=(18, 10))
fig.suptitle('Figure 2 — Confusion Matrices & ROC Curves', fontsize=14, fontweight='bold')
gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
for col, name in enumerate(histories.keys()):
    r = results[name]; cm = r['cm']
    ax_cm = fig.add_subplot(gs[0, col])
    ax_cm.imshow(cm, cmap='Blues', vmin=0, vmax=cm.max())
    ax_cm.set_xticks([0,1]); ax_cm.set_yticks([0,1])
    ax_cm.set_xticklabels(['Normal','Pneumonia'])
    ax_cm.set_yticklabels(['Normal','Pneumonia'])
    ax_cm.set_xlabel('Predicted'); ax_cm.set_ylabel('Actual')
    ax_cm.set_title(f'{name}\nAcc={r["acc"]:.3f}  AUC={r["auc"]:.3f}', fontsize=9, fontweight='bold')
    for i in range(2):
        for j in range(2):
            ax_cm.text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=16, fontweight='bold',
                       color='white' if cm[i,j] > cm.max()*0.5 else 'black')
    ax_roc = fig.add_subplot(gs[1, col])
    fpr, tpr, _ = roc_curve(y_val, r['yp'])
    ax_roc.plot(fpr, tpr, color=COLORS[name], lw=2.5, label=f"AUC={r['auc']:.3f}")
    ax_roc.plot([0,1],[0,1],'--', color='gray', lw=1)
    ax_roc.fill_between(fpr, tpr, alpha=0.12, color=COLORS[name])
    ax_roc.set_xlabel('FPR'); ax_roc.set_ylabel('TPR')
    ax_roc.set_title('ROC Curve', fontsize=9); ax_roc.legend(fontsize=9); ax_roc.grid(alpha=.3)
plt.savefig('fig2_confusion_roc.png', dpi=150, bbox_inches='tight')
plt.close(); print("    fig2 saved")

# Figure 3 — Temporal
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f'Figure 3 — Temporal Analysis ({best_name})', fontsize=13, fontweight='bold')
weeks     = list(temporal.keys())
wlabels   = ['Week 1\n(Train)', 'Week 2\n(Train)', 'Week 3\n(Val)', 'Week 4\n(Test)']
accs = [temporal[w]['acc'] for w in weeks]
aucs = [temporal[w]['auc'] for w in weeks]
sens = [temporal[w]['sens'] for w in weeks]
spec = [temporal[w]['spec'] for w in weeks]
x = np.arange(len(weeks)); wid = 0.35
b1 = ax1.bar(x-wid/2, accs, wid, label='Accuracy', color='#2196F3', alpha=0.85)
b2 = ax1.bar(x+wid/2, aucs, wid, label='AUC-ROC',  color='#FF5722', alpha=0.85)
ax1.set_xticks(x); ax1.set_xticklabels(wlabels, fontsize=9)
ax1.set_ylim(0.5, 1.05); ax1.set_ylabel('Score')
ax1.set_title('Accuracy & AUC per Week', fontweight='bold')
ax1.legend(); ax1.grid(axis='y', alpha=0.3)
for bar in list(b1)+list(b2):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
             f'{bar.get_height():.2f}', ha='center', fontsize=8, fontweight='bold')
ax2.plot(wlabels, sens, 'o-', color='#4CAF50', lw=2.5, ms=9, label='Sensitivity')
ax2.plot(wlabels, spec, 's-', color='#9C27B0', lw=2.5, ms=9, label='Specificity')
ax2.axvspan(-0.5, 1.5, alpha=0.07, color='blue')
ax2.axvspan(1.5,  3.5, alpha=0.07, color='orange')
ax2.set_ylim(0.3, 1.1); ax2.set_ylabel('Score')
ax2.set_title('Sensitivity & Specificity Over Time', fontweight='bold')
ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
for ax in [ax1, ax2]:
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig('fig3_temporal.png', dpi=150, bbox_inches='tight')
plt.close(); print("    fig3 saved")

# Figure 4 — Model comparison
fig, ax = plt.subplots(figsize=(11, 5))
names = list(results.keys())
mkeys   = ['acc', 'auc', 'sens', 'spec']
mlabels = ['Accuracy', 'AUC-ROC', 'Sensitivity', 'Specificity']
mcolors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']
x = np.arange(len(names)); w = 0.18
for i, (mk, ml, mc) in enumerate(zip(mkeys, mlabels, mcolors)):
    vals = [results[n][mk] for n in names]
    bars = ax.bar(x + i*w - 1.5*w, vals, w, label=ml, color=mc, alpha=0.85)
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f'{bar.get_height():.3f}', ha='center', fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10, fontweight='bold')
ax.set_ylim(0.4, 1.08); ax.set_ylabel('Score')
ax.set_title('Figure 4 — All Metrics Comparison', fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
for s in ['top', 'right']: ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig('fig4_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close(); print("    fig4 saved")

# --- Final summary --------------------------------------------
print("\n[6/6] Summary")
print("=" * 55)
print(f"{'Model':<22} {'Acc':>6} {'AUC':>6} {'Sens':>6} {'Spec':>6}")
print("-" * 55)
for name, r in results.items():
    print(f"{name:<22} {r['acc']:>6.4f} {r['auc']:>6.4f} "
          f"{r['sens']:>6.4f} {r['spec']:>6.4f}")
print("\n  Temporal analysis:")
for w, r in temporal.items():
    print(f"  {w:<22} Acc={r['acc']:.4f}  AUC={r['auc']:.4f}")

print("\n✅ All done! 4 figures saved in the same folder as this script.")
print("   fig1_training_curves.png")
print("   fig2_confusion_roc.png")
print("   fig3_temporal.png")
print("   fig4_model_comparison.png")

# Explicitly save the generated weights down to the repository layout workspace
best_model.save("best_model.h5")
