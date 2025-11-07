import os, random, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, metrics, regularizers

# ========= 隨機種子 =========
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# ========= 自動設定支援中文的字型 =========
font_paths = [
    r"C:\Windows\Fonts\msjh.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]
for fp in font_paths:
    if os.path.exists(fp):
        prop = fm.FontProperties(fname=fp)
        plt.rcParams["font.family"] = prop.get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

# ======= 在這裡填入 4 個資料集路徑 =======
DATASETS = [
    ("xg01", r"c:\Users\admin\Desktop\學姊\資料\2024-03-19_sm00_UDP_Bandlock_9S_Phone_#01_All.csv"),
    ("xg02", r"c:\Users\admin\Desktop\學姊\資料\2024-03-19_sm00_UDP_Bandlock_9S_Phone_#02_All.csv"),
    ("xg03", r"c:\Users\admin\Desktop\學姊\資料\2024-03-19_sm01_UDP_Bandlock_9S_Phone_#01_All.csv"),
    ("xg04", r"c:\Users\admin\Desktop\學姊\資料\2024-03-19_sm01_UDP_Bandlock_9S_Phone_#02_All.csv"),
]

# 視窗組合
WINDOW_SIZES   = [10, 20, 30]
FUTURE_WINDOWS = [10, 20, 30]

# 通用參數
TEST_SIZE     = 0.3

# 視覺化細節
SHOW_ROW_SUM = True
STAT_FONT    = 16
STAT_LINE_H  = 0.18

# ======= 工具函式 =======
def ensure_cols(df, required_cols):
    assert all(c in df.columns for c in required_cols), \
        f"CSV 缺少必要欄位，需包含：{required_cols}，目前欄位：{df.columns.tolist()}"

def build_xy_seq(df, ws, fw):
    """回傳序列 X: (N, ws, 2) 及 y: (N,)；特徵= [RSRP, RSRQ]"""
    X, y = [], []
    rsrp_all = df["RSRP"].values
    rsrq_all = df["RSRQ"].values
    rlf_all  = df["RLF_II"].values
    for start_idx in range(0, len(df) - ws - fw + 1):
        seq_rsrp = rsrp_all[start_idx:start_idx+ws]
        seq_rsrq = rsrq_all[start_idx:start_idx+ws]
        future   = rlf_all[start_idx+ws:start_idx+ws+fw]
        X.append(np.stack([seq_rsrp, seq_rsrq], axis=-1))  # (ws, 2)
        label = 1 if (future != 0).any() else 0
        y.append(label)
    return np.array(X, np.float32), np.array(y, np.int32)

def standardize_by_train(X_train, X_val):
    """依 train 的 feature 統計做 Z-score；對每個 time step 同一組均值/方差"""
    feat_dim = X_train.shape[-1]
    flat = X_train.reshape(-1, feat_dim)
    mean = flat.mean(axis=0)
    std  = flat.std(axis=0) + 1e-8
    Xtr = (X_train - mean) / std
    Xva = (X_val  - mean) / std
    return Xtr, Xva

def best_threshold(y_val, y_prob, start=0.1, end=0.91, step=0.01):
    best_t, best_f1 = 0.5, -1.0
    t = start
    while t <= end + 1e-9:
        f1 = f1_score(y_val, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
        t += step
    return best_t, best_f1

def has_consecutive(arr: np.ndarray, run_len: int) -> bool:
    if run_len <= 1:
        return arr.sum() >= 1
    cnt = 0
    for v in arr:
        cnt = cnt + 1 if v == 1 else 0
        if cnt >= run_len: return True
    return False

def compute_stats(det_matrix, num_events_total):
    n_events_eval = det_matrix.shape[0]
    if n_events_eval == 0:
        return dict(at1=0, c2=0, c3=0, a2=0, a3=0)
    sums = det_matrix.sum(axis=1)
    at_least_1 = int((sums >= 1).sum())
    any_2      = int((sums >= 2).sum())
    any_3      = int((sums >= 3).sum())
    cons_2     = int(sum(has_consecutive(det_matrix[i], 2) for i in range(n_events_eval)))
    cons_3     = int(sum(has_consecutive(det_matrix[i], 3) for i in range(n_events_eval)))
    return dict(at1=at_least_1, c2=cons_2, c3=cons_3, a2=any_2, a3=any_3)

def plot_and_save(det_matrix, ws, fw, num_events_total, out_png_path):
    n_events = det_matrix.shape[0]
    fig = plt.figure(figsize=(11, 7.5), dpi=150)
    ax_main = fig.add_axes([0.08, 0.12, 0.62, 0.78])
    ax_stat = fig.add_axes([0.73, 0.20, 0.25, 0.60])
    ax_stat.axis('off')

    if n_events == 0:
        ax_main.set_title("No evaluable events (no valid windows in validation set)")
        fig.savefig(out_png_path, bbox_inches='tight'); plt.close(fig); return

    stats = compute_stats(det_matrix, num_events_total)
    at1, c2, c3, a2, a3 = stats['at1'], stats['c2'], stats['c3'], stats['a2'], stats['a3']

    for i in range(n_events):
        for k in range(ws):
            color = 'red' if det_matrix[i, k] == 1 else 'lightgray'
            ax_main.scatter(k + 1, i + 1, color=color, s=50)
    if SHOW_ROW_SUM:
        counts = det_matrix.sum(axis=1)
        for i, c in enumerate(counts):
            ax_main.text(ws + 1.2, i + 1, str(c), va='center', fontsize=9)

    ax_main.set_xlabel("Window index before event (1 oldest ... N just before)")
    ax_main.set_ylabel("Event # (actual occurrences)")
    ax_main.set_xlim(0.5, ws + 5.0)
    ax_main.set_ylim(0.5, n_events + 1)
    ax_main.invert_yaxis()

    labels = ["至少1點為紅","連續2點為紅","連續3點為紅","任意2點為紅","任意3點為紅"]
    nums   = [at1, c2, c3, a2, a3]
    y, dy, fs = 1.0, STAT_LINE_H, STAT_FONT
    for lab, num in zip(labels, nums):
        ax_stat.text(0.00, y, lab, ha='left', va='top', fontsize=fs)
        ax_stat.text(0.66, y, f": {num}/{num_events_total} = {num/num_events_total*100:.1f}%",
                     ha='left', va='top', fontsize=fs)
        y -= dy

    fig.savefig(out_png_path, bbox_inches='tight'); plt.close(fig)

def build_event_matrices(df, WINDOW_SIZE, FUTURE_WINDOW, val_idx, y_pred_val):
    event_indices = np.where(df["RLF_II"].values != 0)[0]
    num_events_total = len(event_indices)
    val_set = set(val_idx.tolist())
    events_for_eval = []
    for j in event_indices:
        low  = max(0, j - WINDOW_SIZE - FUTURE_WINDOW + 1)
        high = min(j - WINDOW_SIZE, len(df) - WINDOW_SIZE - FUTURE_WINDOW)
        if high >= low:
            windows = list(range(low, high + 1))
            win_in_val = [s for s in windows if s in val_set]
            pos_in_val = []
            for s in win_in_val:
                pos = np.where(val_idx == s)[0]
                if pos.size > 0 and y_pred_val[pos[0]] == 1:
                    pos_in_val.append(s)
            events_for_eval.append((j, win_in_val, pos_in_val))
        else:
            events_for_eval.append((j, [], []))
    n_events_eval = len(events_for_eval)
    det_matrix = np.zeros((n_events_eval, WINDOW_SIZE), dtype=int)
    for i, (j, win_in_val, pos_in_val) in enumerate(events_for_eval):
        if not win_in_val: continue
        low = max(0, j - WINDOW_SIZE - FUTURE_WINDOW + 1)
        for s in pos_in_val:
            k = s - low
            if 0 <= k < WINDOW_SIZE: det_matrix[i, k] = 1
    return det_matrix, num_events_total, n_events_eval

def save_learning_curve(history, out_png_path, title="PatchTST Training and Validation AUC"):
    auc_tr  = history.history.get("auc", [])
    auc_va  = history.history.get("val_auc", [])
    x_axis = range(len(auc_tr))
    plt.figure(figsize=(8,5))
    plt.plot(x_axis, auc_tr, label='Train AUC')
    plt.plot(x_axis, auc_va, label='Validation AUC')
    plt.xlabel('Epochs'); plt.ylabel('AUC'); plt.title(title); plt.legend(); plt.tight_layout()
    plt.savefig(out_png_path, bbox_inches='tight'); plt.close()

# ========= PatchTST 核心 =========
def sinusoidal_position_encoding(length, d_model):
    """產生 (1, length, d_model) 的 sinusoidal 位置編碼（動態長度、無訓練參數）。"""
    position = tf.range(length, dtype=tf.float32)[:, tf.newaxis]  # (L,1)
    div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * (-np.log(10000.0) / d_model))  # (d_model/2,)
    pe = tf.zeros((length, d_model), dtype=tf.float32)
    sin = tf.sin(position * div_term)
    cos = tf.cos(position * div_term)
    pe = tf.reshape(tf.stack([sin, cos], axis=-1), (length, -1))[:,:d_model]
    return pe[tf.newaxis, ...]  # (1, L, d_model)

def build_patchtst(input_shape,
                   patch_len=8,
                   stride=None,
                   d_model=128,
                   depth=3,
                   n_heads=8,
                   d_ff=256,
                   dropout=0.2):
    """
    Keras 版 PatchTST 風格模型（簡化實作）：
    - Patch Embedding：Conv1D(kernel=patch_len, stride=stride, padding='valid') 做線性投影為 patch token
    - Sinusoidal Positional Encoding（動態長度，無訓練參數）
    - Encoder：TransformerEncoder × depth（MHSA + FFN，含殘差與LayerNorm）
    - Pooling：GlobalAveragePooling1D
    - Head：Dense(1, sigmoid)
    輸入：input_shape=(ws, 2)；輸出：sigmoid 二分類。
    """
    assert d_model % n_heads == 0, "d_model 必須可被 n_heads 整除"
    seq_len, feat_dim = input_shape
    if stride is None:
        stride = max(1, patch_len // 2)
    patch_len = min(patch_len, seq_len)

    inputs = layers.Input(shape=input_shape)              # (B, L, C)
    # Patch 嵌入 (B, Np, d_model)
    x = layers.Conv1D(filters=d_model,
                      kernel_size=patch_len,
                      strides=stride,
                      padding='valid',
                      use_bias=True,
                      activation=None)(inputs)           # (B, Np, d_model)

    # 動態 sinusoidal 位置編碼
    def add_sinusoidal_pe(t):
        L = tf.shape(t)[1]
        pe = sinusoidal_position_encoding(L, d_model)    # (1, L, d_model)
        return t + pe
    x = layers.Lambda(add_sinusoidal_pe)(x)

    # Transformer Encoder blocks
    for _ in range(depth):
        # MHSA block
        h = layers.LayerNormalization(epsilon=1e-6)(x)
        h = layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_model // n_heads,
                                      dropout=dropout)(h, h)
        h = layers.Dropout(dropout)(h)
        x = layers.Add()([x, h])
        # FFN block
        h2 = layers.LayerNormalization(epsilon=1e-6)(x)
        h2 = layers.Dense(d_ff, activation=tf.nn.gelu)(h2)
        h2 = layers.Dropout(dropout)(h2)
        h2 = layers.Dense(d_model)(h2)
        h2 = layers.Dropout(dropout)(h2)
        x = layers.Add()([x, h2])

    # 池化 + 分類頭
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation='relu')(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs, name='PatchTST_Keras')
    return model

def get_patchtst_hparams(ws: int):
    """
    依據 window size (ws) 回傳一組穩健的 PatchTST 超參數（確保 patch 數 ≥ 2）。
    """
    if ws <= 12:
        patch_len = min(4, ws)
        stride    = max(1, patch_len // 2)
        d_model   = 96
        n_heads   = 4
        depth     = 2
    elif ws <= 22:
        patch_len = min(6, ws)
        stride    = max(1, patch_len // 2)
        d_model   = 128
        n_heads   = 8
        depth     = 3
    else:
        patch_len = min(8, ws)
        stride    = max(1, patch_len // 2)
        d_model   = 160
        n_heads   = 8
        depth     = 4

    # 確保至少 2 個 patch
    def num_patches(L, P, S): 
        return (L - P) // S + 1 if (L - P) >= 0 else 0
    npatches = num_patches(ws, patch_len, stride)
    while npatches < 2 and stride > 1:
        stride -= 1
        npatches = num_patches(ws, patch_len, stride)
    while npatches < 2 and patch_len > 2:
        patch_len -= 1
        npatches = num_patches(ws, patch_len, stride)

    return dict(
        patch_len=patch_len,
        stride=stride,
        d_model=d_model,
        depth=depth,
        n_heads=n_heads,
        d_ff=d_model * 2,
        dropout=0.2
    )

# ======= 主流程 =======
desktop_dir = os.path.join(os.path.expanduser("~"), "Desktop")

for ds_idx, (ds_name, csv_path) in enumerate(DATASETS, start=1):
    base_dir = os.path.join(desktop_dir, ds_name)
    out_dir  = os.path.join(base_dir, "patchtst")   # ← 改成 patchtst
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isfile(csv_path):
        print(f"⚠️ 資料集 {ds_name} 路徑不存在：{csv_path}，跳過。")
        continue

    print(f"\n=== [PatchTST] 處理資料集 {ds_name}: {csv_path} ===")
    df = pd.read_csv(csv_path)
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.sort_values(by="Timestamp").reset_index(drop=True)
    ensure_cols(df, ["RSRP", "RSRQ", "RLF_II"])
    df[["RSRP","RSRQ","RLF_II"]] = df[["RSRP","RSRQ","RLF_II"]].fillna(0)

    for ws in WINDOW_SIZES:
        for fw in FUTURE_WINDOWS:
            X, y = build_xy_seq(df, ws, fw)
            if len(X) == 0:
                print(f"[PatchTST][{ds_name}] ws={ws}, fw={fw}: 樣本為 0，跳過。")
                continue

            idxs = np.arange(len(X))
            train_idx, val_idx = train_test_split(idxs, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # 標準化
            X_train, X_val = standardize_by_train(X_train, X_val)

            # 類別權重處理不平衡
            pos = int(y_train.sum()); neg = int(len(y_train) - pos)
            class_weight = None
            if pos>0 and neg>0:
                w0 = len(y_train)/(2.0*neg); w1 = len(y_train)/(2.0*pos)
                class_weight = {0: w0, 1: w1}

            # 建模（PatchTST，自動依 ws 帶入超參數）
            hp = get_patchtst_hparams(ws)
            model = build_patchtst(
                input_shape=(ws, 2),
                patch_len=hp["patch_len"],
                stride=hp["stride"],
                d_model=hp["d_model"],
                depth=hp["depth"],
                n_heads=hp["n_heads"],
                d_ff=hp["d_ff"],
                dropout=hp["dropout"]
            )
            model.compile(optimizer=optimizers.Adam(1e-3),
                          loss="binary_crossentropy",
                          metrics=[metrics.AUC(name="auc")])

            cbs = [
                callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=10, restore_best_weights=True),
                callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=5, min_lr=1e-5),
            ]

            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=60,
                batch_size=128,
                class_weight=class_weight,
                verbose=0,
                callbacks=cbs
            )

            # 學習曲線
            lc_png = os.path.join(out_dir, f"learning_ws{ws}_fw{fw}.png")
            save_learning_curve(history, lc_png, title="PatchTST Training and Validation AUC")

            # 推論與門檻
            y_prob_val = model.predict(X_val, batch_size=1024, verbose=0).ravel()
            t, _ = best_threshold(y_val, y_prob_val)
            y_pred_val = (y_prob_val >= t).astype(int)

            # 事件矩陣與圖
            det_matrix, num_events_total, n_events_eval = build_event_matrices(df, ws, fw, val_idx, y_pred_val)
            out_png = os.path.join(out_dir, f"ws{ws}_fw{fw}.png")
            plot_and_save(det_matrix, ws, fw, num_events_total, out_png)

            # 指標
            acc  = accuracy_score(y_val, y_pred_val)
            try:
                auc  = roc_auc_score(y_val, y_prob_val)
            except ValueError:
                auc = float("nan")
            prec = precision_score(y_val, y_pred_val, zero_division=0)
            rec  = recall_score(y_val, y_pred_val, zero_division=0)
            f1v  = f1_score(y_val, y_pred_val, zero_division=0)
            print(f"[PatchTST][{ds_name}] ws={ws}, fw={fw} | thr={t:.2f} | acc={acc:.3f} auc={auc:.3f} "
                  f"prec={prec:.3f} rec={rec:.3f} f1={f1v:.3f} | 圖片：{out_png} / {lc_png}")

print("\n✅ PatchTST 全部處理完成。到每個資料集資料夾的 patchtst/ 查看 9 張事件圖 + 9 張學習曲線。")
