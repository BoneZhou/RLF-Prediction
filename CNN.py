import os, random, warnings, time, math
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, roc_curve

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

# ======= 在這裡填入 4 個資料集路徑 =======（依需要自行調整）
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
            hit = det_matrix[i, k] == 1
            ax_main.scatter(k + 1, i + 1,
                            s=60,
                            facecolors=('red' if hit else 'white'),
                            edgecolors=('red' if hit else 'black'),
                            linewidths=0.8)
    if SHOW_ROW_SUM:
        counts = det_matrix.sum(axis=1)
        for i, c in enumerate(counts):
            ax_main.text(ws + 1.2, i + 1, str(int(c)), va='center', fontsize=9)

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

def save_learning_curve(history, out_png_path, title="CNN Training and Validation AUC"):
    auc_tr  = history.history.get("auc", [])
    auc_va  = history.history.get("val_auc", [])
    x_axis = range(1, len(auc_tr)+1)
    plt.figure(figsize=(8,5), dpi=150)
    plt.plot(x_axis, auc_tr, label='Train AUC')
    plt.plot(x_axis, auc_va, linestyle='--', label='Validation AUC')  # 驗證改虛線
    plt.xlabel('Epochs'); plt.ylabel('AUC'); plt.title(title); plt.legend(); plt.tight_layout()
    plt.savefig(out_png_path, bbox_inches='tight'); plt.close()

def save_roc_curve(y_true, y_prob, out_png_path, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6,5), dpi=150)
    plt.plot(fpr, tpr, label=f"ROC (AUC={auc_val:.3f})")
    plt.plot([0,1], [0,1], linestyle=':', label='Random')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(title); plt.legend(); plt.tight_layout()
    plt.savefig(out_png_path, bbox_inches='tight'); plt.close()

def build_cnn(input_shape,
              base_filters=64,
              kernel_size=3,
              blocks=2,
              dropout=0.2,
              l2_reg=0.0):
    """簡潔 1D CNN：Conv-BN-ReLU-MaxPool × blocks -> GAP -> Dense"""
    inputs = layers.Input(shape=input_shape)
    x = inputs
    for b in range(blocks):
        x = layers.Conv1D(filters=base_filters*(2**b),
                          kernel_size=kernel_size,
                          padding='same',
                          activation='relu',
                          kernel_regularizer=regularizers.l2(l2_reg) if l2_reg>0 else None)(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        if dropout>0: x = layers.Dropout(dropout)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    if dropout>0: x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs, outputs)
    return model

# ======= CNN FLOPs 估算（近似）=======
def estimate_cnn_flops_per_sample(ws, feat_dim=2, base_filters=64, kernel_size=3, blocks=2):
    """
    估算單樣本 FLOPs（≈ 2×MACs）：
      Conv1D(b): L_in * out_ch * (kernel_size * in_ch)
      每個 block 之後 MaxPool1D 將長度約減半（向上取整）
      Head：GAP 忽略、Dense(last_ch->64) + Dense(64->1)
    """
    macs = 0.0
    L = ws
    in_ch = feat_dim
    for b in range(blocks):
        out_ch = base_filters * (2**b)
        macs += L * out_ch * (kernel_size * in_ch)
        L = (L + 1) // 2   # 約等於 ceil(L/2)
        in_ch = out_ch
    # Dense head
    macs += in_ch * 64 + 64 * 1
    return 2.0 * macs

# ======= 主流程 =======
desktop_dir = os.path.join(os.path.expanduser("~"), "Desktop")

for ds_idx, (ds_name, csv_path) in enumerate(DATASETS, start=1):
    base_dir = os.path.join(desktop_dir, ds_name)
    out_dir  = os.path.join(base_dir, "cnn")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isfile(csv_path):
        print(f"⚠️ 資料集 {ds_name} 路徑不存在：{csv_path}，跳過。")
        continue

    print(f"\n=== [CNN] 處理資料集 {ds_name}: {csv_path} ===")
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
                print(f"[CNN][{ds_name}] ws={ws}, fw={fw}: 樣本為 0，跳過。")
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

            # 建模
            base_filters = 64; blocks = 2; kernel_size = 3
            model = build_cnn(input_shape=(ws, 2),
                              base_filters=base_filters,
                              kernel_size=kernel_size,
                              blocks=blocks,
                              dropout=0.2,
                              l2_reg=0.0)
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
            save_learning_curve(history, lc_png, title="CNN Training and Validation AUC")

            # 推論與門檻
            t0 = time.perf_counter()
            y_prob_val = model.predict(X_val, batch_size=1024, verbose=0).ravel()
            t1 = time.perf_counter()
            infer_total_s = t1 - t0
            infer_ms_per_sample = (infer_total_s / max(1,len(X_val))) * 1000.0
            thr, _ = best_threshold(y_val, y_prob_val)
            y_pred_val = (y_prob_val >= thr).astype(int)

            # 事件矩陣與圖
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

            det_matrix, num_events_total, n_events_eval = build_event_matrices(df, ws, fw, val_idx, y_pred_val)
            ev_png = os.path.join(out_dir, f"ws{ws}_fw{fw}.png")
            plot_and_save(det_matrix, ws, fw, num_events_total, ev_png)

            # ROC 曲線
            roc_png = os.path.join(out_dir, f"roc_ws{ws}_fw{fw}.png")
            save_roc_curve(y_val, y_prob_val, roc_png, title=f"CNN ROC (ws={ws}, fw={fw})")

            # 指標
            acc  = accuracy_score(y_val, y_pred_val)
            try:
                auc  = roc_auc_score(y_val, y_prob_val)
            except ValueError:
                auc = float("nan")
            prec = precision_score(y_val, y_pred_val, zero_division=0)
            rec  = recall_score(y_val, y_pred_val, zero_division=0)
            f1v  = f1_score(y_val, y_pred_val, zero_division=0)

            # FLOPs 估算
            flops_per_sample = estimate_cnn_flops_per_sample(ws,
                                                             feat_dim=2,
                                                             base_filters=base_filters,
                                                             kernel_size=kernel_size,
                                                             blocks=blocks)
            flops_total_val = flops_per_sample * len(X_val)

            # 存模型
            model_path = os.path.join(out_dir, f"cnn_ws{ws}_fw{fw}.keras")
            model.save(model_path)

            print(f"[CNN][{ds_name}] ws={ws}, fw={fw} | thr={thr:.2f} | "
                  f"acc={acc:.3f} auc={auc:.3f} prec={prec:.3f} rec={rec:.3f} f1={f1v:.3f} | "
                  f"infer: {infer_total_s:.6f}s total, {infer_ms_per_sample:.3f} ms/sample | "
                  f"FLOPs≈ {flops_per_sample:.0f}/sample, {flops_total_val:.0f} (val total) | "
                  f"圖：{ev_png} / {lc_png} / {roc_png} | 模型：{model_path}")

print("\n✅ CNN 全部處理完成。到每個資料集資料夾的 cnn/ 查看事件圖 + 學習曲線 + ROC 圖 + .keras 模型。")
