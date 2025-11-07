import os, random, warnings, time, math
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
np.random.seed(RANDOM_STATE); random.seed(RANDOM_STATE); tf.random.set_seed(RANDOM_STATE)

# ========= 自動設定支援中文的字型 =========
font_paths = [r"C:\Windows\Fonts\msjh.ttc",
              "/System/Library/Fonts/PingFang.ttc",
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]
for fp in font_paths:
    if os.path.exists(fp):
        try:
            prop = fm.FontProperties(fname=fp)
            plt.rcParams["font.family"] = prop.get_name()
            break
        except Exception:
            pass
plt.rcParams["axes.unicode_minus"] = False

# ======= 4 個資料集路徑（與你之前相同）=======
DATASETS = [
    ("xg01", r"c:\Users\admin\Desktop\學姊\資料\2024-03-19_sm00_UDP_Bandlock_9S_Phone_#01_All.csv"),
    ("xg02", r"c:\Users\admin\Desktop\學姊\資料\2024-03-19_sm00_UDP_Bandlock_9S_Phone_#02_All.csv"),
    ("xg03", r"c:\Users\admin\Desktop\學姊\資料\2024-03-19_sm01_UDP_Bandlock_9S_Phone_#01_All.csv"),
    ("xg04", r"c:\Users\admin\Desktop\學姊\資料\2024-03-19_sm01_UDP_Bandlock_9S_Phone_#02_All.csv"),
]

# 時間窗（以「步數」；你的資料 10Hz → 10步=1秒）
WINDOW_SIZES   = [10, 20, 30]
FUTURE_WINDOWS = [10, 20, 30]
TEST_SIZE      = 0.3

# 視覺化細節（與 CNN/LSTM/PatchTST 同步）
SHOW_ROW_SUM = True
STAT_FONT    = 16
STAT_LINE_H  = 0.18

# ======= 基本工具 =======
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
        X.append(np.stack([seq_rsrp, rsrq_all[start_idx:start_idx+ws]], axis=-1))  # (ws, 2)
        y.append(1 if (future != 0).any() else 0)
    return np.array(X, np.float32), np.array(y, np.int32)

def standardize_by_train(X_train, X_val):
    feat_dim = X_train.shape[-1]
    flat = X_train.reshape(-1, feat_dim)
    mean = flat.mean(axis=0); std = flat.std(axis=0) + 1e-8
    return (X_train - mean)/std, (X_val - mean)/std

def best_threshold(y_val, y_prob, start=0.1, end=0.91, step=0.01):
    best_t, best_f1 = 0.5, -1.0; t = start
    while t <= end + 1e-9:
        f1 = f1_score(y_val, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1: best_f1, best_t = f1, t
        t += step
    return best_t, best_f1

# ======= 事件圖工具（與你其他模型完全一致）=======
def has_consecutive(arr: np.ndarray, run_len: int) -> bool:
    if run_len <= 1: return arr.sum() >= 1
    cnt = 0
    for v in arr:
        cnt = cnt + 1 if v == 1 else 0
        if cnt >= run_len: return True
    return False

def compute_stats(det_matrix, num_events_total):
    n_events_eval = det_matrix.shape[0]
    if n_events_eval == 0: return dict(at1=0, c2=0, c3=0, a2=0, a3=0)
    sums = det_matrix.sum(axis=1)
    return dict(
        at1=int((sums >= 1).sum()),
        a2=int((sums >= 2).sum()),
        a3=int((sums >= 3).sum()),
        c2=int(sum(has_consecutive(det_matrix[i], 2) for i in range(n_events_eval))),
        c3=int(sum(has_consecutive(det_matrix[i], 3) for i in range(n_events_eval))),
    )

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

def plot_and_save(det_matrix, ws, fw, num_events_total, out_png_path):
    n_events = det_matrix.shape[0]
    fig = plt.figure(figsize=(11, 7.5), dpi=150)
    ax_main = fig.add_axes([0.08, 0.12, 0.62, 0.78])
    ax_stat = fig.add_axes([0.73, 0.20, 0.25, 0.60]); ax_stat.axis('off')

    if n_events == 0:
        ax_main.set_title("No evaluable events (no valid windows in validation set)")
        fig.savefig(out_png_path, bbox_inches='tight'); plt.close(fig); return

    stats = compute_stats(det_matrix, num_events_total)
    at1, c2, c3, a2, a3 = stats['at1'], stats['c2'], stats['c3'], stats['a2'], stats['a3']

    for i in range(n_events):
        for k in range(ws):
            hit = det_matrix[i, k] == 1
            ax_main.scatter(k + 1, i + 1, s=60,
                            facecolors=('red' if hit else 'white'),
                            edgecolors=('red' if hit else 'black'),
                            linewidths=0.8, alpha=1.0, antialiased=False)
    if SHOW_ROW_SUM:
        counts = det_matrix.sum(axis=1)
        for i, c in enumerate(counts): ax_main.text(ws + 1.2, i + 1, str(c), va='center', fontsize=9)

    ax_main.set_xlabel("Window index before event (1 oldest ... N just before)")
    ax_main.set_ylabel("Event # (actual occurrences)")
    ax_main.set_xlim(0.5, ws + 5.0); ax_main.set_ylim(0.5, n_events + 1); ax_main.invert_yaxis()

    labels = ["至少1點為紅","連續2點為紅","連續3點為紅","任意2點為紅","任意3點為紅"]
    nums   = [at1, c2, c3, a2, a3]
    y, dy, fs = 1.0, STAT_LINE_H, STAT_FONT
    for lab, num in zip(labels, nums):
        ax_stat.text(0.00, y, lab, ha='left', va='top', fontsize=fs)
        ax_stat.text(0.66, y, f": {num}/{num_events_total} = {num/num_events_total*100:.1f}%",
                     ha='left', va='top', fontsize=fs)
        y -= dy

    fig.patch.set_facecolor('white'); ax_main.set_facecolor('white')
    fig.savefig(out_png_path, bbox_inches='tight'); plt.close(fig)

def save_learning_curve(history, out_png_path, title="TimesNet Training and Validation AUC"):
    auc_tr  = history.history.get("auc", [])
    auc_va  = history.history.get("val_auc", [])
    x_axis = range(1, len(auc_tr)+1)
    plt.figure(figsize=(8,5), dpi=150)
    plt.plot(x_axis, auc_tr, label='Train AUC')
    plt.plot(x_axis, auc_va, linestyle='--', label='Validation AUC')  # Val 虛線
    plt.xlabel('Epochs'); plt.ylabel('AUC'); plt.title(title); plt.legend(); plt.tight_layout()
    plt.savefig(out_png_path, bbox_inches='tight'); plt.close()

# ======= TimesNet（Keras 簡化實作，TimesBlock 多週期卷積）=======
def _pad_to_multiple(x, period):
    """
    把長度 L 補到能被 period 整除（在時間軸尾端補零）。
    回傳：x_pad, L_orig(int32 Tensor), pad(int32 Tensor)
    """
    L = tf.shape(x)[1]  # 原始 L（Tensor）
    p = tf.cast(period, tf.int32)
    rem = tf.math.mod(L, p)
    pad = tf.where(tf.equal(rem, 0), tf.zeros_like(rem), p - rem)  # 若剛好整除→pad=0
    paddings = tf.stack([[0, 0], [0, pad], [0, 0]])               # 後端補 pad
    x_pad = tf.pad(x, paddings)
    return x_pad, L, pad

class TimesBlock(layers.Layer):
    """
    TimesNet 風格的多週期卷積：
      - 對多個 period： (B,L,C) → (B, L//p, p, d) 做 2D Conv，再還原到 (B,L,d)，多週期平均 + 殘差
    """
    def __init__(self, d_model=128, periods=(2,3,4,5), ksize=(3,3), dropout=0.2, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.periods = periods
        self.ksize   = ksize
        self.dropout = layers.Dropout(dropout)
        self.proj_in = layers.Dense(d_model)      # (B,L,C) → (B,L,d)
        self.norm1   = layers.LayerNormalization(epsilon=1e-6)
        # 每個 period 對應一組 Conv2D + 1x1 Conv
        self.convs   = []
        for _ in periods:
            self.convs.append(tf.keras.Sequential([
                layers.Conv2D(filters=d_model, kernel_size=ksize, padding='same', activation='relu'),
                layers.Conv2D(filters=d_model, kernel_size=1,    padding='same', activation=None),
            ]))
        self.proj_out = layers.Dense(d_model)
        self.norm2    = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training=None):
        # x: (B, L, C)
        d_model_i32 = tf.constant(self.d_model, dtype=tf.int32)

        h = self.proj_in(x)        # (B,L,d)
        h = self.norm1(h)

        outs = []
        for p_val, conv in zip(self.periods, self.convs):
            z, L, pad = _pad_to_multiple(h, p_val)     # (B, Lp, d)
            bs = tf.shape(z)[0]
            Lp = tf.shape(z)[1]
            p  = tf.cast(p_val, tf.int32)

            # 轉成 (B, Np, p, d)
            Np = tf.math.floordiv(Lp, p)
            shape1 = tf.stack([bs, Np, p, d_model_i32])
            z = tf.reshape(z, shape1)

            # 2D 卷積
            z = conv(z, training=training)             # (B, Np, p, d)

            # 還原回 (B, Lp, d)
            shape2 = tf.stack([bs, Lp, d_model_i32])
            z = tf.reshape(z, shape2)

            # 去掉補的 pad：保留前 L 步（全 Tensor slice）
            z = tf.slice(z, begin=[0, 0, 0], size=[-1, L, -1])  # (B, L, d)

            outs.append(z)

        y = tf.add_n(outs) / float(len(outs))          # 多週期平均
        y = self.proj_out(y)
        y = self.dropout(y, training=training)

        # ✅ 修正殘差維度：與 h 相加（h 已為 d_model 維度）
        y = self.norm2(h + y)
        return y

def build_timesnet(input_shape,
                   d_model=128,
                   blocks=2,
                   periods_scheme="auto",
                   dropout=0.2):
    """
    簡化 TimesNet：
      TimesBlock × blocks → GAP → Dense(64) → Dropout → Dense(1,sigmoid)
    periods_scheme='auto' 會依 ws 給一組合適的 periods（保證可切出 ≥2 個 patch）。
    """
    ws, in_ch = input_shape
    if periods_scheme == "auto":
        cand = [2,3,4,5,6,7,8,10,12,15]
        periods = tuple([p for p in cand if ws // p >= 2][:4] or [2,3])
    else:
        periods = periods_scheme

    inputs = layers.Input(shape=input_shape)
    x = inputs
    for _ in range(blocks):
        x = TimesBlock(d_model=d_model, periods=periods, ksize=(3,3), dropout=dropout)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs, outputs, name='TimesNet_KerasLite')
    model._timesnet_periods = periods  # 給 FLOPs 估算用
    model._timesnet_d_model = d_model
    model._timesnet_blocks  = blocks
    return model

# ======= FLOPs 估算（近似，每樣本）=======
def estimate_timesnet_flops_per_sample(ws, in_ch=2, d_model=128, blocks=2, periods=(2,3,4,5), ksize=(3,3)):
    """
    近似 2D Conv FLOPs：H*W*Cin*Kh*Kw*Cout*2
    對每個 TimesBlock、每個 period：把 (L,d_model) → (L//p, p, d_model)，做 Conv2D(kh,kw,d_model→d_model) + 1x1 Conv
    忽略 LN/Dropout/Residual 等。
    """
    Kh, Kw = ksize
    flops = 0.0
    # 第一層 Dense 投影 (in_ch→d_model)： L * in_ch * d_model * 2
    flops += ws * in_ch * d_model * 2
    for _ in range(blocks):
        for p in periods:
            Np = math.ceil(ws / p)  # 補齊後的 patch 數
            # 主 3x3 Conv2D： (Np * p) * d_model(in) * Kh*Kw * d_model(out) *2
            flops += (Np * p) * d_model * Kh * Kw * d_model * 2
            # 1x1 Conv2D： (Np * p) * d_model * 1 * 1 * d_model *2
            flops += (Np * p) * d_model * d_model * 2
        # block 末端 Dense(d_model→d_model)： L * d_model * d_model * 2
        flops += ws * d_model * d_model * 2
    # Head Dense 64 → 1
    flops += d_model * 64 * 2 + 64 * 1 * 2
    return flops

# ======= 主流程（輸出到 Desktop/newtimes/times1~times4）=======
desktop_dir = os.path.join(os.path.expanduser("~"), "Desktop")
root_out    = os.path.join(desktop_dir, "newtimes")
os.makedirs(root_out, exist_ok=True)

for ds_idx, (ds_name, csv_path) in enumerate(DATASETS, start=1):
    base_dir = os.path.join(root_out, f"times{ds_idx}")  # times1 ~ times4
    os.makedirs(base_dir, exist_ok=True)

    if not os.path.isfile(csv_path):
        print(f"⚠️ 資料集 {ds_name} 路徑不存在：{csv_path}，跳過。"); continue

    print(f"\n=== [TimesNet] 處理資料集 {ds_name} → 儲存到：{base_dir} ===")
    df = pd.read_csv(csv_path)
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.sort_values(by="Timestamp").reset_index(drop=True)
    ensure_cols(df, ["RSRP", "RSRQ", "RLF_II"])
    df[["RSRP","RSRQ","RLF_II"]] = df[["RSRP","RSRQ","RLF_II"]].fillna(0)

    rows = []

    for ws in WINDOW_SIZES:
        for fw in FUTURE_WINDOWS:
            X, y = build_xy_seq(df, ws, fw)
            if len(X) == 0:
                print(f"[TimesNet][{ds_name}] ws={ws}, fw={fw}: 樣本為 0，跳過。"); continue

            idxs = np.arange(len(X))
            train_idx, val_idx = train_test_split(idxs, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # 標準化
            X_train, X_val = standardize_by_train(X_train, X_val)

            # 類別權重
            pos = int(y_train.sum()); neg = int(len(y_train) - pos)
            class_weight = None
            if pos>0 and neg>0:
                w0 = len(y_train)/(2.0*neg); w1 = len(y_train)/(2.0*pos)
                class_weight = {0: w0, 1: w1}

            # 建模（TimesNet）
            d_model=128; blocks=2
            model = build_timesnet(input_shape=(ws, 2),
                                   d_model=d_model, blocks=blocks,
                                   periods_scheme="auto", dropout=0.2)
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
                epochs=60, batch_size=128,
                class_weight=class_weight, verbose=0, callbacks=cbs
            )

            # 學習曲線（Val AUC 虛線）
            lc_png = os.path.join(base_dir, f"learning_ws{ws}_fw{fw}.png")
            save_learning_curve(history, lc_png, title="TimesNet Training and Validation AUC")

            # 推論時間
            t0 = time.perf_counter()
            y_prob_val = model.predict(X_val, batch_size=1024, verbose=0).ravel()
            t1 = time.perf_counter()
            infer_total_s = t1 - t0
            infer_ms_per_sample = (infer_total_s / max(1, len(X_val))) * 1000.0

            # 最佳門檻與預測
            thr, _ = best_threshold(y_val, y_prob_val)
            y_pred_val = (y_prob_val >= thr).astype(int)

            # 事件矩陣與圖
            det_matrix, num_events_total, _ = build_event_matrices(df, ws, fw, val_idx, y_pred_val)
            out_png = os.path.join(base_dir, f"ws{ws}_fw{fw}.png")
            plot_and_save(det_matrix, ws, fw, num_events_total, out_png)

            # 指標
            acc  = accuracy_score(y_val, y_pred_val)
            try: auc = roc_auc_score(y_val, y_prob_val)
            except ValueError: auc = float("nan")
            prec = precision_score(y_val, y_pred_val, zero_division=0)
            rec  = recall_score(y_val, y_pred_val, zero_division=0)
            f1v  = f1_score(y_val, y_pred_val, zero_division=0)

            # FLOPs 估算
            periods = getattr(model, "_timesnet_periods", (2,3,4,5))
            flops_per_sample = estimate_timesnet_flops_per_sample(ws,
                                                                  in_ch=2, d_model=d_model,
                                                                  blocks=blocks, periods=periods, ksize=(3,3))
            flops_total_val  = flops_per_sample * len(X_val)

            print(f"[TimesNet][{ds_name}] ws={ws}, fw={fw} | thr={thr:.2f} | "
                  f"acc={acc:.3f} auc={auc:.3f} prec={prec:.3f} rec={rec:.3f} f1={f1v:.3f} | "
                  f"infer: {infer_total_s:.6f}s total, {infer_ms_per_sample:.3f} ms/sample | "
                  f"FLOPs≈ {flops_per_sample:.0f}/sample, {flops_total_val:.0f} (val total) | "
                  f"圖：{out_png} / {lc_png}")

            # 存模型
            model_path = os.path.join(base_dir, f"timesnet_ws{ws}_fw{fw}.keras")
            model.save(model_path)

            # 累積到此資料集的表格
            rows.append({
                "dataset": ds_name, "ws": int(ws), "fw": int(fw),
                "thr": float(thr), "accuracy": float(acc),
                "auc": float(auc) if not np.isnan(auc) else None,
                "precision": float(prec), "recall": float(rec), "f1": float(f1v),
                "n_val": int(len(y_val)), "pos_in_val": int(y_val.sum()),
                "neg_in_val": int(len(y_val) - y_val.sum()),
                "infer_total_seconds": float(infer_total_s),
                "infer_ms_per_sample": float(infer_ms_per_sample),
                "est_FLOPs_per_sample": float(flops_per_sample),
                "est_FLOPs_val_total": float(flops_total_val),
                "periods": str(periods),
                "d_model": int(d_model), "blocks": int(blocks)
            })

    # 輸出此資料集的彙整 CSV（若被占用則 fallback）
    if len(rows) > 0:
        df_metrics = pd.DataFrame(rows)
        csv_out = os.path.join(base_dir, "metrics_summary.csv")
        try:
            df_metrics.to_csv(csv_out, index=False, encoding="utf-8-sig")
            print(f"📄 已輸出指標到：{csv_out}")
        except PermissionError:
            alt = os.path.join(base_dir, f"metrics_summary_{int(time.time())}.csv")
            df_metrics.to_csv(alt, index=False, encoding="utf-8-sig")
            print(f"⚠️ 原檔被占用，已改存：{alt}")

print("\n✅ 完成：請到桌面 newtimes/times1~times4 查看 9 張事件圖 + 9 張學習曲線 + metrics_summary.csv + .keras 模型。")
