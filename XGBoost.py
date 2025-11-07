import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score,
    confusion_matrix
)
import xgboost as xgb

# ========= 自動設定支援中文的字型 =========
_font_candidates = [
    r"C:\Windows\Fonts\msjh.ttc",  # 微軟正黑
    r"C:\Windows\Fonts\msjh.ttf",
    r"C:\Windows\Fonts\mingliu.ttc",
    "/System/Library/Fonts/PingFang.ttc",                      # macOS 蘋方
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Linux Noto CJK
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]
for _fp in _font_candidates:
    if os.path.exists(_fp):
        _prop = fm.FontProperties(fname=_fp)
        plt.rcParams["font.family"] = _prop.get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

# =========================
# 參數設定
# =========================
SAMPLE_RATE = 10  # 每秒10筆 (0.1秒/筆)
TEST_SIZE = 0.3
RANDOM_STATE = 42

# ---- 4 份資料集：名稱 + 檔案路徑（請依實際情況調整）----
DATASETS = [
    ("xg01", r"c:\Users\admin\Desktop\學姊\資料\2024-03-19_sm00_UDP_Bandlock_9S_Phone_#01_All.csv"),
    ("xg02", r"c:\Users\admin\Desktop\學姊\資料\2024-03-19_sm00_UDP_Bandlock_9S_Phone_#02_All.csv"),
    ("xg03", r"c:\Users\admin\Desktop\學姊\資料\2024-03-19_sm01_UDP_Bandlock_9S_Phone_#01_All.csv"),
    ("xg04", r"c:\Users\admin\Desktop\學姊\資料\2024-03-19_sm01_UDP_Bandlock_9S_Phone_#02_All.csv"),
]

# =========================
# 工具函式
# =========================
def ensure_cols(df, required_cols):
    assert all(col in df.columns for col in required_cols), \
        f"請確認 CSV 中包含 {required_cols} 這幾個欄位，當前有：{df.columns.tolist()}"

def generate_features_and_labels(df, window_sec, future_sec):
    WINDOW_SIZE = int(window_sec * SAMPLE_RATE)
    FUTURE_WINDOW = int(future_sec * SAMPLE_RATE)
    X, y = [], []
    for start_idx in range(0, len(df) - WINDOW_SIZE - FUTURE_WINDOW + 1):
        seq_df = df.iloc[start_idx : start_idx + WINDOW_SIZE]
        future_df = df.iloc[start_idx + WINDOW_SIZE : start_idx + WINDOW_SIZE + FUTURE_WINDOW]
        rsrp_vals = seq_df["RSRP"].values
        rsrq_vals = seq_df["RSRQ"].values
        feat = [
            rsrp_vals.mean(), rsrp_vals.max(), rsrp_vals.min(), rsrp_vals.std(), np.median(rsrp_vals), rsrp_vals[-1],
            rsrq_vals.mean(), rsrq_vals.max(), rsrq_vals.min(), rsrq_vals.std(), np.median(rsrq_vals), rsrq_vals[-1]
        ]
        label = 1 if (future_df["RLF_II"] != 0).any() else 0
        X.append(feat)
        y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

def estimate_flops_per_sample(model: xgb.XGBClassifier) -> float:
    """
    近似 XGBoost FLOPs/樣本：
      每棵樹的「葉節點平均深度」≈ 預測時比較次數（feature 門檻比較），
      再加上彙總各樹輸出時的加法（每棵樹 ~ 1 次）。
    回傳：FLOPs/樣本（比較 + 加法；近似值）
    """
    try:
        booster = model.get_booster()
        df_trees = booster.trees_to_dataframe()
        leaf_df = df_trees[df_trees['Feature'] == 'Leaf']
        avg_depth_per_tree = leaf_df.groupby('Tree')['Depth'].mean()
        if avg_depth_per_tree.isna().any():
            max_depth = getattr(model, "max_depth", 4) or 4
            avg_depth_per_tree = avg_depth_per_tree.fillna(max_depth)
        comparisons = float(avg_depth_per_tree.sum())
        n_trees = int(len(avg_depth_per_tree))
        additions = float(n_trees)
        return comparisons + additions
    except Exception:
        n_trees = int(getattr(model, "n_estimators", 100))
        max_depth = int(getattr(model, "max_depth", 4) or 4)
        return n_trees * (max_depth + 1.0)

# ---- 右圖樣式事件圖（紅=陽性、白底黑邊=陰性；右側五統計）----
def plot_event_window_detections(val_idx, y_pred, df,
                                 window_size, future_window,
                                 window_sec, future_sec,
                                 out_dir):
    event_indices = np.where(df["RLF_II"].values != 0)[0]
    total_events = len(event_indices)
    val_set = set(val_idx.tolist())

    # 建 det_matrix: [事件數, window_size]，1=該相對位置被判陽性
    events_for_eval = []
    for j in event_indices:
        low  = max(0, j - window_size - future_window + 1)
        high = min(j - window_size, len(df) - window_size - future_window)
        if high < low:
            events_for_eval.append((j, [], []))
            continue
        windows = list(range(low, high + 1))
        win_in_val = [s for s in windows if s in val_set]
        pos_in_val = []
        for s in win_in_val:
            pos = np.where(val_idx == s)[0]
            if pos.size > 0 and y_pred[pos[0]] == 1:
                pos_in_val.append(s)
        events_for_eval.append((j, win_in_val, pos_in_val))

    n_events = len(events_for_eval)
    det_matrix = np.zeros((n_events, window_size), dtype=int)
    for i, (j, win_in_val, pos_in_val) in enumerate(events_for_eval):
        if not win_in_val:
            continue
        low = max(0, j - window_size - future_window + 1)
        for s in pos_in_val:
            k = s - low
            if 0 <= k < window_size:
                det_matrix[i, k] = 1

    # 5 統計
    def _has_consecutive(arr, run_len):
        cnt = 0
        for v in arr:
            cnt = cnt + 1 if v == 1 else 0
            if cnt >= run_len:
                return True
        return False
    sums = det_matrix.sum(axis=1) if n_events > 0 else np.array([])
    at1 = int((sums >= 1).sum())
    any2 = int((sums >= 2).sum())
    any3 = int((sums >= 3).sum())
    con2 = int(sum(_has_consecutive(det_matrix[i], 2) for i in range(n_events)))
    con3 = int(sum(_has_consecutive(det_matrix[i], 3) for i in range(n_events)))

    # 繪圖
    fig = plt.figure(figsize=(13, 7.5), dpi=150)
    ax = fig.add_axes([0.07, 0.12, 0.62, 0.78])
    ax_stat = fig.add_axes([0.73, 0.20, 0.25, 0.60]); ax_stat.axis('off')

    ax.set_title(f"每事件偵測（視窗={window_sec}s, 預測={future_sec}s）\n紅=陽性；白=陰性；右側數字=該事件於驗證集可評估的視窗數")
    for i in range(n_events):
        for k in range(window_size):
            hit = det_matrix[i, k] == 1
            ax.scatter(k + 1, i + 1,
                       s=60,
                       facecolors=('red' if hit else 'white'),
                       edgecolors=('red' if hit else 'black'),
                       linewidths=0.8)
        ax.text(window_size + 1.2, i + 1, str(int(det_matrix[i].sum())),
                va='center', fontsize=10, color='black')

    ax.set_xlabel("Window index before event (1 oldest ... N just before)")
    ax.set_ylabel("Event # (actual occurrences)")
    ax.set_xlim(0.5, window_size + 5.0)
    ax.set_ylim(0.5, (n_events + 1) if n_events > 0 else 1.5)
    ax.invert_yaxis()
    ax.grid(False)
    fig.patch.set_facecolor('white'); ax.set_facecolor('white')

    labels = ["至少1點為紅","連續2點為紅","連續3點為紅","任意2點為紅","任意3點為紅"]
    nums   = [at1, con2, con3, any2, any3]
    denom  = max(1, total_events)
    y, dy, fs = 1.0, 0.18, 16
    for lab, num in zip(labels, nums):
        ax_stat.text(0.00, y, lab, ha='left', va='top', fontsize=fs)
        ax_stat.text(0.66, y, f": {num}/{denom} = {num/denom*100:.1f}%", ha='left', va='top', fontsize=fs)
        y -= dy

    out_png = os.path.join(out_dir, f"event_win{window_sec}s_pred{future_sec}s.png")
    fig.savefig(out_png, bbox_inches='tight'); plt.close(fig)
    print(f"📊 已儲存事件視窗圖（5統計版）：{out_png}")

def plot_auc_curve(evals_result: dict, window_sec: int, future_sec: int, out_dir: str):
    """繪製訓練/驗證 AUC 曲線；驗證曲線使用虛線。"""
    try:
        train_auc = evals_result['validation_0']['auc']
        val_auc   = evals_result['validation_1']['auc']
    except KeyError:
        keys = list(evals_result.keys())
        if len(keys) >= 2:
            train_auc = list(evals_result[keys[0]].values())[0]
            val_auc   = list(evals_result[keys[1]].values())[0]
        else:
            return
    rounds = np.arange(1, len(train_auc) + 1)
    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(rounds, train_auc, label='Train AUC')
    plt.plot(rounds, val_auc, linestyle='--', label='Valid AUC')
    plt.xlabel('Boosting Rounds'); plt.ylabel('AUC')
    plt.title(f'XGBoost AUC Curve (win={window_sec}s, pred={future_sec}s)')
    plt.legend(); plt.grid(True, linestyle=':', alpha=0.5)
    out_path = os.path.join(out_dir, f"xgb_auc_curve_win{window_sec}s_pred{future_sec}s.png")
    plt.tight_layout(); plt.savefig(out_path, bbox_inches='tight'); plt.close()
    print(f"📈 已儲存 AUC 曲線圖：{out_path}")

def save_roc_curve(y_true, y_prob, window_sec, future_sec, out_dir):
    """儲存 ROC 曲線（含 AUC）。"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6, 5), dpi=150)
    plt.plot(fpr, tpr, label=f"ROC (AUC={auc_val:.3f})")
    plt.plot([0,1], [0,1], linestyle=':', label="Random")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC (win={window_sec}s, pred={future_sec}s)")
    plt.legend(); plt.tight_layout()
    out_path = os.path.join(out_dir, f"xgb_roc_win{window_sec}s_pred{future_sec}s.png")
    plt.savefig(out_path, bbox_inches='tight'); plt.close()
    print(f"🧪 已儲存 ROC 曲線圖：{out_path}")

# =========================
# 主流程：逐一處理 4 份資料，輸出到桌面/newxg/newxg01~04/cnn/
# =========================
desktop_dir = os.path.join(os.path.expanduser("~"), "Desktop")
base_root = os.path.join(desktop_dir, "newxg")  # 新總資料夾：Desktop/newxg

for ds_name, csv_path in DATASETS:
    new_ds_name = f"new{ds_name}"  # xg01 -> newxg01
    out_dir = os.path.join(base_root, new_ds_name, "cnn")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== 資料集 {new_ds_name} ===")
    print("CSV 路徑：", csv_path)
    print("輸出資料夾：", out_dir)

    if not os.path.isfile(csv_path):
        print(f"⚠️ 找不到 CSV：{csv_path}，跳過 {new_ds_name}")
        continue

    # 讀檔 & 基本檢查
    df = pd.read_csv(csv_path)
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.sort_values(by="Timestamp").reset_index(drop=True)
    ensure_cols(df, ["RSRP", "RSRQ", "RLF_II"])

    # 每個資料集各自的彙總
    summary_rows = []

    for window_sec in [1, 2, 3]:
        for future_sec in [1, 2, 3]:
            print(f"\n—— 訓練：前 {window_sec} 秒 → 預測後 {future_sec} 秒 ——")
            X, y = generate_features_and_labels(df, window_sec, future_sec)
            if len(X) == 0:
                print("資料量不足，跳過此組。")
                continue

            idxs = np.arange(X.shape[0])
            train_idx, val_idx = train_test_split(
                idxs, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
            )
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            num_pos = int(y_train.sum())
            num_neg = int(len(y_train) - num_pos)
            scale_pos_weight = num_neg / max(1, num_pos) if num_pos > 0 else 1.0

            model = xgb.XGBClassifier(
                objective="binary:logistic",
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                use_label_encoder=False,
                eval_metric="auc",
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_STATE
            )

            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=False
            )
            evals_result = model.evals_result()

            # 量測推論時間（predict_proba）
            t0 = time.perf_counter()
            y_prob = model.predict_proba(X_val)[:, 1]
            t1 = time.perf_counter()
            infer_time_total_s = t1 - t0
            infer_time_ms_per_sample = (infer_time_total_s / max(1, len(X_val))) * 1000.0

            # 以 F1 最佳尋找閾值
            best_thresh = 0.5
            best_f1 = -1.0
            for t in np.arange(0.1, 0.91, 0.01):
                f1_tmp = f1_score(y_val, (y_prob >= t).astype(int), zero_division=0)
                if f1_tmp > best_f1:
                    best_f1 = f1_tmp
                    best_thresh = t

            y_pred = (y_prob >= best_thresh).astype(int)

            # 指標
            acc  = accuracy_score(y_val, y_pred)
            try:
                auc_val  = roc_auc_score(y_val, y_prob)
            except ValueError:
                auc_val = float("nan")
            prec = precision_score(y_val, y_pred, zero_division=0)
            rec  = recall_score(y_val, y_pred, zero_division=0)
            f1v  = f1_score(y_val, y_pred, zero_division=0)
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=[0,1]).ravel()
            pos_rate = 100.0 * y_val.mean()

            # 估算 FLOPs
            flops_per_sample = estimate_flops_per_sample(model)
            flops_total_val = flops_per_sample * len(X_val)

            print(f"[最佳閾值] 閾值={best_thresh:.2f}, F1={best_f1:.4f}")
            print(f"準確率={acc:.4f}  ROC-AUC={auc_val:.4f}  精確率={prec:.4f}  召回率={rec:.4f}  F1={f1v:.4f}")
            print(f"混淆矩陣：TN:{tn} FP:{fp} FN:{fn} TP:{tp} | 驗證集陽性比例:{pos_rate:.1f}%")
            print(f"推論總時間={infer_time_total_s:.6f}s | 單筆={infer_time_ms_per_sample:.3f}ms | 估算FLOPs/樣本={flops_per_sample:.0f} | 估算FLOPs(驗證集)={flops_total_val:.0f}")

            # 儲存模型
            model_filename = os.path.join(out_dir, f"xgb_win{window_sec}s_pred{future_sec}s.json")
            model.save_model(model_filename)
            print(f"✅ 模型已儲存：{model_filename}")

            # AUC 學習曲線（驗證用虛線）與 ROC 曲線
            plot_auc_curve(evals_result, window_sec, future_sec, out_dir)
            save_roc_curve(y_val, y_prob, window_sec, future_sec, out_dir)

            # 事件預測圖（右圖 5 統計樣式）
            window_size = int(window_sec * SAMPLE_RATE)
            future_window = int(future_sec * SAMPLE_RATE)
            plot_event_window_detections(
                val_idx, y_pred, df, window_size, future_window, window_sec, future_sec, out_dir
            )

            # 加入中文彙總
            summary_rows.append({
                "視窗秒數": window_sec,
                "預測秒數": future_sec,
                "最佳閾值": best_thresh,
                "準確率": acc,
                "ROC-AUC": auc_val,
                "精確率": prec,
                "召回率": rec,
                "F1 分數": f1v,
                "TN(真陰性)": tn, "FP(假陽性)": fp, "FN(假陰性)": fn, "TP(真陽性)": tp,
                "驗證集陽性比例(%)": pos_rate,
                "推論總時間(秒)": infer_time_total_s,
                "推論單筆毫秒": infer_time_ms_per_sample,
                "估算FLOPs/樣本": flops_per_sample,
                "估算FLOPs(驗證集)": flops_total_val
            })

    # 每份資料集各自輸出中文彙總 CSV 到 cnn 夾
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        out_csv = os.path.join(out_dir, "xgb_metrics_summary.csv")
        df_summary.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"\n🧾 已輸出彙總檔（中文欄位）：{out_csv}")

print("\n✅ 全部完成。請到桌面 newxg/newxg01 ~ newxg/newxg04 各自的 cnn 資料夾查看輸出。")
