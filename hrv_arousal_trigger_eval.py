import os
import pickle
import numpy as np
import xml.etree.ElementTree as ET
from scipy import signal
from glob import glob
from tqdm import tqdm


# ============================================================
#              1. HRV 计算模块（适配任意长度 rPPG）
# ============================================================

def compute_hrv_rmssd(rppg, fs=30):
    """
    输入 rppg: 1D array
    输出 rmssd_ms: 毫秒单位 RMSSD
    """
    # 带通滤波
    b, a = signal.butter(2, [0.75, 2.5], btype='bandpass', fs=fs)
    filtered = signal.filtfilt(b, a, rppg)

    # 峰值检测
    peaks, _ = signal.find_peaks(filtered, distance=int(fs * 0.5))

    if len(peaks) < 3:
        return None

    ibi = np.diff(peaks) / fs  # 秒
    if len(ibi) < 2:
        return None

    rmssd = np.sqrt(np.mean(np.diff(ibi) ** 2))
    return rmssd * 1000     # 转换为 ms


# ============================================================
#              2. HRV 触发器（使用 RMSSD 阈值）
# ============================================================

def classify_arousal_by_rmssd(rmssd_ms,
                              calm_thr=260,
                              low_thr=220):
    """
    rmssd_ms: 毫秒
    阈值来自 MAHNOB 有效191样本统计分布
    """
    if rmssd_ms >= calm_thr:
        return "calm"
    elif rmssd_ms >= low_thr:
        return "low_arousal"
    else:
        return "high_arousal"


def arousal_binary_map(state):
    """HRV 三分类 → 二分类 (Low / High) 与 feltArsl 对齐"""
    if state in ["calm", "low_arousal"]:
        return 0   # Low Arousal
    else:
        return 1   # High Arousal


# ============================================================
#              3. 从 session.xml 读取 arousal groundtruth
# ============================================================

def read_arousal_label(xml_path):
    """读取 MAHNOB session.xml 里的 feltArsl（1~9）"""
    root = ET.parse(xml_path).getroot()
    arsl = int(root.attrib["feltArsl"])
    return 1 if arsl >= 5 else 0   # High Arousal if ≥ 5


# ============================================================
#              4. 主流程：对每个 session 计算 HRV → Arousal
# ============================================================

def evaluate_hrv_arousal(rppg_pickle_path, sessions_root,
                         fs=30, win=180, step=60):
    """
    rPPG 来自 predicted_rPPG（长度 800、900 都支持）

    返回：ACC
    """
    with open(rppg_pickle_path, "rb") as f:
        data = pickle.load(f)

    preds = data["predictions"]
    sessions = sorted(preds.keys())

    results = []

    print(f"共检测到 {len(sessions)} 个 session\n")

    for sid in tqdm(sessions):
        pred_dict = preds[sid]

        # 拼接多个 chunk → 长 rppg
        chunks = [np.asarray(pred_dict[k]).flatten()
                  for k in sorted(pred_dict.keys(), key=lambda x: int(x))]
        rppg = np.concatenate(chunks)

        # 从 MAHNOB session 读取 XML
        session_folder = os.path.join(sessions_root, str(sid))
        xml_path = os.path.join(session_folder, "session.xml")
        if not os.path.exists(xml_path):
            continue

        gt = read_arousal_label(xml_path)  # 0/1 groundtruth

        # --- sliding window ---
        rmssd_list = []
        for start in range(0, len(rppg) - win + 1, step):
            seg = rppg[start:start + win]
            rmssd = compute_hrv_rmssd(seg, fs)
            if rmssd is not None:
                rmssd_list.append(rmssd)

        if len(rmssd_list) == 0:
            pred_bin = 0  # 如果无法检测到峰，默认 low_arousal
        else:
            # 对所有窗口做多数投票
            states = [classify_arousal_by_rmssd(r) for r in rmssd_list]
            bin_states = [arousal_binary_map(s) for s in states]
            pred_bin = int(np.round(np.mean(bin_states)))

        results.append((sid, pred_bin, gt))

    # ---- 计算 ACC ----
    preds = [x[1] for x in results]
    gts = [x[2] for x in results]

    acc = np.mean(np.array(preds) == np.array(gts))
    print("\n============== HRV Arousal Trigger 评估结果 ==============")
    print(f"有效 Session 数: {len(results)}")
    print(f"Arousal ACC = {acc:.4f}")
    print("==========================================================\n")

    return results, acc


# ============================================================
#              5. 主函数
# ============================================================

if __name__ == "__main__":
    rppg_pickle = "./accurate_pickle/accurate.pickle"
    sessions_root = "/dataset/MAHNOB-HCI/Sessions"

    results, acc = evaluate_hrv_arousal(
        rppg_pickle_path=rppg_pickle,
        sessions_root=sessions_root,
        fs=30,
        win=180,   # 6 秒
        step=60    # 每 2 秒滑动一次
    )

