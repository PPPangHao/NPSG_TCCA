import os
import pickle
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from scipy import signal
import torch
from tqdm import tqdm


# ==========================
# 配置
# ==========================
RPPG_PICKLE = "./accurate_pickle/accurate.pickle"
SESSIONS_ROOT = "/dataset/MAHNOB-HCI/Sessions"
OUTPUT_CSV = "./output/session_rmssd_arousal.csv"
FS = 30  # rPPG sampling rate


# ==========================
# HRV 计算
# ==========================
def compute_rmssd_from_rppg(rppg, fs=30):
    if len(rppg) < fs * 6:
        return None

    # bandpass 0.75–2.5 Hz
    b, a = signal.butter(2, [0.75, 2.5], btype="bandpass", fs=fs)
    rppg_f = signal.filtfilt(b, a, rppg)

    # peak detection
    peaks, _ = signal.find_peaks(
        rppg_f,
        distance=int(0.5 * fs),
        prominence=0.1
    )

    if len(peaks) < 3:
        return None

    ibi = np.diff(peaks) / fs  # seconds
    if len(ibi) < 2:
        return None

    rmssd = np.sqrt(np.mean(np.diff(ibi) ** 2))
    return rmssd * 1000.0  # ms


# ==========================
# 读取 arousal label
# ==========================
def read_arousal_label(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    arousal = int(root.attrib.get("feltArsl", -1))
    if arousal < 0:
        return None, None
    label = 1 if arousal >= 5 else 0  # 1=High, 0=Low
    return label, arousal


# ==========================
# 主流程
# ==========================
def main():
    with open(RPPG_PICKLE, "rb") as f:
        data = pickle.load(f)

    predictions = data["predictions"]

    records = []

    print(f"共检测到 {len(predictions)} 个 session\n")

    for session_id in tqdm(predictions.keys()):
        sid = str(session_id)
        session_dir = os.path.join(SESSIONS_ROOT, sid)
        xml_path = os.path.join(session_dir, "session.xml")

        if not os.path.exists(xml_path):
            continue

        # merge rPPG chunks
        merged = []
        for k in sorted(predictions[session_id].keys(), key=lambda x: int(x)):
            sig = predictions[session_id][k]
            if isinstance(sig, torch.Tensor):
                sig = sig.cpu().numpy()
            merged.append(np.asarray(sig).flatten())

        if not merged:
            continue

        rppg = np.concatenate(merged)

        rmssd_ms = compute_rmssd_from_rppg(rppg, FS)
        if rmssd_ms is None:
            continue

        arousal_label, arousal_raw = read_arousal_label(xml_path)
        if arousal_label is None:
            continue

        records.append({
            "session_id": sid,
            "rmssd_ms": rmssd_ms,
            "arousal_label": arousal_label,   # 0=Low, 1=High
            "feltArsl_raw": arousal_raw
        })

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n=========== 导出完成 ===========")
    print(f"有效 session 数: {len(df)}")
    print(f"保存路径: {OUTPUT_CSV}")
    print("\nRMSSD(ms) 描述统计:")
    print(df["rmssd_ms"].describe())


if __name__ == "__main__":
    main()

