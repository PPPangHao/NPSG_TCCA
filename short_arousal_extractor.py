import numpy as np
import pickle
import scipy.signal as signal
import csv
import os


class ShortRRPG_ArousalExtractor:
    def __init__(self, fs=30):
        self.fs = fs

    # ---------------------------
    # 读取并合并 rPPG
    # ---------------------------
    def load_rppg_pickle(self, pickle_path):
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)

        predictions = data["predictions"]
        sessions = {}

        for sid, chunks in predictions.items():
            merged = []
            for k in sorted(chunks.keys(), key=lambda x: int(x)):
                arr = chunks[k]
                arr = arr.cpu().numpy() if hasattr(arr, "cpu") else np.array(arr)
                merged.append(arr.flatten())

            sessions[sid] = np.concatenate(merged) if merged else np.array([])

        return sessions

    # ---------------------------
    # 峰检测
    # ---------------------------
    def detect_peaks(self, sig):
        peaks, _ = signal.find_peaks(sig, distance=int(self.fs * 0.4))
        return peaks

    # ---------------------------
    # 提取短序列 Arousal 特征
    # ---------------------------
    def extract_features(self, sig):
        if len(sig) < 100:
            return None

        # 1) 峰检测
        peaks = self.detect_peaks(sig)
        if len(peaks) < 2:
            return None

        ibi = np.diff(peaks) / self.fs  # 秒
        hr = 60 / ibi  # BPM

        # ---- 心率特征 ----
        mean_HR = np.mean(hr)
        std_HR = np.std(hr)
        hr_slope = (hr[-1] - hr[0]) / len(hr)  # 趋势

        # ---- 短 RMSSD ----
        successive_diffs = np.diff(ibi)
        rmssd_short = np.sqrt(np.mean(successive_diffs ** 2))

        # ---- 能量特征（心率带宽）----
        b, a = signal.butter(2, [0.75/(self.fs/2), 2.5/(self.fs/2)], btype="bandpass")
        filtered = signal.filtfilt(b, a, sig)

        band_energy = np.sum(filtered ** 2)
        band_ratio = np.mean(np.abs(filtered))

        return {
            "mean_HR": mean_HR,
            "std_HR": std_HR,
            "hr_slope": hr_slope,
            "ibi_mean": np.mean(ibi),
            "ibi_std": np.std(ibi),
            "rmssd_short": rmssd_short,
            "band_energy": band_energy,
            "band_ratio": band_ratio,
        }

    # ---------------------------
    # 简洁 Arousal 分类器
    # ---------------------------
    def classify_arousal(self, feat):
        if feat is None:
            return "unknown"

        # 阈值可调
        if feat["mean_HR"] > 85 and feat["std_HR"] < 10:
            return "high"

        if feat["mean_HR"] < 70 and feat["std_HR"] < 8:
            return "low"

        return "calm"

    # ---------------------------
    # 批量处理并输出 CSV
    # ---------------------------
    def process(self, pickle_path, output_csv):
        sessions = self.load_rppg_pickle(pickle_path)

        os.makedirs(os.path.dirname(output_csv), exist_ok=True)

        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "session_id", "arousal_label", "mean_HR", "std_HR",
                "hr_slope", "ibi_mean", "ibi_std", "rmssd_short",
                "band_energy", "band_ratio"
            ])

            for sid, sig in sessions.items():
                feat = self.extract_features(sig)
                label = self.classify_arousal(feat)

                if feat is None:
                    writer.writerow([sid, "unknown"] + [""]*8)
                else:
                    writer.writerow([
                        sid, label,
                        feat["mean_HR"],
                        feat["std_HR"],
                        feat["hr_slope"],
                        feat["ibi_mean"],
                        feat["ibi_std"],
                        feat["rmssd_short"],
                        feat["band_energy"],
                        feat["band_ratio"],
                    ])

        print(f"✔ Arousal results saved to: {output_csv}")


pickle_path = "./accurate_pickle/accurate.pickle"
output_csv = "./arousal_results/arousal.csv"

extractor = ShortRRPG_ArousalExtractor(fs=30)
extractor.process(pickle_path, output_csv)

