import numpy as np
import pickle
from scipy import signal
from collections import deque


# ============================================================
# AROUSAL 触发器（基于 RMSSD_ms）
# ============================================================

class ArousalTriggerHRV:
    """
    稳定版 Arousal Trigger – 使用 RMSSD(ms) 做 arousal 三分类：
        - high_arousal    : RMSSD < 230 ms
        - low_arousal     : 230–270 ms
        - calm            : RMSSD >= 270 ms

    这些阈值来自你统计的 191 个 MAHNOB-HCI 有效 rPPG 样本。
    """

    def __init__(self, fs=30, win_size=900, step=300,
                 low_thr_ms=230.0, calm_thr_ms=270.0, debug=False):

        self.fs = fs                        # rPPG 采样率（30 Hz）
        self.win_size = win_size            # 30s  → 30 * 30 = 900 个点
        self.step = step                    # 可调整滑动步长
        self.low_thr_ms = low_thr_ms        # 230ms（你统计的 33% 分位）
        self.calm_thr_ms = calm_thr_ms      # 270ms（你的 66% 分位）
        self.debug = debug

        self.last_states = deque(maxlen=10)  # 最近状态

    # ------------------------------------------------------------
    # 滤波 + 峰检测 → 计算 RMSSD / SDNN / 频域 LF/HF
    # ------------------------------------------------------------
    def compute_hrv_features(self, signal_raw):

        # 1) 带通滤波 (0.75–2.5 Hz)
        b, a = signal.butter(2, [0.75, 2.5], btype='bandpass', fs=self.fs)
        filtered = signal.filtfilt(b, a, signal_raw)

        # 2) 峰值检测（心跳周期）
        peaks, properties = signal.find_peaks(
            filtered,
            distance=int(self.fs * 0.5),  # 允许最短心率 ~120bpm
            height=np.mean(filtered) + 0.3 * np.std(filtered)
        )

        if len(peaks) < 3:
            return None, None, None

        # 3) 计算 IBI（秒）
        ibi = np.diff(peaks) / self.fs
        if len(ibi) < 2:
            return None, None, None

        # 4) RMSSD
        rmssd = np.sqrt(np.mean(np.diff(ibi) ** 2))  # 秒
        rmssd_ms = rmssd * 1000

        # 5) SDNN
        sdnn = np.std(ibi)
        sdnn_ms = sdnn * 1000

        # 6) LF/HF（可选，如果太短可能为0）
        lf_hf_ratio = 0

        return rmssd_ms, sdnn_ms, lf_hf_ratio

    # ------------------------------------------------------------
    # 通过 RMSSD_ms 判断 arousal 三类别
    # ------------------------------------------------------------
    def determine_state(self, rmssd_ms):

        if rmssd_ms is None:
            return "unknown"

        if rmssd_ms >= self.calm_thr_ms:
            return "calm"

        elif rmssd_ms >= self.low_thr_ms:
            return "low_arousal"

        else:
            return "high_arousal"

    # ------------------------------------------------------------
    # 滑动窗口分析的主函数
    # rppg_signal: 长度 900（30s）
    # ------------------------------------------------------------
    def analyze(self, rppg_signal):

        results = []

        for start in range(0, len(rppg_signal) - self.win_size + 1, self.step):
            end = start + self.win_size
            window = rppg_signal[start:end]

            rmssd_ms, sdnn_ms, lf_hf = self.compute_hrv_features(window)

            state = self.determine_state(rmssd_ms)

            results.append({
                "start": start,
                "end": end,
                "rmssd_ms": rmssd_ms,
                "sdnn_ms": sdnn_ms,
                "lf_hf": lf_hf,
                "state": state,
            })

            self.last_states.append(state)

        return results

    # ------------------------------------------------------------
    # 给 Multimodal 模型用的接口
    # 返回一个最终 arousal 状态（窗口投票）
    # ------------------------------------------------------------
    def final_arousal_state(self):

        if len(self.last_states) == 0:
            return "unknown"

        # 投票
        return max(set(self.last_states), key=self.last_states.count)


# ============================================================
# 示例：处理一个 session 的 predicted_rPPG
# ============================================================

if __name__ == "__main__":

    # 加载你的 pickle
    PICKLE_PATH = "./accurate_pickle/accurate.pickle"
    data = pickle.load(open(PICKLE_PATH, "rb"))

    session = list(data["predictions"].keys())[0]
    print("Using session:", session)

    # 合并 chunks → 长度 900 （30s）
    chunks = data["predictions"][session]
    rppg = np.concatenate([
        np.array(chunks[k]).flatten()
        for k in sorted(chunks.keys(), key=lambda x: int(x))
    ])

    print("Loaded rPPG len =", len(rppg))

    # 创建 Arousal Trigger
    hrv_trigger = ArousalTriggerHRV(
        fs=30,
        win_size=900,      # 30s
        step=300,          # 每 10s 触发一次
        low_thr_ms=230.0,
        calm_thr_ms=270.0,
        debug=False
    )

    # 分析
    results = hrv_trigger.analyze(rppg)

    print("\n=== Arousal 分析结果 ===")
    for idx, r in enumerate(results):
        print(f"[{idx}] RMSSD={r['rmssd_ms']:.1f}ms  →  {r['state']}")

    print("\n最终（投票）Arousal 状态 =", hrv_trigger.final_arousal_state())
