import numpy as np
from scipy import signal


class StableHRVArousalExtractor:
    """
    超稳定 HRV → Arousal 分类器
    适用于 rPPG（噪声大、峰值不准、短时窗）
    """

    def __init__(self, fs=30, window_sec=30, overlap_sec=15):
        self.fs = fs
        self.window = int(window_sec * fs)
        self.step = int((window_sec - overlap_sec) * fs)

        # baseline & 阈值
        self.baseline_rmssd = None
        self.baseline_count = 0
        self.calm_ratio = 0.85
        self.low_ratio = 0.65

    # ----------------------------------------------------------------------
    # 核心预处理：平滑滤波 + 去噪
    # ----------------------------------------------------------------------
    def preprocess_signal(self, sig):
        """平滑 rPPG（强稳版）"""
        sig = signal.medfilt(sig, kernel_size=5)
        sig = signal.savgol_filter(sig, 11, 3)
        return sig

    # ----------------------------------------------------------------------
    # 峰值检测（高可靠性版本）
    # ----------------------------------------------------------------------
    def detect_peaks(self, sig):
        """增加显著性过滤，减少假峰值"""
        peaks, props = signal.find_peaks(
            sig,
            distance=int(0.4 * self.fs),  # 心率 ≥ 150 bpm
            prominence=np.std(sig) * 0.3,
            height=np.mean(sig)
        )
        return peaks

    # ----------------------------------------------------------------------
    # 稳定 RMSSD（健壮性增强版）
    # ----------------------------------------------------------------------
    def compute_robust_rmssd(self, ibi):
        """裁剪 5%-95% 区间，提高稳定性"""
        if len(ibi) < 3:
            return None

        ibi = np.array(ibi)
        low, high = np.percentile(ibi, [5, 95])
        ibi = ibi[(ibi >= low) & (ibi <= high)]

        diff = np.diff(ibi)
        rmssd = np.sqrt(np.median(diff ** 2))  # median 代替 mean
        return rmssd

    # ----------------------------------------------------------------------
    # baseline 自适应更新
    # ----------------------------------------------------------------------
    def update_baseline(self, rmssd):
        if rmssd is None:
            return

        if self.baseline_rmssd is None:
            self.baseline_rmssd = rmssd
            self.baseline_count = 1
        else:
            # 平滑更新
            self.baseline_rmssd = 0.97 * self.baseline_rmssd + 0.03 * rmssd
            self.baseline_count += 1

    # ----------------------------------------------------------------------
    # 状态分类
    # ----------------------------------------------------------------------
    def classify(self, rmssd):
        if rmssd is None or self.baseline_rmssd is None:
            return "calibrating"

        calm_th = self.baseline_rmssd * self.calm_ratio
        low_th  = self.baseline_rmssd * self.low_ratio

        if rmssd >= calm_th:
            return "calm"
        elif rmssd >= low_th:
            return "low_arousal"
        else:
            return "high_arousal"

    # ----------------------------------------------------------------------
    # 主流程：输入完整 rPPG，输出每个窗口 arousal
    # ----------------------------------------------------------------------
    def process(self, rppg_signal):
        sig = self.preprocess_signal(rppg_signal)
        N = len(sig)

        results = []

        for start in range(0, N - self.window + 1, self.step):
            end = start + self.window
            window = sig[start:end]

            peaks = self.detect_peaks(window)
            if len(peaks) < 3:
                results.append((start, end, None, "unknown"))
                continue

            # 计算 IBI（秒）
            ibi = np.diff(peaks) / self.fs

            rmssd = self.compute_robust_rmssd(ibi)

            # update baseline
            self.update_baseline(rmssd)

            # classify
            state = self.classify(rmssd)

            results.append((start, end, rmssd, state))

        return results

