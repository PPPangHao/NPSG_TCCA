import os
import pickle
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')

# ================= 配置 =================
RPPG_PICKLE_PATH = "./accurate_pickle/accurate.pickle"
SESSIONS_ROOT = "/dataset/MAHNOB-HCI/Sessions"
OUTPUT_IMG_PATH = "./theta_scan_result.png"

# 扫描范围：从 0.02 (全噪音) 到 0.15 (极严苛)
THETA_CANDIDATES = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]

# 固定 Lambda
FIXED_LAMBDA = 0.05
WINDOW_SEC = 3

GT_TH_LOW = 3.5  
GT_TH_HIGH = 6.5 
# =======================================

def flatten_data(chunks):
    flat = []
    if isinstance(chunks, dict):
        try: sorted_keys = sorted(chunks.keys(), key=lambda x: int(x))
        except: sorted_keys = sorted(chunks.keys())  
        for k in sorted_keys:
            c = chunks[k]
            c_np = c.numpy() if hasattr(c, 'numpy') else np.array(c)
            flat.append(c_np.flatten())
    elif isinstance(chunks, list):
        for c in chunks:
            c_np = c.numpy() if hasattr(c, 'numpy') else np.array(c)
            flat.append(c_np.flatten())
    else:
        c_np = chunks.numpy() if hasattr(chunks, 'numpy') else np.array(chunks)
        flat.append(c_np.flatten())
    if len(flat) == 0: return np.array([])
    return np.concatenate(flat)

class SimpleSTM:
    def __init__(self, theta_arousal, lambda_a, fs=30, window_sec=3):
        self.fs = fs
        self.window_size = int(fs * window_sec)
        self.step_size = int(fs)
        self.theta_arousal = theta_arousal
        self.lambda_a = lambda_a

    def get_intensity(self, rppg_signal):
        if len(rppg_signal) < 2 or np.std(rppg_signal) < 1e-6: return 0.0
        rppg_norm = (rppg_signal - np.mean(rppg_signal)) / np.std(rppg_signal)

        rmssd_series = []
        L = len(rppg_norm)
        if L < self.window_size: return 0.0
        
        for i in range(0, L - self.window_size + 1, self.step_size):
            window = rppg_norm[i : i + self.window_size]
            diff = np.diff(window)
            if len(diff) == 0: 
                rmssd_series.append(0)
            else:
                rmssd_series.append(np.sqrt(np.mean(diff ** 2)))
        
        if not rmssd_series: return 0.0
        
        rmssd_series = np.array(rmssd_series)
        rmssd_mean = np.mean(rmssd_series)
        rmssd_res = np.abs(rmssd_series - rmssd_mean)
        
        # 线性加权逻辑
        S_A = np.maximum(0, rmssd_res - self.theta_arousal)
        
        I_A = np.zeros_like(S_A)
        decay = np.exp(-self.lambda_a * (self.step_size / self.fs))
        
        I_A[0] = S_A[0]
        for t in range(1, len(S_A)):
            I_A[t] = S_A[t] + I_A[t-1] * decay
            
        return np.mean(I_A) * 10.0

def main():
    print("Loading dataset...")
    with open(RPPG_PICKLE_PATH, 'rb') as f:
        data = pickle.load(f)
    predictions = data['predictions'] if 'predictions' in data else data

    # 1. 预加载所有 Session 的 Signal 和 Label，避免重复解析 XML
    print("Parsing sessions...")
    dataset_cache = []
    
    for sid_key, chunks in tqdm(predictions.items()):
        try:
            sid = int(sid_key)
            full_signal = flatten_data(chunks)
            if len(full_signal) == 0: continue
            
            xml_path = os.path.join(SESSIONS_ROOT, str(sid), "session.xml")
            if not os.path.exists(xml_path): continue
            tree = ET.parse(xml_path)
            root = tree.getroot()
            aro_str = root.get('feltArsl')
            if not aro_str: continue
            
            aro_val = float(aro_str)
            if aro_val <= GT_TH_LOW: label = 0
            elif aro_val >= GT_TH_HIGH: label = 2
            else: label = 1
            
            dataset_cache.append({'signal': full_signal, 'label': label})
        except: pass

    print(f"Cached {len(dataset_cache)} sessions.")

    # 2. 扫描 Theta
    results_mean = {'theta': [], 'low': [], 'mid': [], 'high': [], 'gap': []}
    
    print("\nScanning Thetas...")
    for theta in THETA_CANDIDATES:
        stm = SimpleSTM(theta, FIXED_LAMBDA, window_sec=WINDOW_SEC)
        
        intensities = {0: [], 1: [], 2: []}
        for item in dataset_cache:
            val = stm.get_intensity(item['signal'])
            intensities[item['label']].append(val)
            
        mean_low = np.mean(intensities[0]) if intensities[0] else 0
        mean_mid = np.mean(intensities[1]) if intensities[1] else 0
        mean_high = np.mean(intensities[2]) if intensities[2] else 0
        
        # 关键指标：High 和 Low 的差距
        gap = mean_high - mean_low
        
        results_mean['theta'].append(theta)
        results_mean['low'].append(mean_low)
        results_mean['mid'].append(mean_mid)
        results_mean['high'].append(mean_high)
        results_mean['gap'].append(gap)
        
        print(f"Theta={theta:.2f} | Low={mean_low:.4f}, Mid={mean_mid:.4f}, High={mean_high:.4f} | Gap={gap:.4f}")

    # 3. 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(results_mean['theta'], results_mean['low'], 'g-o', label='Low (Mean)')
    plt.plot(results_mean['theta'], results_mean['mid'], 'b-o', label='Mid (Mean)')
    plt.plot(results_mean['theta'], results_mean['high'], 'r-o', label='High (Mean)')
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    plt.title('Signal Intensity vs Theta (Finding the Sweet Spot)')
    plt.xlabel('Theta Arousal (Sensitivity Threshold)')
    plt.ylabel('Mean Intensity')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(OUTPUT_IMG_PATH, dpi=300)
    print(f"\n✅ Scan plot saved to: {OUTPUT_IMG_PATH}")
    
    # 自动推荐
    best_idx = np.argmax(results_mean['gap'])
    best_theta = results_mean['theta'][best_idx]
    print(f"\n🔥 Recommendation: Best Theta seems to be {best_theta:.2f} (Max Gap: {results_mean['gap'][best_idx]:.4f})")

if __name__ == "__main__":
    main()
