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
OUTPUT_IMG_PATH = "./intensity_debug_plot.png"

# ⚠️ 临时把阈值调极低，看看能不能把信号"逼"出来
THETA_AROUSAL = 0.02  # 之前是 0.05
LAMBDA_A = 0.05
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
        # Z-Score
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
            
        return np.mean(I_A) * 10.0 # 放大10倍

def main():
    print(f"Loading data from {RPPG_PICKLE_PATH}...")
    with open(RPPG_PICKLE_PATH, 'rb') as f:
        data = pickle.load(f)
    predictions = data['predictions'] if 'predictions' in data else data

    stm = SimpleSTM(THETA_AROUSAL, LAMBDA_A, window_sec=WINDOW_SEC)
    intensities = {0: [], 1: [], 2: []}
    
    print("Calculating intensities...")
    count_processed = 0
    for sid_key, chunks in tqdm(predictions.items()):
        try:
            sid = int(sid_key)
            full_signal = flatten_data(chunks)
            if len(full_signal) == 0: continue
            
            # XML 读取
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
            
            val = stm.get_intensity(full_signal)
            intensities[label].append(val)
            count_processed += 1
        except Exception as e:
            pass

    print(f"\nProcessed {count_processed} sessions.")

    # === 关键：打印每个类别的统计数据 ===
    all_data = []
    all_labels = []
    
    print("\n" + "="*40)
    print("DATA DIAGNOSIS (Check this!)")
    print("="*40)
    
    for label, name in [(0, 'Low'), (1, 'Mid'), (2, 'High')]:
        vals = np.array(intensities[label])
        if len(vals) == 0:
            print(f"⚠️ Class {name} has NO DATA!")
            continue
            
        zeros = np.sum(vals == 0)
        non_zeros = np.sum(vals > 0)
        max_val = np.max(vals)
        p75 = np.percentile(vals, 75)
        
        print(f"Class {name}: Total={len(vals)}")
        print(f"   -> Zeros: {zeros} ({zeros/len(vals)*100:.1f}%)")
        print(f"   -> Max Value: {max_val:.4f}")
        print(f"   -> 75th Percentile: {p75:.4f}")
        
        if p75 == 0:
            print("   ⚠️ WARNING: 75% of data is 0. Boxplot will be flat.")
            
        all_data.extend(vals)
        all_labels.extend([name] * len(vals))
        print("-" * 30)

    # === 绘图 ===
    plt.figure(figsize=(8, 6))
    
    # 1. 散点图 (Jitter Plot)：强制显示每个数据点
    sns.stripplot(x=all_labels, y=all_data, jitter=True, alpha=0.5, color='black', size=3)
    
    # 2. 箱线图 (Boxplot)：放在上面
    sns.boxplot(x=all_labels, y=all_data, showfliers=False, boxprops=dict(alpha=0.3))
    
    plt.title(f'Intensity Diagnosis (Theta={THETA_AROUSAL})')
    plt.ylabel('Calculated Intensity (x10)')
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.savefig(OUTPUT_IMG_PATH, dpi=300)
    print(f"\n✅ Diagnostic plot saved to: {OUTPUT_IMG_PATH}")

if __name__ == "__main__":
    main()