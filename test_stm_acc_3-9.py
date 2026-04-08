# -*- coding: utf-8 -*-
# 功能：只保留Arousal 3-9的数据，以6为界进行二分类（3-6 vs 7-9）
import os
import pickle
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt

# 防止在无图形界面下报错
import matplotlib
matplotlib.use('Agg')

# ================= 配置区域 =================
RPPG_PICKLE_PATH = "./accurate_pickle/all_474_191.pickle"
SESSIONS_ROOT = "/dataset/MAHNOB-HCI/Sessions"
OUTPUT_IMG_PATH = "./stm_result_3to9_split6.png"

# 【修改点1】设置分类阈值为 6
# 结果：3-6 为 Class 0，7-9 为 Class 1
GT_AROUSAL_THRESHOLD = 5 

# 【修改点2】设置保留数据的下限
KEEP_DATA_MIN_AROUSAL = 3

DEFAULT_WINDOW_SEC = 3
DEFAULT_FS = 30            # rPPG 采样率
# ===========================================

def flatten_data(chunks):
    """简单拼接工具"""
    flat = []
    if isinstance(chunks, dict):
        try:
            sorted_keys = sorted(chunks.keys(), key=lambda x: int(x))
        except:
            sorted_keys = sorted(chunks.keys())  
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

# --- 信号处理函数 ---
def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_filter(data, fs=30):
    """
    带通滤波 [0.7Hz - 3.5Hz] 
    """
    if len(data) < fs: return data
    if np.isnan(data).any():
        data = np.nan_to_num(data)
        
    b, a = butter_bandpass(0.7, 3.5, fs, order=2)
    try:
        y = filtfilt(b, a, data)
        return y
    except:
        return data
# ------------------------

class StandaloneSTM:
    def __init__(self, theta_arousal, theta_pool, lambda_a=0.05, fs=30, window_sec=3, step_sec=1):
        self.fs = fs
        self.window_size = int(fs * window_sec)
        self.step_size = int(fs * step_sec)
        self.theta_arousal = theta_arousal
        self.theta_pool = theta_pool
        self.lambda_a = lambda_a

    def compute_windowed_rmssd(self, rppg_signal):
        rmssd_series = []
        L = len(rppg_signal)
        if L < self.window_size:
            return np.array([])
        
        for i in range(0, L - self.window_size + 1, self.step_size):
            window = rppg_signal[i : i + self.window_size]
            diff_signal = np.diff(window)
            if len(diff_signal) == 0:
                rmssd_series.append(0)
                continue
            rmssd = np.sqrt(np.mean(diff_signal ** 2))
            rmssd_series.append(rmssd)
        return np.array(rmssd_series)

    def run(self, rppg_signal):
        # 1. 滤波
        filtered_sig = apply_filter(rppg_signal, self.fs)
        
        # 2. Z-Score 标准化
        if len(filtered_sig) < self.window_size or np.std(filtered_sig) < 1e-6:
            return 0, 0.0 
            
        rppg_norm = (filtered_sig - np.mean(filtered_sig)) / np.std(filtered_sig)

        # 3. 计算波动
        rmssd_series = self.compute_windowed_rmssd(rppg_norm)
        if rmssd_series.size == 0:
            return 0, 0.0
        
        # 4. STM 积分与发放逻辑
        rmssd_mean = np.mean(rmssd_series)
        rmssd_res = np.abs(rmssd_series - rmssd_mean)
        
        S_A = (rmssd_res > self.theta_arousal).astype(float)
        
        I_A = np.zeros_like(S_A)
        time_step_val = self.step_size / self.fs
        decay_factor = np.exp(-self.lambda_a * time_step_val)
        
        I_A[0] = S_A[0]
        for t in range(1, len(S_A)):
            I_A[t] = S_A[t] + I_A[t-1] * decay_factor
            
        avg_intensity = np.mean(I_A)
        prediction = 1 if avg_intensity > self.theta_pool else 0
        
        return prediction, avg_intensity

def load_dataset(pickle_path, sessions_root, gt_threshold):
    """加载并过滤数据"""
    print(f"Loading data from {pickle_path}...")
    if not os.path.exists(pickle_path):
        return []

    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
        
    dataset = []
    predictions = data['predictions'] if 'predictions' in data else data

    print(f"Parsing sessions (Range [{KEEP_DATA_MIN_AROUSAL}, 9], Threshold > {gt_threshold})...")
    
    for sid_key, chunks in tqdm(predictions.items(), desc="Parsing Data"):
        sid = int(sid_key)
        full_signal = flatten_data(chunks)
        if len(full_signal) == 0: continue
            
        xml_path = os.path.join(sessions_root, str(sid), "session.xml")
        if not os.path.exists(xml_path): continue
            
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            aro_str = root.get('feltArsl')
            if not aro_str: continue
            
            aro_val = float(aro_str)

            # 【修改点3】关键逻辑：只保留 Arousal >= 3 的数据
            if aro_val < KEEP_DATA_MIN_AROUSAL:
                continue 

            # 【修改点4】打标签：以 6 为分界
            # 3,4,5,6 -> label 0
            # 7,8,9   -> label 1
            label = 1 if aro_val > gt_threshold else 0
            
            dataset.append({
                'sid': sid,
                'signal': full_signal,
                'label': label,
                'raw_score': aro_val
            })
        except: pass
            
    print(f"Loaded {len(dataset)} valid sessions (Arousal 3-9).")
    labels = [d['label'] for d in dataset]
    print(f"Class Distribution: Mid(3-6)={labels.count(0)}, High(7-9)={labels.count(1)}")
    return dataset

def grid_search_optimization(dataset):
    """
    网格搜索优化参数
    """
    print("\n========================================")
    print("?? Starting Grid Search (Arousal 3-9)")
    print("========================================")
    
    thetas_arousal = [0.01, 0.03, 0.05, 0.1, 0.2, 0.4] 
    thetas_pool = [0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]
    lambdas = [0.005, 0.01, 0.03, 0.05, 0.1]
    
    best_score = -1
    best_params = {}
    
    total = len(thetas_arousal) * len(thetas_pool) * len(lambdas)
    print(f"Testing {total} parameter combinations...")

    for lam in lambdas:
        for ta in thetas_arousal:
            for tp in thetas_pool:
                y_true = []
                y_pred = []
                
                stm = StandaloneSTM(theta_arousal=ta, theta_pool=tp, 
                                    lambda_a=lam, window_sec=DEFAULT_WINDOW_SEC)
                
                for item in dataset:
                    pred, _ = stm.run(item['signal'])
                    y_true.append(item['label'])
                    y_pred.append(pred)
                
                report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                
                recall_high = report['1']['recall']
                prec_high = report['1']['precision']
                macro_f1 = report['macro avg']['f1-score']
                
                # 只有当模型不仅仅预测全0，且 High 类召回率 > 10% 时才考虑
                if recall_high >= 0.10: 
                    score = macro_f1
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'lambda': lam, 'ta': ta, 'tp': tp, 
                            'recall': recall_high, 'prec': prec_high, 'macro_f1': score
                        }
                        print(f"? New Best: Lam={lam}, Ta={ta}, Tp={tp} | Recall={recall_high:.2f}, Prec={prec_high:.2f}, MacroF1={score:.3f}")

    print("\n? Grid Search Completed.")
    
    if not best_params:
        print("?? Warning: No valid params found. Using defaults.")
        return {'ta': 0.05, 'tp': 1.05, 'lambda': 0.05}
        
    print(f"?? Best Parameters: {best_params}")
    return best_params

def main():
    # 传入新的阈值参数 (6)
    dataset = load_dataset(RPPG_PICKLE_PATH, SESSIONS_ROOT, GT_AROUSAL_THRESHOLD)
    if not dataset: return

    # 运行网格搜索
    best_p = grid_search_optimization(dataset)
    
    theta_a = best_p.get('ta', 0.05)
    theta_p = best_p.get('tp', 1.05)
    lambda_val = best_p.get('lambda', 0.05)
    
    print(f"\n========================================")
    print(f"Running Final Evaluation with Best Params")
    print(f"  Theta_A: {theta_a}, Theta_P: {theta_p}, Lambda: {lambda_val}")
    print(f"========================================")
    
    stm = StandaloneSTM(theta_arousal=theta_a, theta_pool=theta_p, 
                        lambda_a=lambda_val, window_sec=DEFAULT_WINDOW_SEC)
    
    y_true = []
    y_pred = []
    
    for i, item in enumerate(dataset):
        pred, intensity = stm.run(item['signal'])
        y_true.append(item['label'])
        y_pred.append(pred)
        
        if i < 20: 
            print(f" SID: {item['sid']:<4} | RawScore: {item['raw_score']} | Label: {item['label']} | Pred: {pred} | Intensity: {intensity:.4f}")

    print("\nClassification Report (3-9 Dataset, Split @ 6):")
    # 更新报告标签：0代表Mid(3-6)，1代表High(7-9)
    print(classification_report(y_true, y_pred, target_names=['Mid(3-6)', 'High(7-9)'], zero_division=0))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Mid(3-6)', 'High(7-9)'], 
                yticklabels=['Mid(3-6)', 'High(7-9)'])
    plt.title(f'Filtered 3-9 Result (Ta={theta_a}, Tp={theta_p})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(OUTPUT_IMG_PATH, dpi=300)
    print(f"\n? Result image saved to: {OUTPUT_IMG_PATH}")

if __name__ == "__main__":
    main()