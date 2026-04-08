# # -*- coding: utf-8 -*-
# # 功能：基于STM的三分类 (Hierarchical Cascade)
# # Class 0: 1-2 (Low)
# # Class 1: 3-6 (Mid)
# # Class 2: 7-9 (High)

# import os
# import pickle
# import numpy as np
# import xml.etree.ElementTree as ET
# from tqdm import tqdm
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.signal import butter, filtfilt

# import matplotlib
# matplotlib.use('Agg')

# # ================= 配置区域 =================
# RPPG_PICKLE_PATH = "./accurate_pickle/all_474_191.pickle"
# SESSIONS_ROOT = "/dataset/MAHNOB-HCI/Sessions"
# OUTPUT_IMG_PATH = "./stm_3class_cascade_result.png"

# DEFAULT_WINDOW_SEC = 3
# DEFAULT_FS = 30 

# # === 用户提供的最优参数 ===
# # 第一级参数：用于区分 [1-2] vs [3-9]
# PARAMS_STAGE_1 = {'ta': 0.1,  'tp': 0.5, 'lambda': 0.05}

# # 第二级参数：用于区分 [3-6] vs [7-9]
# PARAMS_STAGE_2 = {'ta': 0.05, 'tp': 1.2, 'lambda': 0.1}
# # ===========================================

# # --- 基础工具函数 (保持不变) ---
# def flatten_data(chunks):
#     flat = []
#     if isinstance(chunks, dict):
#         try:
#             sorted_keys = sorted(chunks.keys(), key=lambda x: int(x))
#         except:
#             sorted_keys = sorted(chunks.keys())  
#         for k in sorted_keys:
#             c = chunks[k]
#             c_np = c.numpy() if hasattr(c, 'numpy') else np.array(c)
#             flat.append(c_np.flatten())
#     elif isinstance(chunks, list):
#         for c in chunks:
#             c_np = c.numpy() if hasattr(c, 'numpy') else np.array(c)
#             flat.append(c_np.flatten())
#     else:
#         c_np = chunks.numpy() if hasattr(chunks, 'numpy') else np.array(chunks)
#         flat.append(c_np.flatten())

#     if len(flat) == 0: return np.array([])
#     return np.concatenate(flat)

# def butter_bandpass(lowcut, highcut, fs, order=2):
#     nyq = 0.5 * fs
#     low = lowcut / nyq
#     high = highcut / nyq
#     b, a = butter(order, [low, high], btype='band')
#     return b, a

# def apply_filter(data, fs=30):
#     if len(data) < fs: return data
#     if np.isnan(data).any():
#         data = np.nan_to_num(data)
#     b, a = butter_bandpass(0.7, 3.5, fs, order=2)
#     try:
#         return filtfilt(b, a, data)
#     except:
#         return data

# # --- 核心 STM 模型 ---
# class StandaloneSTM:
#     def __init__(self, theta_arousal, theta_pool, lambda_a, fs=30, window_sec=3, step_sec=1):
#         self.fs = fs
#         self.window_size = int(fs * window_sec)
#         self.step_size = int(fs * step_sec)
#         self.theta_arousal = theta_arousal
#         self.theta_pool = theta_pool
#         self.lambda_a = lambda_a

#     def run(self, rppg_signal):
#         # 1. 预处理
#         filtered_sig = apply_filter(rppg_signal, self.fs)
#         if len(filtered_sig) < self.window_size or np.std(filtered_sig) < 1e-6:
#             return 0, 0.0
        
#         # 2. 标准化
#         rppg_norm = (filtered_sig - np.mean(filtered_sig)) / np.std(filtered_sig)

#         # 3. 计算波动 RMSSD
#         rmssd_series = []
#         L = len(rppg_norm)
#         for i in range(0, L - self.window_size + 1, self.step_size):
#             window = rppg_norm[i : i + self.window_size]
#             diff_signal = np.diff(window)
#             if len(diff_signal) == 0: val = 0
#             else: val = np.sqrt(np.mean(diff_signal ** 2))
#             rmssd_series.append(val)
        
#         rmssd_series = np.array(rmssd_series)
#         if rmssd_series.size == 0: return 0, 0.0
        
#         # 4. 积分发放
#         rmssd_mean = np.mean(rmssd_series)
#         rmssd_res = np.abs(rmssd_series - rmssd_mean)
        
#         S_A = (rmssd_res > self.theta_arousal).astype(float)
#         I_A = np.zeros_like(S_A)
        
#         time_step_val = self.step_size / self.fs
#         decay_factor = np.exp(-self.lambda_a * time_step_val)
        
#         I_A[0] = S_A[0]
#         for t in range(1, len(S_A)):
#             I_A[t] = S_A[t] + I_A[t-1] * decay_factor
            
#         avg_intensity = np.mean(I_A)
#         prediction = 1 if avg_intensity > self.theta_pool else 0
        
#         return prediction, avg_intensity

# # --- 级联分类器 (Cascade Classifier) ---
# class CascadeSTMClassifier:
#     def __init__(self, params_stage1, params_stage2):
#         # Stage 1: 负责区分 [1-2] 和 [3-9]
#         self.stm1 = StandaloneSTM(
#             theta_arousal=params_stage1['ta'],
#             theta_pool=params_stage1['tp'],
#             lambda_a=params_stage1['lambda']
#         )
#         # Stage 2: 负责区分 [3-6] 和 [7-9]
#         self.stm2 = StandaloneSTM(
#             theta_arousal=params_stage2['ta'],
#             theta_pool=params_stage2['tp'],
#             lambda_a=params_stage2['lambda']
#         )
    
#     def predict(self, signal):
#         # === 第一级判断 ===
#         # 这里的 output 0 代表 "Low/Resting" (Class 0: 1-2)
#         # 这里的 output 1 代表 "High/Active" (Class 1 or 2)
#         pred1, _ = self.stm1.run(signal)
        
#         if pred1 == 0:
#             return 0  # 直接归为 Low Arousal (1-2)
        
#         # === 第二级判断 ===
#         # 如果第一级认为是 Active，则进入第二级细分
#         # 这里的 output 0 代表 "相对较低" (Class 1: 3-6)
#         # 这里的 output 1 代表 "相对较高" (Class 2: 7-9)
#         pred2, _ = self.stm2.run(signal)
        
#         if pred2 == 0:
#             return 1  # Mid Arousal (3-6)
#         else:
#             return 2  # High Arousal (7-9)

# def load_3class_dataset(pickle_path, sessions_root):
#     """加载数据并分配三分类标签"""
#     print(f"Loading data from {pickle_path}...")
#     if not os.path.exists(pickle_path): return []

#     with open(pickle_path, 'rb') as f:
#         data = pickle.load(f)
        
#     dataset = []
#     predictions = data['predictions'] if 'predictions' in data else data

#     print("Parsing sessions into 3 Classes...")
#     for sid_key, chunks in tqdm(predictions.items(), desc="Parsing Data"):
#         sid = int(sid_key)
#         full_signal = flatten_data(chunks)
#         if len(full_signal) == 0: continue
            
#         xml_path = os.path.join(sessions_root, str(sid), "session.xml")
#         if not os.path.exists(xml_path): continue
            
#         try:
#             tree = ET.parse(xml_path)
#             root = tree.getroot()
#             aro_str = root.get('feltArsl')
#             if not aro_str: continue
            
#             aro_val = float(aro_str)
            
#             # === 三分类标签逻辑 ===
#             if aro_val <= 2:
#                 label = 0 # Low (1-2)
#             elif aro_val <= 6:
#                 label = 1 # Mid (3-6)
#             else:
#                 label = 2 # High (7-9)
            
#             dataset.append({
#                 'sid': sid,
#                 'signal': full_signal,
#                 'label': label,
#                 'raw_score': aro_val
#             })
#         except: pass
            
#     # 统计分布
#     labels = [d['label'] for d in dataset]
#     print(f"Total Samples: {len(dataset)}")
#     print(f"Class 0 (1-2): {labels.count(0)}")
#     print(f"Class 1 (3-6): {labels.count(1)}")
#     print(f"Class 2 (7-9): {labels.count(2)}")
#     return dataset

# def main():
#     # 1. 加载数据
#     dataset = load_3class_dataset(RPPG_PICKLE_PATH, SESSIONS_ROOT)
#     if not dataset: return

#     # 2. 初始化级联分类器
#     cascade_model = CascadeSTMClassifier(PARAMS_STAGE_1, PARAMS_STAGE_2)
    
#     print(f"\n========================================")
#     print(f"Running Cascade 3-Class Evaluation")
#     print(f" Stage 1 Params: {PARAMS_STAGE_1}")
#     print(f" Stage 2 Params: {PARAMS_STAGE_2}")
#     print(f"========================================")
    
#     y_true = []
#     y_pred = []
    
#     # 3. 推理
#     for item in tqdm(dataset, desc="Evaluating"):
#         pred = cascade_model.predict(item['signal'])
#         y_true.append(item['label'])
#         y_pred.append(pred)
        
#     # 4. 评估指标
#     acc = accuracy_score(y_true, y_pred)
#     print(f"\n>>> Overall Accuracy: {acc*100:.2f}%")
    
#     class_names = ['Low (1-2)', 'Mid (3-6)', 'High (7-9)']
#     print("\nClassification Report:")
#     print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    
#     # 5. 绘制混淆矩阵
#     cm = confusion_matrix(y_true, y_pred)
#     plt.figure(figsize=(7,6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
#                 xticklabels=class_names, 
#                 yticklabels=class_names)
#     plt.title(f'3-Class Cascade Result (Acc: {acc:.3f})')
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.tight_layout()
#     plt.savefig(OUTPUT_IMG_PATH, dpi=300)
#     print(f"\n? Result image saved to: {OUTPUT_IMG_PATH}")

# if __name__ == "__main__":
#     main()

# -*- coding: gbk -*-
# 功能：自动搜索最优参数的级联三分类 (Auto-Optimized 3-Class Cascade)
# Class 0: 1-2 (Low)
# Class 1: 3-6 (Mid)
# Class 2: 7-9 (High)

import os
import pickle
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt

# 防止无图形界面报错
import matplotlib
matplotlib.use('Agg')

# ================= 配置区域 =================
RPPG_PICKLE_PATH = "./accurate_pickle/all_474_191.pickle"
SESSIONS_ROOT = "/dataset/MAHNOB-HCI/Sessions"
OUTPUT_IMG_PATH = "./stm_3class_auto_optimized.png"

DEFAULT_WINDOW_SEC = 3
DEFAULT_FS = 30 

# === 搜索空间配置 (根据之前的经验设置范围) ===
SEARCH_SPACE = {
    'ta': [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3],   # 激活阈值
    'tp': [0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0],       # 池化/发放阈值
    'lambda': [0.01, 0.05, 0.1, 0.5]                 # 衰减因子
}
# ===========================================

# --- 基础工具函数 ---
def flatten_data(chunks):
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

def apply_filter(data, fs=30):
    if len(data) < fs: return data
    if np.isnan(data).any(): data = np.nan_to_num(data)
    nyq = 0.5 * fs
    b, a = butter(2, [0.7/nyq, 3.5/nyq], btype='band')
    try: return filtfilt(b, a, data)
    except: return data

# --- STM 模型 ---
class StandaloneSTM:
    def __init__(self, theta_arousal, theta_pool, lambda_a, fs=30, window_sec=3, step_sec=1):
        self.fs = fs
        self.window_size = int(fs * window_sec)
        self.step_size = int(fs * step_sec)
        self.theta_arousal = theta_arousal
        self.theta_pool = theta_pool
        self.lambda_a = lambda_a

    def run(self, rppg_signal):
        filtered_sig = apply_filter(rppg_signal, self.fs)
        if len(filtered_sig) < self.window_size or np.std(filtered_sig) < 1e-6:
            return 0
        
        rppg_norm = (filtered_sig - np.mean(filtered_sig)) / np.std(filtered_sig)

        rmssd_series = []
        L = len(rppg_norm)
        for i in range(0, L - self.window_size + 1, self.step_size):
            window = rppg_norm[i : i + self.window_size]
            diff = np.diff(window)
            val = np.sqrt(np.mean(diff ** 2)) if len(diff) > 0 else 0
            rmssd_series.append(val)
        
        rmssd_series = np.array(rmssd_series)
        if rmssd_series.size == 0: return 0
        
        rmssd_mean = np.mean(rmssd_series)
        S_A = (np.abs(rmssd_series - rmssd_mean) > self.theta_arousal).astype(float)
        
        I_A = np.zeros_like(S_A)
        decay = np.exp(-self.lambda_a * (self.step_size / self.fs))
        
        I_A[0] = S_A[0]
        for t in range(1, len(S_A)):
            I_A[t] = S_A[t] + I_A[t-1] * decay
            
        return 1 if np.mean(I_A) > self.theta_pool else 0

# --- 级联分类器 ---
class CascadeSTMClassifier:
    def __init__(self, params_stage1, params_stage2):
        self.stm1 = StandaloneSTM(params_stage1['ta'], params_stage1['tp'], params_stage1['lambda'])
        self.stm2 = StandaloneSTM(params_stage2['ta'], params_stage2['tp'], params_stage2['lambda'])
    
    def predict(self, signal):
        # Stage 1: 判断是否为 Low (0)
        if self.stm1.run(signal) == 0:
            return 0 
        # Stage 2: 判断是 Mid (1) 还是 High (2)
        if self.stm2.run(signal) == 0:
            return 1
        else:
            return 2

# --- 自动搜索逻辑 ---
def optimize_stage(dataset, target_labels, search_space, task_name="Stage"):
    """
    通用网格搜索函数
    dataset: 包含 'signal' 和 'label' 的列表
    target_labels: 一个字典，定义二分类转换逻辑。例如 {0:0, 1:1, 2:1} 把原始标签转为二分类
    """
    print(f"\n>>> Optimizing {task_name}...")
    best_score = -1
    best_params = {}
    
    # 准备二分类真值
    binary_y_true = []
    valid_indices = [] # 记录参与此阶段训练的数据索引
    
    for i, item in enumerate(dataset):
        orig_label = item['label']
        if orig_label in target_labels:
            binary_y_true.append(target_labels[orig_label])
            valid_indices.append(i)
            
    if not valid_indices:
        print("Error: No valid data for this stage.")
        return {'ta': 0.05, 'tp': 1.0, 'lambda': 0.05}

    total_combs = len(search_space['ta']) * len(search_space['tp']) * len(search_space['lambda'])
    
    # 开始搜索
    with tqdm(total=total_combs, desc=f"{task_name} Grid Search") as pbar:
        for lam in search_space['lambda']:
            for ta in search_space['ta']:
                for tp in search_space['tp']:
                    
                    stm = StandaloneSTM(theta_arousal=ta, theta_pool=tp, lambda_a=lam)
                    binary_y_pred = []
                    
                    for idx in valid_indices:
                        pred = stm.run(dataset[idx]['signal'])
                        binary_y_pred.append(pred)
                    
                    # 使用 Macro F1 作为优化目标
                    score = f1_score(binary_y_true, binary_y_pred, average='macro', zero_division=0)
                    
                    if score > best_score:
                        best_score = score
                        best_params = {'ta': ta, 'tp': tp, 'lambda': lam}
                    
                    pbar.update(1)
    
    print(f"   Best {task_name} F1: {best_score:.4f} | Params: {best_params}")
    return best_params

def load_3class_dataset(pickle_path, sessions_root):
    if not os.path.exists(pickle_path): return []
    with open(pickle_path, 'rb') as f: data = pickle.load(f)
    dataset = []
    preds = data['predictions'] if 'predictions' in data else data
    
    for sid_key, chunks in preds.items():
        sid = int(sid_key)
        sig = flatten_data(chunks)
        if len(sig)==0: continue
        
        xml_path = os.path.join(sessions_root, str(sid), "session.xml")
        if not os.path.exists(xml_path): continue
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            val = float(root.get('feltArsl'))
            
            # 0: Low(1-2), 1: Mid(3-6), 2: High(7-9)
            if val <= 2: label = 0
            elif val <= 6: label = 1
            else: label = 2
            
            dataset.append({'sid': sid, 'signal': sig, 'label': label})
        except: pass
    return dataset

def main():
    # 1. 加载数据
    dataset = load_3class_dataset(RPPG_PICKLE_PATH, SESSIONS_ROOT)
    if not dataset: 
        print("Data load failed.")
        return
    
    print(f"Loaded {len(dataset)} samples.")

    # 2. 自动搜索 Stage 1 参数
    # 逻辑：将 Class 0 视为负例(0)，Class 1 和 2 视为正例(1)
    stage1_mapping = {0: 0, 1: 1, 2: 1}
    best_p1 = optimize_stage(dataset, stage1_mapping, SEARCH_SPACE, task_name="Stage 1 (Low vs Active)")
    
    # 3. 自动搜索 Stage 2 参数
    # 逻辑：仅使用 Class 1 和 2 的数据。Class 1 视为(0), Class 2 视为(1)
    # Class 0 的数据不参与 Stage 2 的参数优化
    stage2_mapping = {1: 0, 2: 1} 
    best_p2 = optimize_stage(dataset, stage2_mapping, SEARCH_SPACE, task_name="Stage 2 (Mid vs High)")
    
    # 4. 使用最优参数构建最终模型
    print(f"\n========================================")
    print(f"Building Final Cascade Model")
    print(f" Stage 1 (0 vs 1,2): {best_p1}")
    print(f" Stage 2 (1 vs 2)  : {best_p2}")
    print(f"========================================")
    
    cascade_model = CascadeSTMClassifier(best_p1, best_p2)
    
    # 5. 全局评估
    y_true = []
    y_pred = []
    
    for item in tqdm(dataset, desc="Final Evaluation"):
        pred = cascade_model.predict(item['signal'])
        y_true.append(item['label'])
        y_pred.append(pred)
        
    acc = accuracy_score(y_true, y_pred)
    class_names = ['Low (1-2)', 'Mid (3-6)', 'High (7-9)']
    
    print(f"\n>>> Final 3-Class Accuracy: {acc*100:.2f}%")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    
    # 绘图
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Auto-Optimized 3-Class Result (Acc: {acc:.3f})')
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG_PATH, dpi=300)
    print(f"Saved plot to {OUTPUT_IMG_PATH}")

if __name__ == "__main__":
    main()