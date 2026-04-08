# -*- coding: utf-8 -*-
# ==============================================================================
# 完整集成代码：rPPG信号处理 + STM模型 + 高速网格搜索 + 敏感性分析(Study 4)
# ==============================================================================
import os
import pickle
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt

# --- 配置 Matplotlib 后端 (防止服务器无GUI报错) ---
import matplotlib
matplotlib.use('Agg')

# ================= 配置区域 (请根据实际路径修改) =================
RPPG_PICKLE_PATH = "./accurate_pickle/all_474_191.pickle"  # 输入数据路径
SESSIONS_ROOT = "/dataset/MAHNOB-HCI/Sessions"             # GT标签路径
OUTPUT_IMG_PATH = "./stm_best_result_matrix.png"            # 最终混淆矩阵输出
STUDY4_IMG_PATH = "./study4_sensitivity_curve.png"          # Study 4 曲线输出

GT_AROUSAL_THRESHOLD = 3   # 标签二值化阈值 (1-9分制)
DEFAULT_WINDOW_SEC = 3     # 滑动窗口大小
DEFAULT_FS = 30            # 采样率
# ===============================================================

# ---------------------------------------------------------------
# 1. 信号处理基础工具
# ---------------------------------------------------------------
def flatten_data(chunks):
    """将分段的Tensor或Array拼接成一维信号"""
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

def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_filter(data, fs=30):
    """
    [关键] 带通滤波: 0.7Hz - 3.5Hz (对应 42-210 BPM)
    去除呼吸基线漂移(低频)和高频噪声
    """
    if len(data) < fs: return data # 信号太短不处理
    # 填补NaN
    if np.isnan(data).any():
        data = np.nan_to_num(data)
        
    b, a = butter_bandpass(0.7, 3.5, fs, order=2)
    try:
        # 使用 filtfilt 保证零相位偏移
        y = filtfilt(b, a, data)
        return y
    except:
        return data

# ---------------------------------------------------------------
# 2. STM 模型 (Short-Term Memory)
# ---------------------------------------------------------------
class StandaloneSTM:
    def __init__(self, theta_arousal, theta_pool, lambda_a=0.05, fs=30, window_sec=3, step_sec=1):
        self.fs = fs
        self.window_size = int(fs * window_sec)
        self.step_size = int(fs * step_sec)
        self.theta_arousal = theta_arousal  # Stage 1 阈值 (Xi)
        self.theta_pool = theta_pool        # Stage 2 阈值
        self.lambda_a = lambda_a            # 衰减因子

    def compute_windowed_rmssd(self, rppg_signal):
        """计算滑动窗口内的信号波动性 (近似HRV/能量)"""
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
            # RMSSD: Root Mean Square of Successive Differences
            rmssd = np.sqrt(np.mean(diff_signal ** 2))
            rmssd_series.append(rmssd)
        return np.array(rmssd_series)

    def run(self, input_signal, is_already_filtered=False):
        """
        运行模型
        Args:
            input_signal: rPPG信号
            is_already_filtered: 如果外部已经做过带通滤波，设为True以加速
        """
        # 1. 滤波 (如果尚未滤波)
        if is_already_filtered:
            filtered_sig = input_signal
        else:
            filtered_sig = apply_filter(input_signal, self.fs)
        
        # 2. Z-Score 标准化 (基于整段信号)
        if len(filtered_sig) < self.window_size or np.std(filtered_sig) < 1e-6:
            return 0, 0.0 
            
        rppg_norm = (filtered_sig - np.mean(filtered_sig)) / np.std(filtered_sig)

        # 3. 计算特征序列
        rmssd_series = self.compute_windowed_rmssd(rppg_norm)
        if rmssd_series.size == 0:
            return 0, 0.0
        
        # 4. STM 核心逻辑
        rmssd_mean = np.mean(rmssd_series)
        rmssd_res = np.abs(rmssd_series - rmssd_mean)
        
        # Stage 1: 瞬时激活检测
        S_A = (rmssd_res > self.theta_arousal).astype(float)
        
        # Stage 2: 记忆衰减与累积
        I_A = np.zeros_like(S_A)
        time_step_val = self.step_size / self.fs
        decay_factor = np.exp(-self.lambda_a * time_step_val)
        
        if len(I_A) > 0:
            I_A[0] = S_A[0]
            for t in range(1, len(S_A)):
                # Leaky Integrator
                I_A[t] = S_A[t] + I_A[t-1] * decay_factor
            
        avg_intensity = np.mean(I_A) if len(I_A) > 0 else 0
        
        # 最终分类判决
        prediction = 1 if avg_intensity > self.theta_pool else 0
        
        return prediction, avg_intensity

# ---------------------------------------------------------------
# 3. 数据加载与预处理
# ---------------------------------------------------------------
def load_and_preprocess_dataset(pickle_path, sessions_root, gt_threshold):
    print(f"Loading raw data from {pickle_path}...")
    if not os.path.exists(pickle_path):
        print("Error: Pickle file not found.")
        return []

    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
        
    raw_dataset = []
    predictions = data['predictions'] if 'predictions' in data else data

    # 1. 解析原始数据
    print(f"Parsing sessions (Threshold > {gt_threshold})...")
    for sid_key, chunks in tqdm(predictions.items(), desc="Parsing Raw Data"):
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
            label = 1 if aro_val > gt_threshold else 0
            
            raw_dataset.append({
                'sid': sid,
                'signal': full_signal, # 原始信号
                'label': label
            })
        except: pass

    # 2. [优化] 预滤波处理 (Pre-filtering)
    # 这一步将滤波移出循环，极大地加速后续的 Grid Search 和 Study 4
    print(f"Pre-filtering {len(raw_dataset)} sessions to speed up optimization...")
    processed_dataset = []
    for item in tqdm(raw_dataset, desc="Pre-filtering"):
        item['filtered_signal'] = apply_filter(item['signal'], fs=DEFAULT_FS)
        processed_dataset.append(item)
        
    labels = [d['label'] for d in processed_dataset]
    print(f"Dataset Ready: Low(0)={labels.count(0)}, High(1)={labels.count(1)}")
    return processed_dataset

# ---------------------------------------------------------------
# 4. 网格搜索 (使用预滤波数据)
# ---------------------------------------------------------------
def grid_search_optimization(dataset):
    print("\n========================================")
    print("?? Starting Grid Search (Optimized)")
    print("========================================")
    
    # 参数空间
    thetas_arousal = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2] 
    thetas_pool = [0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    lambdas = [0.005, 0.01, 0.03, 0.05, 0.1]
    
    best_score = -1
    best_params = {}
    
    total_combs = len(thetas_arousal) * len(thetas_pool) * len(lambdas)
    print(f"Testing {total_combs} combinations...")

    # 为了进度条显示整洁，这里简略打印
    for lam in lambdas:
        for ta in thetas_arousal:
            for tp in thetas_pool:
                y_true = []
                y_pred = []
                
                stm = StandaloneSTM(theta_arousal=ta, theta_pool=tp, 
                                    lambda_a=lam, window_sec=DEFAULT_WINDOW_SEC)
                
                for item in dataset:
                    # 注意：这里使用 'filtered_signal' 且 is_already_filtered=True
                    pred, _ = stm.run(item['filtered_signal'], is_already_filtered=True)
                    y_true.append(item['label'])
                    y_pred.append(pred)
                
                report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                macro_f1 = report['macro avg']['f1-score']
                recall_high = report['1']['recall'] # 关注 High Arousal 的召回率
                
                # 约束：必须至少召回 5% 的正样本，防止模型全猜 0 获得虚高准确率
                if recall_high > 0.05:
                    if macro_f1 > best_score:
                        best_score = macro_f1
                        best_params = {'lambda': lam, 'ta': ta, 'tp': tp, 'score': macro_f1}
                        # print(f"New Best: {best_params}") # 可选：打印中间结果

    print(f"\n?? Best Parameters Found: {best_params}")
    if not best_params:
        print("Warning: No valid params found. Using safe defaults.")
        return {'ta': 0.05, 'tp': 1.05, 'lambda': 0.05}
    return best_params

# ---------------------------------------------------------------
# 5. Study 4: 阈值敏感性分析 (新增)
# ---------------------------------------------------------------
def run_study4_sensitivity(dataset, best_tp, best_lambda):
    """
    生成 Sensitivity Curve: Accuracy/F1 vs. Stage 1 Threshold (Xi)
    """
    print("\n========================================")
    print("?? Running Study 4: Threshold Sensitivity")
    print("========================================")
    
    # 在 0.01 到 0.30 之间生成 30 个测试点
    thresholds = np.linspace(0.01, 0.30, 30)
    accuracies = []
    f1_scores = []
    
    for th in thresholds:
        y_true = []
        y_pred = []
        
        # 固定 tp 和 lambda，只改变 theta_arousal (th)
        stm = StandaloneSTM(theta_arousal=th, theta_pool=best_tp, 
                            lambda_a=best_lambda, window_sec=DEFAULT_WINDOW_SEC)
        
        for item in dataset:
            pred, _ = stm.run(item['filtered_signal'], is_already_filtered=True)
            y_true.append(item['label'])
            y_pred.append(pred)
            
        accuracies.append(accuracy_score(y_true, y_pred))
        f1_scores.append(f1_score(y_true, y_pred, average='macro', zero_division=0))

    # 绘图
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    plt.plot(thresholds, accuracies, marker='o', markersize=5, linewidth=2, label='Accuracy', color='#2b8cbe')
    plt.plot(thresholds, f1_scores, marker='s', markersize=5, linewidth=2, label='Macro F1', color='#e6550d', linestyle='--')
    
    # 标注峰值
    max_idx = np.argmax(accuracies)
    best_th = thresholds[max_idx]
    best_acc = accuracies[max_idx]
    
    plt.annotate(f'Peak Acc: {best_acc:.3f}\nThreshold: {best_th:.3f}',
                 xy=(best_th, best_acc), xytext=(best_th+0.05, best_acc-0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.axvline(x=best_th, color='gray', linestyle=':', alpha=0.5)
    plt.title(f'Study 4: Sensitivity of Threshold $\\xi$ (Fixed $\\lambda$={best_lambda}, $T_p$={best_tp})')
    plt.xlabel('Stage 1 Threshold $\\xi$ (Signal Volatility)')
    plt.ylabel('Score')
    plt.legend()
    
    plt.savefig(STUDY4_IMG_PATH, dpi=300, bbox_inches='tight')
    print(f"?? Study 4 Plot saved to: {STUDY4_IMG_PATH}")

# ---------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------
def main():
    # 1. 加载数据 (含预滤波优化)
    dataset = load_and_preprocess_dataset(RPPG_PICKLE_PATH, SESSIONS_ROOT, GT_AROUSAL_THRESHOLD)
    if not dataset: return

    # 2. 网格搜索寻找最佳参数
    best_p = grid_search_optimization(dataset)
    
    theta_a = best_p.get('ta', 0.05)
    theta_p = best_p.get('tp', 1.05)
    lambda_val = best_p.get('lambda', 0.05)

    # 3. 使用最佳参数进行最终评估
    print(f"\n========================================")
    print(f"Running Final Evaluation")
    print(f"Params: Ta={theta_a}, Tp={theta_p}, Lambda={lambda_val}")
    print(f"========================================")
    
    stm = StandaloneSTM(theta_arousal=theta_a, theta_pool=theta_p, 
                        lambda_a=lambda_val, window_sec=DEFAULT_WINDOW_SEC)
    
    y_true = []
    y_pred = []
    
    for item in dataset:
        pred, _ = stm.run(item['filtered_signal'], is_already_filtered=True)
        y_true.append(item['label'])
        y_pred.append(pred)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Low(0)', 'High(1)'], zero_division=0))
    
    # 绘制混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Best Model (Ta={theta_a}, Tp={theta_p})')
    plt.savefig(OUTPUT_IMG_PATH, dpi=300)
    print(f"? Confusion Matrix saved to: {OUTPUT_IMG_PATH}")
    
    # 4. 运行 Study 4 (敏感性分析)
    # 使用刚才找到的最佳 Lambda 和 Tp，只分析 Ta 的变化
    run_study4_sensitivity(dataset, best_tp=theta_p, best_lambda=lambda_val)

if __name__ == "__main__":
    main()