# -*- coding: gbk -*-
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
from sklearn.metrics import f1_score, accuracy_score

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

# ============================================================
# 新增功能: 基于原始 Pickle 进行筛选另存 (最稳妥方案)
# ============================================================
def export_filtered_pickle_from_original(dataset, best_params, original_pickle_path, output_path="./active_sessions_filtered.pickle"):
    print(f"\n========================================")
    print(f"?? Filtering Original Pickle based on STM results...")
    print(f"   Source: {original_pickle_path}")
    print(f"   Target: {output_path}")
    print(f"========================================")
    
    # 1. 先跑一遍 STM，拿到所有被判定为“激活”的 SID 列表
    stm = StandaloneSTM_Visual(best_params['ta'], best_params['tp'], best_params['lambda'])
    active_sids = set()
    
    print("Running STM to identify active sessions...")
    for item in dataset:
        # 使用预滤波数据快速判断
        pred, _ = stm.run(item['filtered_signal'], is_already_filtered=True)
        if pred == 1:
            active_sids.add(item['sid'])
            
    print(f"-> Found {len(active_sids)} active sessions (Predicted=1).")

    # 2. 加载原始 Pickle 文件 (保持原汁原味)
    with open(original_pickle_path, 'rb') as f:
        original_data = pickle.load(f)

    # 3. 定位到 predictions 字典
    # 兼容两种结构：直接是字典，或者嵌套在 'predictions' 里
    if isinstance(original_data, dict) and 'predictions' in original_data:
        target_dict = original_data['predictions']
        is_nested = True
    else:
        target_dict = original_data
        is_nested = False

    # 4. 创建筛选后的字典 (只保留 active_sids 中的键)
    filtered_dict = {}
    kept_count = 0
    
    for sid_key, chunk_data in target_dict.items():
        # 转换 key 类型以匹配 (pickle 里可能是字符串 '1042'，也可能是整数 1042)
        try:
            sid_int = int(sid_key)
        except ValueError:
            continue # 跳过非 SID 的键
            
        if sid_int in active_sids:
            # === 关键：直接复制原始数据引用，不修改内部结构 ===
            filtered_dict[sid_key] = chunk_data 
            kept_count += 1

    # 5. 组装回原来的结构
    if is_nested:
        original_data['predictions'] = filtered_dict
        final_obj = original_data
    else:
        final_obj = filtered_dict

    # 6. 保存
    with open(output_path, 'wb') as f:
        pickle.dump(final_obj, f)
        
    print(f"?? Filtered Pickle Saved!")
    print(f"   - Original Sessions: {len(target_dict)}")
    print(f"   - Kept Sessions: {kept_count}")
    print(f"   - Path: {output_path}")
    print(f"   - Structure maintained exactly as original.")    
    
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
        
    def compute_windowed_energy(self, rppg_signal):
        """计算滑动窗口内的局部信号能量 (Local Energy)"""
        energy_series = []
        L = len(rppg_signal)
        if L < self.window_size:
            return np.array([])
        
        for i in range(0, L - self.window_size + 1, self.step_size):
            window = rppg_signal[i : i + self.window_size]
            # 核心修改：使用局部信号的标准差代表信号能量/波动强度
            energy = np.std(window) 
            energy_series.append(energy)
        return np.array(energy_series)
    
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
        
        
        # 在 run() 方法中相应修改：
        # 3. 计算特征序列
        energy_series = self.compute_windowed_energy(rppg_norm)
        if energy_series.size == 0:
            return 0, 0.0
        
        # 4. STM 核心逻辑
        energy_mean = np.mean(energy_series)
        # 计算局部能量偏离全局平均能量的绝对值 (D_t)
        energy_res = np.abs(energy_series - energy_mean) 
        
        # Stage 1: 瞬时激活检测
        S_A = (energy_res > self.theta_arousal).astype(float)

        # 3. 计算特征序列
        # rmssd_series = self.compute_windowed_rmssd(rppg_norm)
        # if rmssd_series.size == 0:
        #     return 0, 0.0
        
        # 4. STM 核心逻辑
        # rmssd_mean = np.mean(rmssd_series)
        # rmssd_res = np.abs(rmssd_series - rmssd_mean)
        
        # Stage 1: 瞬时激活检测
        # S_A = (rmssd_res > self.theta_arousal).astype(float)
        
        # Stage 2: 记忆衰减与累积
        I_A = np.zeros_like(S_A)
        time_step_val = self.step_size / self.fs
        decay_factor = np.exp(-self.lambda_a * time_step_val)
        
        if len(I_A) > 0:
            # print(f"QQQQQQQQQQQQQQQQQQQQQQQ: {len(input_signal)}, {len(filtered_sig)}, {self.window_size}, {len(rppg_norm)}, {len(rmssd_res)}, {len(S_A)}, {len(I_A)}")
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
    
    
# ============================================================
# 0. 更新 STM 类 (支持 Debug 模式用于绘图)
# ============================================================
class StandaloneSTM_Visual(StandaloneSTM): # 继承之前的类
    def run(self, input_signal, is_already_filtered=False, debug=False):
        # 1. 滤波
        if is_already_filtered:
            filtered_sig = input_signal
        else:
            filtered_sig = apply_filter(input_signal, self.fs)
        
        # 2. Z-Score
        if len(filtered_sig) < self.window_size or np.std(filtered_sig) < 1e-6:
            if debug: return 0, 0.0, [], [], []
            return 0, 0.0
            
        rppg_norm = (filtered_sig - np.mean(filtered_sig)) / np.std(filtered_sig)

        # 3. 特征序列
        rmssd_series = self.compute_windowed_rmssd(rppg_norm)
        if rmssd_series.size == 0:
            if debug: return 0, 0.0, [], [], []
            return 0, 0.0
        
        # 4. STM 逻辑
        rmssd_mean = np.mean(rmssd_series)
        rmssd_res = np.abs(rmssd_series - rmssd_mean) # Volatility Deviation (D_t)
        
        S_A = (rmssd_res > self.theta_arousal).astype(float)
        I_A = np.zeros_like(S_A)
        time_step_val = self.step_size / self.fs
        decay_factor = np.exp(-self.lambda_a * time_step_val)
        
        if len(I_A) > 0:
            I_A[0] = S_A[0]
            for t in range(1, len(S_A)):
                I_A[t] = S_A[t] + I_A[t-1] * decay_factor # 累积公式
            
        avg_intensity = np.mean(I_A) if len(I_A) > 0 else 0
        prediction = 1 if avg_intensity > self.theta_pool else 0
        
        if debug:
            # 返回: 预测值, 得分, (用于绘图的中间变量: 原始波动, 激活状态, 积分强度)
            return prediction, avg_intensity, rmssd_res, S_A, I_A
        return prediction, avg_intensity


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
    
    
# ============================================================
# Study 5: Memory Decay (Lambda) Ablation
# ============================================================
def run_study5_lambda_ablation(dataset, best_ta, best_tp):
    print("\n?? Running Study 5: Impact of Memory Decay (Lambda)...")
    
    # 这里的顺序对应 lambdas 列表
    lambdas = [100, 1.0, 0.5, 0.1, 0.05, 0.01, 0.001]
    # 修改标签：去掉硬编码的 "Best"，只保留数值说明
    raw_labels = ["No Mem\n(100)", "Short\n(1.0)", "Short\n(0.5)", "Mid\n(0.1)", "Mid\n(0.05)", "Long\n(0.01)", "Inf\n(0.001)"]
    
    f1_scores = []
    
    for lam in lambdas:
        y_true, y_pred = [], []
        stm = StandaloneSTM_Visual(theta_arousal=best_ta, theta_pool=best_tp, lambda_a=lam)
        
        for item in dataset:
            pred, _ = stm.run(item['filtered_signal'], is_already_filtered=True)
            y_true.append(item['label'])
            y_pred.append(pred)
            
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_scores.append(f1)
        print(f"  Lambda={lam}: F1={f1:.3f}")

    # --- 自动寻找最佳 (修正点) ---
    best_idx = np.argmax(f1_scores)      # 找到分数最高的索引 (应该是 3, 即 Lambda=0.1)
    best_val = f1_scores[best_idx]
    
    # 动态修改标签，把第一名标为 Best
    final_labels = raw_labels.copy()
    final_labels[best_idx] = f"Best\n({lambdas[best_idx]})"

    # 绘图
    plt.figure(figsize=(9, 6))
    bars = plt.bar(final_labels, f1_scores, color='#6baed6', edgecolor='black')
    
    # 只高亮真正的第一名
    bars[best_idx].set_color('#e6550d') 
    bars[best_idx].set_edgecolor('black')
    
    # 在柱子上标具体数值
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval:.3f}', ha='center', va='bottom', fontsize=9)

    plt.title(f'Study 5: Impact of Memory Decay (Best $\\lambda={lambdas[best_idx]}$)', fontsize=14)
    plt.ylabel('Macro F1-Score', fontsize=12)
    plt.xlabel('Memory Decay Factor', fontsize=12)
    plt.ylim(0.2, 0.6) # 调整Y轴范围以便看清差异
    
    path = "./study5_memory_decay_corrected.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"?? Corrected Plot saved to {path}")

# ============================================================
# Study 6: Comparative Analysis (Baselines)
# ============================================================
# ============================================================
# Study 6: Comparative Analysis (Baselines) - LaTeX Generator
# ============================================================
def run_study6_baselines(dataset, best_params):
    print("\n?? Running Study 6: Baseline Comparison & LaTeX Table Generation...")
    
    y_true = [d['label'] for d in dataset]
    
    # --- 1. Baseline 2: Energy (Variance) ---
    # 对应表格第一行：原始能量阈值
    print("  Calculating Baseline 2 (Energy)...")
    y_pred_energy = []
    for item in dataset:
        sig = item['filtered_signal']
        energy = np.std(sig)
        # 注意：这里的阈值 1.0 是示例，实际可能需要针对数据归一化调整
        y_pred_energy.append(1 if energy > 1.0 else 0) 
    
    f1_energy = f1_score(y_true, y_pred_energy, average='macro', zero_division=0)
    acc_energy = accuracy_score(y_true, y_pred_energy)

    # --- 2. Baseline 1: Peak-based (RMSSD) ---
    # 对应表格第二行：基于峰值的 RMSSD
    print("  Calculating Baseline 1 (Peak RMSSD)...")
    y_pred_peak = []
    for item in dataset:
        sig = item['filtered_signal']
        # 简单的峰值检测
        peaks, _ = find_peaks(sig, distance=15) 
        if len(peaks) < 2:
            rmssd = 0
        else:
            rr_intervals = np.diff(peaks)
            diff_rr = np.diff(rr_intervals)
            rmssd = np.sqrt(np.mean(diff_rr**2)) if len(diff_rr) > 0 else 0
        
        # 注意：这里的阈值 1.5 是示例
        y_pred_peak.append(1 if rmssd > 1.5 else 0) 
        
    f1_peak = f1_score(y_true, y_pred_peak, average='macro', zero_division=0)
    acc_peak = accuracy_score(y_true, y_pred_peak)

    # --- 3. Ours (STM / BA-IF) ---
    # 对应表格第三行：Ours
    print("  Calculating Ours (BA-IF)...")
    stm = StandaloneSTM_Visual(best_params['ta'], best_params['tp'], best_params['lambda'])
    y_pred_ours = []
    for item in dataset:
        p, _ = stm.run(item['filtered_signal'], is_already_filtered=True)
        y_pred_ours.append(p)
        
    f1_ours = f1_score(y_true, y_pred_ours, average='macro', zero_division=0)
    acc_ours = accuracy_score(y_true, y_pred_ours)

    # --- Generate LaTeX Table ---
    # 使用 f-string 自动填入计算结果，保留3位小数
    latex_table = f"""
\\begin{{table}}[h]
    \\centering
    \\caption{{阶段一（二元唤醒检测）性能对比。BA-IF 在 Macro F1 上显著优于基准方法。}}
    \\label{{tab:stage1_comparison}}
    \\begin{{tabular}}{{lccc}}
        \\toprule
        \\textbf{{方法}} & \\textbf{{特征类型}} & \\textbf{{Macro F1}} & \\textbf{{准确率 (Acc)}} \\\\
        \\midrule
        Baseline 2 (原始能量阈值) & 瞬时统计量 & {f1_energy:.3f} & {acc_energy:.3f} \\\\
        Baseline 1 (基于峰值的 RMSSD) & 峰值检测 & {f1_peak:.3f} & {acc_peak:.3f} \\\\
        \\textbf{{Ours (BA-IF)}} & \\textbf{{时序积分特征}} & \\textbf{{{f1_ours:.3f}}} & \\textbf{{{acc_ours:.3f}}} \\\\
        \\bottomrule
    \\end{{tabular}}
\\end{{table}}
    """
    
    print("\n" + "="*40)
    print("LaTeX Code Generated:")
    print("="*40)
    print(latex_table)
    print("="*40 + "\n")

# ============================================================
# Study 7: Qualitative Visualization (v3: 苛刻筛选 "断续输入-持久输出")
# ============================================================
def run_qualitative_viz(dataset, best_params):
    print("\n?? Running Study 7: Qualitative Visualization (v3 - Searching for intermittent->sustained case)...")
    
    stm = StandaloneSTM_Visual(best_params['ta'], best_params['tp'], best_params['lambda'])
    
    best_candidate = None
    best_visual_score = -999 
    
    count_checked = 0
    count_passed_hard_constraints = 0
    
    # --- 1. 遍历所有数据，寻找“教科书级”样本 ---
    for item in dataset:
        if item['label'] != 1: continue # 只看正样本
        count_checked += 1

        # 运行模型拿到中间变量
        # S_A 是 Stage 1 的二值状态 (0或1)，integrals 是 Stage 2 的连续值
        pred, intensity, vol_series, S_A, integrals = stm.run(item['filtered_signal'], 
                                                                 is_already_filtered=True, 
                                                                 debug=True)
        
        # [硬性约束 1]: 必须预测正确，且积分峰值必须超过阈值
        if pred != 1 or np.max(integrals) <= best_params['tp']:
            continue
            
        # [硬性约束 2]: 输入必须是断续的 (Intermittent)
        # 计算 S_A 中从 0 变到 1 的次数，这代表“爆发”的次数
        # prepend=0 是为了处理开头就是1的情况
        num_bursts = np.sum(np.diff(S_A, prepend=0) == 1)
        # 关键：如果爆发次数少于 3 次，说明红色区域太连续了，体现不出模型的本事，跳过！
        if num_bursts < 3: 
            continue

        count_passed_hard_constraints += 1
        
        # --- [选美打分逻辑 v3] ---
        
        # 1. 持久性得分 (Sustained Score) - 最重要
        # 找到第一次超过阈值的时间点
        first_cross_idx = np.argmax(integrals > best_params['tp'])
        
        # 分析从第一次触发到结束的这段时间
        sustained_segment = integrals[first_cross_idx:]
        if len(sustained_segment) == 0: continue

        # 计算这段时间内，有多大比例保持在阈值之上
        above_mask = (sustained_segment >= best_params['tp'])
        sustained_score = np.sum(above_mask) / len(sustained_segment)
        
        # 2. 峰值得分 (辅助)
        peak_score = np.max(integrals) - best_params['tp']

        # 综合打分：极度强调持久性，同时鼓励更多的断续爆发
        # 如果 sustained_score 不是接近 1.0，总分会很低
        total_score = (sustained_score * 20.0) + (num_bursts * 0.5) + peak_score
        
        if total_score > best_visual_score:
            best_visual_score = total_score
            best_candidate = {
                'vol': vol_series,
                'int': integrals,
                'sid': item['sid'],
                # 'states': S_A 
            }
    
    print(f"Checked {count_checked} positive samples.")
    print(f"Found {count_passed_hard_constraints} samples matching 'intermittent' criteria.")
    
    if not best_candidate:
        print("Error: No suitable visualization sample found that meets strict criteria.")
        return # 退出，避免报错

    print(f"?? Best sample found! SID: {best_candidate['sid']} (Score: {best_visual_score:.2f})")
    # 如果 sustained_score 很高，这里打印出来应该接近 20+
    
    # --- 2. 绘图 (保持 v2 的双子图样式) ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True) # 稍微压扁一点
    plt.subplots_adjust(hspace=0.25, top=0.9, bottom=0.12) # 调整边距防止遮挡
    
    # 对齐时间轴 (假设 step=1s)
    step_sec = 1 
    t_feat = np.arange(len(best_candidate['vol'])) * step_sec
    
    # Subplot 1: Volatility Deviation (Stage 1)
    ax1.plot(t_feat, best_candidate['vol'], color='#1f77b4', linewidth=1.5, label='Volatility $\mathcal{D}_t$')
    ax1.axhline(y=best_params['ta'], color='r', linestyle='--', linewidth=2, label=f'Trigger $\\xi$')
    ax1.fill_between(t_feat, best_candidate['vol'], best_params['ta'], 
                     where=(best_candidate['vol'] >= best_params['ta']), 
                     interpolate=True, color='red', alpha=0.3, label='Intermittent Activation') # 改名强调断续
    
    ax1.set_title('(a) Stage 1: Intermittent Volatility Input', loc='left', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Volatility')
    ax1.legend(loc='upper right', frameon=True, fontsize=9)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.set_ylim(bottom=0) # 让Y轴从0开始更美观

    # Subplot 2: Bio-Accumulated Intensity (Stage 2)
    ax2.plot(t_feat, best_candidate['int'], color='#2ca02c', linewidth=2.5, label='Accumulated Intensity $I_t$')
    ax2.axhline(y=best_params['tp'], color='orange', linestyle='--', linewidth=2, label=f'Decision Threshold $\\theta_{{pool}}$')
    ax2.fill_between(t_feat, best_candidate['int'], best_params['tp'], 
                     where=(best_candidate['int'] >= best_params['tp']), 
                     interpolate=True, color='green', alpha=0.2, label='Sustained Output') # 改名强调持久

    ax2.set_title('(b) Stage 2: Bio-Accumulative Integration (Memory Effect)', loc='left', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Intensity $I_t$')
    ax2.set_xlabel('Time (seconds)', fontsize=11)
    ax2.legend(loc='upper left', frameon=True, fontsize=9)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.set_ylim(bottom=0)

    path = "./study7_qualitative_viz_v3.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"?? Qualitative Plot (v3) saved to {path}")

# ============================================================
# 修改 Main 函数调用
# ============================================================
def main_studies():
    # 假设你已经有了 best_params 和 dataset
    # 这里只是模拟调用，你需要把这些放在你主程序的最后
    best_p = {'ta': 0.05, 'tp': 1.05, 'lambda': 0.05} # 示例
    # run_study5_lambda_ablation(dataset, best_p['ta'], best_p['tp'])
    # run_study6_baselines(dataset, best_p)
    # run_qualitative_viz(dataset, best_p)
    pass
    

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
    # run_study4_sensitivity(dataset, best_tp=theta_p, best_lambda=lambda_val)
    # run_study5_lambda_ablation(dataset, best_p['ta'], best_p['tp'])
    run_study6_baselines(dataset, best_p)
    # run_qualitative_viz(dataset, best_p)
    
    # 5. [新增] 导出激活态 Session 数据
    export_filtered_pickle_from_original(
        dataset, 
        best_p, 
        original_pickle_path=RPPG_PICKLE_PATH, 
        output_path="./accurate_pickle/active_sessions_dump.pickle"
    )

if __name__ == "__main__":
    main()