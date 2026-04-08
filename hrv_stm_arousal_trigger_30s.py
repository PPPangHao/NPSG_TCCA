import pickle
import numpy as np
import torch
import xml.etree.ElementTree as ET
import os
from scipy.signal import find_peaks

# --- 1. Arousal-STM 模块的超参数 ---
# 注意：这些参数需要通过数据驱动的方法进行优化 (如论文中提到的)
THETA_AROUSAL = 0.05       # θ_arousal: HRV_res 的变化阈值，用于产生方波
THETA_POOL_A = 1.0         # θ_pool_A: 状态累积值 I_A(t) 的触发阈值
LAMBDA_A = 0.05            # λ_A: 指数衰减率 (值越大，衰减越快)

# --- 2. 改进的 HRVEmotionProcessor 类 ---

class HRVEmotionProcessor:
    def __init__(self, fs=30, window_sec=6, step_sec=1):
        self.fs = fs                                # rPPG 采样频率
        self.window_size = int(fs * window_sec)     # 滑动窗口大小 (例如 180 帧)
        self.step_size = int(fs * step_sec)         # 滑动步长 (例如 30 帧)

    # 保持 load_pickle, concatenate_rppg_signal, parse_arousal_from_xml

    def concatenate_rppg_signal(self, pred_dict):
        """将多个 rPPG 信号片段拼接成一个长信号"""
        pred_signal = np.concatenate([
            v.numpy().flatten() if isinstance(v, torch.Tensor) else np.array(v).flatten()
            for k, v in sorted(pred_dict.items())
        ])
        return pred_signal
        
    def load_pickle(self, file_path):
        """加载pickle文件"""
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data

    def compute_windowed_rmssd(self, rppg_signal):
        """
        计算 rPPG 信号的滑动窗口 RMSSD，返回时序 RMSSD 信号
        返回: (rmssd_series, timestamps)
        """
        rmssd_series = []
        timestamps = []
        L = len(rppg_signal)
        
        # 确保窗口大小足够
        if L < self.window_size:
            # 信号太短，返回空
            return np.array([]), np.array([]) 

        # 滑动窗口计算
        for i in range(0, L - self.window_size + 1, self.step_size):
            window = rppg_signal[i : i + self.window_size]
            # 计算连续采样点之间的差值
            diff_signal = np.diff(window)
            # 计算RMSSD
            rmssd = np.sqrt(np.mean(diff_signal ** 2))
            rmssd_series.append(rmssd)
            timestamps.append(i / self.fs) # 记录窗口的起始时间

        return np.array(rmssd_series), np.array(timestamps)

    def extract_arousal_truth(self, xml_file_path):
        """
        从 MAHNOB-HCI 的 session.xml 文件中提取 Arousal 真值 (feltArsl 属性)
        """
        try:
            tree = ET.parse(xml_file_path)
            root = tree.getroot()
            
            # 直接从根标签的属性中查找 'feltArsl'
            arousal_score_str = root.get('feltArsl')
            
            if arousal_score_str:
                # 将属性值转换为浮点数
                arousal_score = float(arousal_score_str)
                return arousal_score
            else:
                # 属性不存在
                return None
        except Exception as e:
            # 捕获文件解析错误
            print(f"Error parsing XML file {xml_file_path}: {e}")
            return None

    def arousal_soft_triggering_mechanism(self, rmssd_series):
        """
        实现 Arousal 柔性触发机制 (Arousal-STM)
        基于 RMSSD 信号的时序变化。

        返回: I_A(t) 时序累积值 和 S_A(t) 时序方波信号
        """
        if rmssd_series.size == 0:
            return np.array([]), np.array([])
        
        # 1. RMSSD 变化率 (简化版 H_res(t))
        # 我们使用相邻窗口 RMSSD 差值的绝对值来代表变化率
        # 为了稳定，可以先对 rmssd_series 进行平滑
        
        # 使用 RMSSD 信号的均值作为参考，计算残差
        rmssd_mean = np.mean(rmssd_series)
        rmssd_res = np.abs(rmssd_series - rmssd_mean)

        # 2. 方波提取器 S_A(t)
        # 当 RMSSD 变化残差超过阈值 θ_arousal 时，激活
        S_A = (rmssd_res > THETA_AROUSAL).astype(float)

        # 3. 状态累积 I_A(t) (时间衰减函数)
        I_A = np.zeros_like(S_A)
        time_step = self.step_size / self.fs # I_A 的时间步长 (等于滑动窗口的步长)
        
        # 使用离散形式实现时间衰减累积 (近似于公式 5)
        # I(t) = S(t) + I(t-dt) * e^(-lambda * dt)
        
        I_A[0] = S_A[0]
        
        # e^(-lambda * dt) 是前一个状态的衰减系数
        decay_factor = np.exp(-LAMBDA_A * time_step) 

        for t in range(1, len(S_A)):
            # I_A(t) = S_A(t) + I_A(t-1) * 衰减系数
            I_A[t] = S_A[t] + I_A[t-1] * decay_factor
            
        return I_A, S_A

    def generate_stm_features(self, rppg_signal):
        """
        从 rPPG 信号中提取 STM 特征 F
        """
        # 1. 提取时序 RMSSD 信号
        rmssd_series, _ = self.compute_windowed_rmssd(rppg_signal)
        
        if rmssd_series.size == 0:
            return None

        # 2. 运行 Arousal-STM 机制
        I_A, S_A = self.arousal_soft_triggering_mechanism(rmssd_series)

        # 3. 触发判别器 (可选的二值触发信号 T_A)
        # T_A = (I_A > THETA_POOL_A).astype(float)
        
        # 4. 组成 STM 输出 F (对应公式 6 的思想)
        # 这里 F 包含 RMSSD 时序信号本身 (x_acc/x_HR 对应部分)
        # 和 Arousal 强度累积值 I_A (S 对应部分，嵌入了事件信息)
        
        # 将 rmssd_series 和 I_A(t) 堆叠成特征矩阵
        # (N_windows, 2) 维度的特征矩阵
        F = np.stack([rmssd_series, I_A], axis=1)

        # 5. 最终的 STM 输出 F_tensor
        return torch.tensor(F, dtype=torch.float32)

# --- 3. 示例使用和流程修改 ---

if __name__ == "__main__":
    
    # 假设 MAHNOB-HCI 数据的路径结构如下 (需要您自行修改)
    DATA_ROOT = '/dataset/MAHNOB-HCI/' 
    PICKLE_FILE = './accurate_pickle/accurate.pickle'  

    processor = HRVEmotionProcessor(fs=30, window_sec=6, step_sec=1)
    data = processor.load_pickle(PICKLE_FILE)

    # 仅处理一个 Subject 作为示例
    subject_id = list(data['predictions'].keys())[0] 
    
    print(f"--- Processing Subject: {subject_id} ---")
    
    # 1. 拼接 rPPG 信号
    rppg_signal = processor.concatenate_rppg_signal(data['predictions'][subject_id])

    # 2. 提取 STM 特征
    stm_features = processor.generate_stm_features(rppg_signal)

    if stm_features is not None:
        print(f"RMSSD/I_A STM features shape: {stm_features.shape}")
        
        # 3. 获取 Arousal 真值 (假设 session.xml 路径已知)
        # 路径示例: {DATA_ROOT}/Session{subject_id}/session.xml
        xml_path = os.path.join(DATA_ROOT, f"Sessions", f"{subject_id}", "session.xml")
        arousal_truth = processor.extract_arousal_truth(xml_path)
        
        print(f"Extracted Discrete Arousal Truth: {arousal_truth}")
        
        # 4. 后续步骤：将 stm_features (时序特征) 和 arousal_truth (标签) 
        # 送入您的深度学习网络进行 Arousal 分类训练
        # 注意: 这里的 stm_features 是一个时间序列，需要您的分类网络能够处理序列数据 (如 Transformer, RNN 或 1D-CNN)

    print("STM Feature Generation Complete.")