import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy import signal

# ================= 配置 =================
# 你的预处理数据路径
CACHED_PATH = "/root/autodl-tmp/Code/rPPG-Toolbox/data/MAHNOB-HCI/PreprocessedData/MAHNOB-HCI_SizeW72_SizeH72_ClipLength180_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse"
# =======================================

def normalize(arr):
    return (arr - np.mean(arr)) / (np.std(arr) + 1e-7)

def check_correlation():
    # 找到所有 input 文件
    input_files = sorted(glob.glob(os.path.join(CACHED_PATH, "*_input*.npy")))
    
    if not input_files:
        print("Error: No input files found!")
        return

    print(f"Checking {len(input_files)} clips...")
    
    corrs = []
    
    # 随机抽查 5 个文件，或者检查前 10 个
    for i in range(min(10, len(input_files))):
        input_path = input_files[i]
        label_path = input_path.replace("input", "label")
        
        if not os.path.exists(label_path):
            continue
            
        # 1. 加载数据
        # Input Shape 通常是: (Frames, Height, Width, Channels) 或 (F, C, H, W)
        # Config: ['DiffNormalized','Standardized'] -> 6 Channels
        # Channel 0-2: DiffNormalized (Motion)
        # Channel 3-5: Standardized (Appearance/RGB) -> 我们主要看这个
        input_data = np.load(input_path)
        label_data = np.load(label_path)
        
        # 检查 Shape 并调整为 (F, H, W, C)
        if input_data.shape[1] == 6: # (F, 6, H, W)
            input_data = np.transpose(input_data, (0, 2, 3, 1))
        
        # 2. 提取绿色通道 (Standardized 分支)
        # Standardized 是后 3 个通道。通常顺序是 RGB 或 BGR。
        # 无论 RGB 还是 BGR，绿色通常都在中间 (Index 1)。
        # 这里 Input 的最后 3 个通道是 Raw/Standardized。
        # 索引：3=R/B, 4=G, 5=B/R
        green_frames = input_data[:, :, :, 4] # 取第 5 个通道作为绿色通道
        
        # 计算每帧的均值 -> 得到 raw rPPG 信号
        raw_green_signal = np.mean(green_frames, axis=(1, 2))
        
        # 3. 简单的带通滤波 (0.75-2.5 Hz) 以便与 Label 比较
        # 因为 Input 是 Standardized 的，包含很多基线漂移，必须滤一下才能和 DiffNormalized 的 Label 比
        fs = 61.0
        b, a = signal.butter(2, [0.75 / (fs/2), 2.5 / (fs/2)], btype='bandpass')
        try:
            raw_green_signal = signal.filtfilt(b, a, raw_green_signal)
        except:
            pass # 可能是数据太短
            
        # 4. 对 Label 积分 (如果 Label 是 DiffNormalized)
        # 如果 Label 是 DiffNormalized (差分过的)，它对应的是变化率。
        # 而 Raw Green 是原始波形。
        # 为了比较，我们可以把 Raw Green 差分一下，或者把 Label 积分一下。
        # 这里我们把 Raw Green 差分一下，让它变成 DiffNormalized 形式
        raw_green_diff = np.diff(raw_green_signal)
        raw_green_diff = np.append(raw_green_diff, 0) # 补齐长度
        
        # 5. 计算 Pearson
        # 截取中间一段以避免边界效应
        L = len(label_data)
        if L > 60:
            crop = slice(30, -30)
        else:
            crop = slice(0, L)
            
        p_val = np.corrcoef(raw_green_diff[crop], label_data[crop])[0, 1]
        corrs.append(p_val)
        
        print(f"File: {os.path.basename(input_path)}")
        print(f"  Shape: {input_data.shape}")
        print(f"  Raw Green vs Label Correlation: {p_val:.4f}")
        
        # 画图保存 (只画第一张)
        if i == 0:
            plt.figure(figsize=(12, 6))
            plt.subplot(2,1,1)
            plt.plot(normalize(raw_green_diff), label='Raw Green (Diff)', alpha=0.7)
            plt.plot(normalize(label_data), label='GT Label', alpha=0.7)
            plt.title(f"Input vs Label (Corr: {p_val:.4f})")
            plt.legend()
            
            plt.subplot(2,1,2)
            plt.imshow(input_data[0, :, :, 3:6].astype(np.float32) - np.min(input_data[0, :, :, 3:6]))
            plt.title("Face Check (Standardized Channel)")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig("check_input_quality.png")
            print("Saved check_input_quality.png")

    print("-" * 20)
    print(f"Average Correlation: {np.mean(corrs):.4f}")
    if np.mean(corrs) < 0.1:
        print("FAIL: Data is likely NOT aligned or Video is garbage.")
    else:
        print("PASS: Data quality is good. Issue is in Model/Training.")

if __name__ == "__main__":
    check_correlation()
