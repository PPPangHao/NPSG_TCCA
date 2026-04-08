import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# 修改为你的缓存路径
CACHED_PATH = "/root/autodl-tmp/Code/rPPG-Toolbox/data/MAHNOB-HCI/PreprocessedData/MAHNOB-HCI_SizeW72_SizeH72_ClipLength180_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse"

def check_data():
    # 找到第一个 label 文件
    label_files = glob.glob(os.path.join(CACHED_PATH, "*_label0.npy"))
    if not label_files:
        print("没有找到预处理数据，请先运行预处理！")
        return

    # 随机看 3 个样本
    for i in range(min(3, len(label_files))):
        label_path = label_files[i]
        input_path = label_path.replace("label", "input")
        
        # 加载
        label = np.load(label_path) # Shape: (T,)
        video = np.load(input_path) # Shape: (T, H, W, 3) or (3, T, H, W) depending on format
        
        print(f"正在检查: {os.path.basename(label_path)}")
        print(f"Label Mean: {label.mean():.4f}, Std: {label.std():.4f}")
        
        # 绘图
        plt.figure(figsize=(10, 4))
        
        # 画 Label 波形
        plt.plot(label, label='Ground Truth PPG', color='red')
        plt.title(f"Sample {i}: Label Waveform")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"debug_sample_{i}.png")
        print(f"保存了 debug_sample_{i}.png")
        plt.close()

if __name__ == "__main__":
    check_data()
