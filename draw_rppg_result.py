import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import glob
import re

# ================= 配置路径 =================
# 1. 模型预测结果文件 (.pickle)
PICKLE_PATH = "./data/MAHNOB-HCI/predict/MAHNOB_HCI_tscan_Epoch1_MAHNOB-HCI_outputs.pickle"

# 2. 预处理缓存文件夹 (直接读取这里的 GT)
CACHED_PATH = "/root/autodl-tmp/Code/rPPG-Toolbox/data/MAHNOB-HCI/PreprocessedData/MAHNOB-HCI_SizeW72_SizeH72_ClipLength180_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse"

# 3. 图片保存位置
SAVE_DIR = "./rppg_result_cache_gt"
# ===========================================

def natural_sort_key(s):
    """
    自然排序键生成器。
    让 'label2.npy' 排在 'label10.npy' 前面，而不是后面。
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"创建输出目录: {SAVE_DIR}")

    # 加载预测结果
    print(f"正在加载预测文件: {PICKLE_PATH}")
    with open(PICKLE_PATH, "rb") as f:
        data = pickle.load(f)

    predictions = data['predictions']
    fs = data.get('fs', 61) # 默认 61，如果 pickle 里没有

    print(f"共发现 {len(predictions)} 个受试者/Session。")

    for subject_name in predictions.keys():
        print(f"\nProcessing Subject: {subject_name}")

        # --------------------- 1. Load Pred (从 Pickle) ---------------------
        pred_dict = predictions[subject_name]
        # pred_dict 的 key 通常是 chunk index (0, 1, 2...)
        pred_signal = np.concatenate([
            v.numpy().flatten() if isinstance(v, torch.Tensor) else np.array(v).flatten()
            for k, v in sorted(pred_dict.items())
        ])

        # --------------------- 2. Load GT (从 Cache 文件夹) ---------------------
        # 构造搜索模式，例如: "session_10_label*.npy"
        # 注意：这里假设 pickle 里的 subject_name (如 'session_1') 和文件名开头是一致的
        search_pattern = os.path.join(CACHED_PATH, f"{subject_name}_label*.npy")
        gt_files = glob.glob(search_pattern)

        if not gt_files:
            print(f"  [警告] 在缓存路径中找不到 {subject_name} 的标签文件！跳过。")
            continue

        # [关键] 对文件名进行排序，确保时间顺序正确
        # 如果直接 sort，label10 会排在 label2 前面，导致波形错乱
        gt_files.sort(key=natural_sort_key)

        gt_chunks = []
        for gf in gt_files:
            # 加载 .npy (Shape: [Chunk_Length])
            chunk_data = np.load(gf)
            gt_chunks.append(chunk_data.flatten())
        
        # 拼接所有 chunk
        gt_from_cache = np.concatenate(gt_chunks)

        # --------------------- 3. 对齐长度 ---------------------
        # 理论上长度应该完全一样，但为了防止少许误差报错，取最小值
        min_len = min(len(pred_signal), len(gt_from_cache))
        if len(pred_signal) != len(gt_from_cache):
            print(f"  [提示] 长度不一致: Pred={len(pred_signal)}, GT_Cache={len(gt_from_cache)}. 截断至 {min_len}")
        
        pred_signal = pred_signal[:min_len]
        gt_from_cache = gt_from_cache[:min_len]

        # --------------------- 4. 可视化归一化 (Z-score) ---------------------
        # 这一步是为了让两条线在同一尺度下对比
        # 缓存里的数据可能已经是 Z-score 的了，但再做一次能保证均值为0，方差为1，方便画图
        gt_norm = (gt_from_cache - gt_from_cache.mean()) / (gt_from_cache.std() + 1e-7)
        # 也可以对预测值做同样的归一化以便观察波形趋势（因为幅值对于rPPG往往不准确）
        pred_norm = (pred_signal - pred_signal.mean()) / (pred_signal.std() + 1e-7)

        # --------------------- 5. 打印统计信息 ---------------------
        print(f"  GT (Cache) raw stats: min={gt_from_cache.min():.4f}, max={gt_from_cache.max():.4f}, mean={gt_from_cache.mean():.4f}")

        # --------------------- 6. 绘图 ---------------------
        time_axis = np.arange(min_len) / fs

        plt.figure(figsize=(12, 4))
        
        # 绘制 GT (蓝色)
        plt.plot(time_axis, gt_norm, label='GT from Cache (Normalized)', color='blue', linewidth=1.5)
        
        # 绘制 Pred (红色) - 这里我也对 Pred 做了归一化，方便看相位对齐情况
        # 如果你想看原始预测幅度，改回 pred_signal 即可
        plt.plot(time_axis, pred_norm, label='Prediction (Normalized)', color='red', alpha=0.7, linewidth=1)

        plt.title(f"Comparison: Pred vs Cache GT - {subject_name}")
        plt.xlabel("Time (s)")
        plt.ylabel("Normalized Amplitude (Z-score)")
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(SAVE_DIR, f"compare_{subject_name}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"  -> 已保存: {save_path}")

    print("\n所有绘图完成。")

if __name__ == "__main__":
    main()