import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt

# ================= 配置区域 =================
# 指向你的缓存文件夹
CACHED_PATH = "/root/autodl-tmp/Code/rPPG-Toolbox/data/MAHNOB-HCI/PreprocessedData/MAHNOB-HCI_SizeW72_SizeH72_ClipLength180_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse"
# 保存调试图片的文件夹
OUTPUT_DIR = "./debug_output"
# 想要检查的文件数量
NUM_SAMPLES = 5
# ===========================================

def normalize_to_uint8(img_data):
    """
    将标准化(Standardized)或归一化的浮点数据转换为 0-255 的图像以便查看
    """
    # Min-Max 归一化到 0-1
    img_min = img_data.min()
    img_max = img_data.max()
    if img_max - img_min == 0:
        return np.zeros_like(img_data, dtype=np.uint8)
    
    img_norm = (img_data - img_min) / (img_max - img_min)
    # 转换到 0-255
    img_uint8 = (img_norm * 255).astype(np.uint8)
    return img_uint8

def main():
    if not os.path.exists(CACHED_PATH):
        print(f"错误: 找不到缓存路径 {CACHED_PATH}")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"创建输出目录: {OUTPUT_DIR}")

    # 寻找所有的 input npy 文件
    input_files = glob.glob(os.path.join(CACHED_PATH, "*_input*.npy"))
    if not input_files:
        print("错误: 目录下没有找到 *_input*.npy 文件，请先运行预处理！")
        return

    print(f"找到 {len(input_files)} 个数据文件，正在采样前 {NUM_SAMPLES} 个...")

    # 随机打乱或者取前几个
    # np.random.shuffle(input_files) 
    
    for i in range(min(NUM_SAMPLES, len(input_files))):
        fpath = input_files[i]
        fname = os.path.basename(fpath)
        
        try:
            # 加载数据
            # Shape通常是 (T, H, W, C) 或者 (T, C, H, W)，取决于你的BaseLoader怎么存的
            # 根据你之前的代码 BaseLoader.preprocess -> concatenate -> chunk -> save
            # 存下来的应该是 (T, H, W, Channels)
            data = np.load(fpath)
            
            print(f"\n[{i+1}/{NUM_SAMPLES}] 处理文件: {fname}")
            print(f"  数据形状: {data.shape}")
            print(f"  数据类型: {data.dtype}")
            print(f"  数据范围: min={data.min():.3f}, max={data.max():.3f}")

            # 假设数据是 (T, H, W, C)
            # T: 时间长度 (Chunk Length)
            # H, W: 72, 72
            # C: 通道数 (DiffNormalized=3 + Standardized=3 = 6)
            
            if data.ndim == 4:
                T, H, W, C = data.shape
                mid_frame_idx = T // 2
                
                # 提取中间一帧
                frame_data = data[mid_frame_idx] # (H, W, C)
                
                # 分离通道
                # 你的配置是 ["DiffNormalized", "Standardized"]
                # 通道 0-2: DiffNormalized (帧差)
                # 通道 3-5: Standardized (原始人脸归一化)
                
                if C >= 6:
                    # 提取 Standardized 部分 (比较像人脸)
                    face_std = frame_data[:, :, 3:6]
                    # 提取 DiffNormalized 部分 (噪点)
                    face_diff = frame_data[:, :, 0:3]
                    
                    # 还原为可视化图像
                    img_face = normalize_to_uint8(face_std)
                    img_diff = normalize_to_uint8(face_diff)
                    
                    # 因为 rPPG-Toolbox 通常用 RGB 读取，OpenCV 保存需要 BGR
                    img_face = cv2.cvtColor(img_face, cv2.COLOR_RGB2BGR)
                    img_diff = cv2.cvtColor(img_diff, cv2.COLOR_RGB2BGR)
                    
                    # 保存图片
                    cv2.imwrite(os.path.join(OUTPUT_DIR, f"sample_{i}_face.png"), img_face)
                    cv2.imwrite(os.path.join(OUTPUT_DIR, f"sample_{i}_diff.png"), img_diff)
                    
                    print(f"  -> 已保存人脸样本: sample_{i}_face.png (检查裁剪范围)")
                    print(f"  -> 已保存帧差样本: sample_{i}_diff.png")
                else:
                    print("  警告: 通道数少于6，可能只包含 Raw 或一种数据类型，尝试直接保存前3通道")
                    img_raw = normalize_to_uint8(frame_data[:, :, :3])
                    img_raw = cv2.cvtColor(img_raw, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(OUTPUT_DIR, f"sample_{i}_raw.png"), img_raw)
            else:
                print(f"  跳过: 数据维度 {data.ndim} 不是预期的 4 (T,H,W,C)")

        except Exception as e:
            print(f"  读取失败: {e}")

    print(f"\n完成！请去 {OUTPUT_DIR} 查看图片。")

if __name__ == "__main__":
    main()
