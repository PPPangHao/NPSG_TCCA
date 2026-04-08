import numpy as np
import matplotlib.pyplot as plt
import glob

# 指向你的缓存路径
cached_path = "/root/autodl-tmp/Code/rPPG-Toolbox/data/MAHNOB-HCI/PreprocessedData/MAHNOB-HCI_SizeW72_SizeH72_ClipLength180_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse"

# 随便找一个 label 文件
label_files = glob.glob(cached_path + "/*_label7.npy")

if len(label_files) > 0:
    # 读取第一个文件
    file_path = label_files[0]
    label_data = np.load(file_path)
    
    print(f"Checking file: {file_path}")
    print(f"Data shape: {label_data.shape}")
    print(f"Mean: {label_data.mean():.4f}, Std: {label_data.std():.4f}")
    
    # 画图
    plt.figure(figsize=(10, 4))
    plt.plot(label_data)
    plt.title("Final Preprocessed Label (from .npy)")
    plt.grid(True)
    plt.savefig("check_final_npy.png")
    print("Saved check_final_npy.png")
else:
    print("No .npy files found. Did you run preprocess?")
