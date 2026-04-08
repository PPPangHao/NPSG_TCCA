# validate_data.py
import numpy as np
import os
import glob

def validate_preprocessed_data(cached_path):
    """验证预处理数据的完整性"""
    input_files = glob.glob(os.path.join(cached_path, "*_input*.npy"))
    label_files = glob.glob(os.path.join(cached_path, "*_label*.npy"))
    
    print(f"Found {len(input_files)} input files and {len(label_files)} label files")
    
    issues = []
    
    for i, input_file in enumerate(input_files):
        try:
            # 检查对应的标签文件是否存在
            label_file = input_file.replace("_input", "_label")
            if not os.path.exists(label_file):
                issues.append(f"Missing label file for {input_file}")
                continue
            
            # 加载并检查数据形状
            data = np.load(input_file)
            label = np.load(label_file)
            
            # 检查数据形状
            if data.ndim != 4:
                issues.append(f"Input {input_file} has wrong dimensions: {data.shape}")
            
            if label.ndim != 1:
                issues.append(f"Label {label_file} has wrong dimensions: {label.shape}")
            
            # 检查数据范围
            if np.isnan(data).any() or np.isinf(data).any():
                issues.append(f"Input {input_file} contains NaN or Inf values")
            
            if np.isnan(label).any() or np.isinf(label).any():
                issues.append(f"Label {label_file} contains NaN or Inf values")
                
        except Exception as e:
            issues.append(f"Error loading {input_file}: {e}")
    
    # 输出结果
    if issues:
        print("Found issues:")
        for issue in issues[:10]:  # 只显示前10个问题
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more issues")
    else:
        print("All data files are valid!")
    
    return len(issues) == 0

if __name__ == "__main__":
    cached_path = "/root/autodl-tmp/Code/rPPG-Toolbox/data/MAHNOB-HCI/PreprocessedData/MAHNOB-HCI_SizeW72_SizeH72_ClipLength180_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse" # 更新为你的路径
    is_valid = validate_preprocessed_data(cached_path)
    
    if not is_valid:
        print("\nData validation failed. Consider re-running preprocessing.")
        print("Set DO_PREPROCESS=True in your config file and run training again.")
