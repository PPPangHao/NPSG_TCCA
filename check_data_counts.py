import os
import glob
import pandas as pd
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 原始数据集根目录 (包含 Sessions 文件夹)
RAW_DATA_ROOT = "/root/autodl-tmp/Code/rPPG-Toolbox/data/MAHNOB-HCI" 
# 或者如果是 Mini 数据集: "/dataset/Mini_MAHNOB-HCI"

# 2. 预处理缓存目录 (包含 .npy 文件)
CACHED_PATH = "/root/autodl-tmp/Code/rPPG-Toolbox/data/MAHNOB-HCI/PreprocessedData/MAHNOB-HCI_SizeW72_SizeH72_ClipLength180_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse"
# ===========================================

def scan_and_count():
    # 1. 获取所有 Session ID
    sessions_dir = os.path.join(RAW_DATA_ROOT, "Sessions")
    if not os.path.exists(sessions_dir):
        # 兼容扁平结构 (直接就是 /1, /2 ...)
        if os.path.exists(os.path.join(RAW_DATA_ROOT, "1")):
            sessions_dir = RAW_DATA_ROOT
        else:
            print(f"Error: Sessions folder not found in {RAW_DATA_ROOT}")
            return

    # 获取所有数字命名的文件夹作为 Session ID
    session_ids = sorted([d for d in os.listdir(sessions_dir) 
                          if os.path.isdir(os.path.join(sessions_dir, d))])
    
    print(f"Found {len(session_ids)} source sessions in {sessions_dir}")
    print(f"Scanning cached files in {CACHED_PATH}...")
    print("-" * 50)

    stats = []
    
    for sid in tqdm(session_ids):
        # 构造匹配模式
        # 注意：根据之前的代码，文件名格式通常是 "{sid}_input{i}.npy"
        # 例如: "10_input0.npy"
        
        # 模式 1: 标准格式 "10_input*.npy"
        input_pattern_1 = os.path.join(CACHED_PATH, f"{sid}_input*.npy")
        label_pattern_1 = os.path.join(CACHED_PATH, f"{sid}_label*.npy")
        
        inputs = glob.glob(input_pattern_1)
        labels = glob.glob(label_pattern_1)
        
        # 模式 2: 防止之前有 session_ 前缀的情况 "session_10_input*.npy"
        # 如果模式1没找到，尝试模式2 (防止统计漏)
        if len(inputs) == 0:
            input_pattern_2 = os.path.join(CACHED_PATH, f"session_{sid}_input*.npy")
            label_pattern_2 = os.path.join(CACHED_PATH, f"session_{sid}_label*.npy")
            inputs = glob.glob(input_pattern_2)
            labels = glob.glob(label_pattern_2)

        n_inputs = len(inputs)
        n_labels = len(labels)
        
        # 状态判断
        status = "OK"
        if n_inputs == 0 and n_labels == 0:
            status = "Missing (Not Processed)"
        elif n_inputs != n_labels:
            status = "ERROR: Mismatch"
        elif n_inputs == 0:
            status = "Empty"
            
        stats.append({
            "Session": sid,
            "Input_Count": n_inputs,
            "Label_Count": n_labels,
            "Status": status
        })

    # 转换为 DataFrame 展示
    df = pd.DataFrame(stats)
    
    # === 打印统计报告 ===
    print("\n" + "=" * 30)
    print("      DATASET SCAN REPORT      ")
    print("=" * 30)
    
    total_sessions = len(df)
    processed_sessions = len(df[df['Input_Count'] > 0])
    missing_sessions = len(df[df['Input_Count'] == 0])
    mismatch_sessions = len(df[df['Input_Count'] != df['Label_Count']])
    
    print(f"Total Sessions in Source: {total_sessions}")
    print(f"Successfully Processed:   {processed_sessions}")
    print(f"Not Processed (0 files):  {missing_sessions}")
    print(f"Mismatch Errors:          {mismatch_sessions}")
    
    print("-" * 30)
    
    # 打印前 10 个处理过的样本信息
    print("Sample Data (First 10 Processed):")
    processed_df = df[df['Input_Count'] > 0].head(10)
    if not processed_df.empty:
        print(processed_df.to_string(index=False))
    else:
        print("No processed data found.")

    # 打印所有异常的 Session
    if mismatch_sessions > 0:
        print("\n[WARNING] Mismatched Sessions (Input != Label):")
        print(df[df['Input_Count'] != df['Label_Count']].to_string(index=False))
        
    # 打印所有未处理的 Session (如果太多则只打印前20个)
    if missing_sessions > 0:
        print(f"\n[INFO] Missing Sessions (Total {missing_sessions}):")
        missing_ids = df[df['Input_Count'] == 0]['Session'].tolist()
        print(f"{missing_ids[:20]} ...")

    # 保存 CSV 方便查看
    df.to_csv("dataset_check_report.csv", index=False)
    print("\nDetailed report saved to 'dataset_check_report.csv'")

if __name__ == "__main__":
    scan_and_count()
