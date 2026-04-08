import os
import glob
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from scipy import signal
import pyedflib
from tqdm import tqdm
import shutil

# ================= 配置区域 =================
# 1. 原始数据集根目录
DATA_ROOT = "/dataset/MAHNOB-HCI" 

# 2. 预处理缓存目录 (你的 input_*.npy 所在的目录)
# 请务必确认这个路径，脚本会覆盖这里的 label 文件
CACHED_PATH = "/root/autodl-tmp/Code/rPPG-Toolbox/data/MAHNOB-HCI/PreprocessedData/MAHNOB-HCI_SizeW72_SizeH72_ClipLength180_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse"

# 3. 参数设置 (必须与你之前预处理视频时的 Config 一致)
FS = 61.0            # 目标采样率 (视频帧率)
CHUNK_LENGTH = 180   # 切片长度
LABEL_TYPE = "DiffNormalized" # "DiffNormalized" 或 "Standardized"

# 4. 输出图片的文件夹
DEBUG_PLOT_DIR = "./debug_labels_check"
# ===========================================

def get_sessions(data_root):
    sessions_path = os.path.join(data_root, "Sessions")
    if not os.path.exists(sessions_path):
        sessions_path = data_root # 兼容扁平结构
    
    # 寻找所有包含 session.xml 的文件夹
    session_dirs = []
    candidates = sorted(glob.glob(os.path.join(sessions_path, "*")))
    for d in candidates:
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "session.xml")):
            session_dirs.append(d)
    return session_dirs

def read_xml_duration(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        return float(root.attrib.get('cutLenSec', 0))
    except:
        return 0

def get_video_meta(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps < 1 or fps > 120: fps = 61.0
    cap.release()
    return frames, fps

def read_bdf_exg2(bdf_path):
    try:
        f = pyedflib.EdfReader(bdf_path)
    except Exception as e:
        print(f"Error opening BDF: {e}")
        return None, 0
        
    labels = f.getSignalLabels()
    target_idx = -1
    
    # 优先级查找
    for i, l in enumerate(labels):
        if 'EXG2' in l.upper(): # 论文推荐
            target_idx = i; break
    if target_idx == -1:
        for i, l in enumerate(labels):
            if 'ECG' in l.upper() or 'EXG1' in l.upper():
                target_idx = i; break
    if target_idx == -1:
        for i, l in enumerate(labels):
            if 'PLET' in l.upper() or 'BVP' in l.upper():
                target_idx = i; break
                
    if target_idx == -1:
        f._close()
        return None, 0
        
    sig = f.readSignal(target_idx)
    fs = f.getSampleFrequency(target_idx)
    f._close()
    return np.nan_to_num(sig), fs

def bandpass_filter(data, fs):
    nyq = fs / 2
    # 0.75 - 2.5 Hz (45 - 150 BPM)
    b, a = signal.butter(4, [0.75/nyq, 2.5/nyq], btype='bandpass')
    return signal.filtfilt(b, a, data)

def diff_normalize_label(label):
    diff = np.diff(label)
    std = np.std(diff)
    if std == 0: std = 1
    norm = diff / std
    # 补一个 0 保持长度一致
    norm = np.append(norm, 0)
    return norm

def standardized_label(label):
    return (label - np.mean(label)) / np.std(label)

def process_single_session(sess_path):
    sess_id = os.path.basename(sess_path)
    
    # 1. 找文件
    avis = glob.glob(os.path.join(sess_path, "*.avi"))
    bdfs = glob.glob(os.path.join(sess_path, "*.bdf"))
    xml = os.path.join(sess_path, "session.xml")
    
    if not avis or not bdfs:
        return False, "Missing files"
        
    # 2. 获取元数据
    n_frames, fps = get_video_meta(avis[0])
    if n_frames == 0: return False, "Video empty"
    
    video_dur = n_frames / fps
    valid_dur = read_xml_duration(xml)
    if valid_dur == 0: valid_dur = 999999
    
    # 3. 读取 BDF 并对齐
    raw_sig, bdf_fs = read_bdf_exg2(bdfs[0])
    if raw_sig is None: return False, "BDF error"
    
    # === 核心对齐逻辑 ===
    end_idx = int(valid_dur * bdf_fs)
    end_idx = min(end_idx, len(raw_sig))
    needed_samples = int(video_dur * bdf_fs)
    start_idx = end_idx - needed_samples
    
    if start_idx < 0:
        # 视频比XML记录时长还长？取开头
        start_idx = 0
        cropped = raw_sig[:end_idx]
    else:
        cropped = raw_sig[start_idx:end_idx]
        
    # 4. 重采样
    # 强制拉伸到与视频帧数一致
    resampled = np.interp(
        np.linspace(0, len(cropped), n_frames),
        np.arange(len(cropped)),
        cropped
    )
    
    # 5. 去噪与滤波 (Raw -> Clean Wave)
    # Clip 1% - 99%
    lower = np.percentile(resampled, 1)
    upper = np.percentile(resampled, 99)
    clipped = np.clip(resampled, lower, upper)
    
    # Bandpass 0.75-2.5Hz
    filtered = bandpass_filter(clipped, FS)
    
    # 6. 归一化 (Clean Wave -> Model Label)
    if LABEL_TYPE == "DiffNormalized":
        final_label = diff_normalize_label(filtered)
    else:
        final_label = standardized_label(filtered)
        
    # 7. 保存图片 (Visual Check)
    plot_path = os.path.join(DEBUG_PLOT_DIR, f"check_{sess_id}.png")
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(filtered, color='green', label='Filtered Signal (0.75-2.5Hz)')
    plt.title(f"Session {sess_id}: Cleaned Physiological Signal")
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(final_label, color='blue', label=f'Final Label ({LABEL_TYPE})')
    plt.title("Final Normalized Label (Input to Loss Function)")
    plt.xlabel("Frames")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    
    # 8. 切片并保存 NPY
    n_chunks = n_frames // CHUNK_LENGTH
    
    for i in range(n_chunks):
        chunk_data = final_label[i*CHUNK_LENGTH : (i+1)*CHUNK_LENGTH]
        
        # 构造文件名: session_10_label0.npy
        # 注意：这里假设你的文件名格式是 {dir_name}_label{i}.npy
        # 如果你的预处理文件名带有 "session_" 前缀，请相应修改
        save_name = f"{sess_id}_label{i}.npy" 
        save_path = os.path.join(CACHED_PATH, save_name)
        
        np.save(save_path, chunk_data)
        
    return True, f"Saved {n_chunks} chunks"

def main():
    if os.path.exists(DEBUG_PLOT_DIR):
        shutil.rmtree(DEBUG_PLOT_DIR)
    os.makedirs(DEBUG_PLOT_DIR)
    
    if not os.path.exists(CACHED_PATH):
        print(f"Error: Cached path {CACHED_PATH} does not exist.")
        return

    sessions = get_sessions(DATA_ROOT)
    print(f"Found {len(sessions)} sessions.")
    print(f"Processing Labels Only...")
    print(f"  - Alignment: XML 'cutLenSec' Guided")
    print(f"  - Filter: 0.75 - 2.5 Hz")
    print(f"  - Label Type: {LABEL_TYPE}")
    print("-" * 40)

    success_count = 0
    for sess in tqdm(sessions):
        ok, msg = process_single_session(sess)
        if ok:
            success_count += 1
        else:
            print(f"Skipped {os.path.basename(sess)}: {msg}")

    print("-" * 40)
    print(f"Finished. Successfully updated {success_count} sessions.")
    print(f"Please check images in: {DEBUG_PLOT_DIR}")
    print("If images look like good sine waves, start training immediately!")

if __name__ == "__main__":
    main()
