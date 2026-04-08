import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pyedflib

def check_raw_bdf(session_id, data_root):
    # 1. 寻找 Session 路径
    # 兼容 Sessions/10 或 直接 10 的结构
    search_path = os.path.join(data_root, "Sessions", str(session_id))
    if not os.path.exists(search_path):
        search_path = os.path.join(data_root, str(session_id))
    
    if not os.path.exists(search_path):
        print(f"Error: Session {session_id} not found in {data_root}")
        return

    # 2. 寻找 BDF 文件
    bdf_files = glob.glob(os.path.join(search_path, "*.bdf"))
    if not bdf_files:
        print("Error: No BDF file found.")
        return
    
    bdf_path = bdf_files[0]
    print(f"Checking file: {bdf_path}")

    # 3. 读取 BDF
    try:
        f = pyedflib.EdfReader(bdf_path)
    except ImportError:
        print("Error: Please pip install pyedflib")
        return

    signal_labels = f.getSignalLabels()
    print(f"Available Channels: {signal_labels}")

    # 寻找 EXG2 或 ECG
    target_idx = -1
    target_label = ""
    for i, label in enumerate(signal_labels):
        if 'EXG2' in label.upper() or 'ECG' in label.upper():
            target_idx = i
            target_label = label
            break
    
    if target_idx == -1:
        # Fallback to anything just to see the length
        target_idx = 0
        target_label = signal_labels[0]
        print("Warning: EXG2/ECG not found, checking first channel.")

    # 读取信号
    sig = f.readSignal(target_idx)
    fs = f.getSampleFrequency(target_idx)
    f._close()

    duration = len(sig) / fs
    print(f"Channel: {target_label}")
    print(f"Sampling Rate: {fs} Hz")
    print(f"Total Samples: {len(sig)}")
    print(f"Total Duration: {duration:.2f} seconds")

    # 4. 绘图
    plt.figure(figsize=(12, 8))

    # 子图 1: 全长信号
    plt.subplot(2, 1, 1)
    time_axis = np.arange(len(sig)) / fs
    plt.plot(time_axis, sig, color='blue')
    plt.title(f"Full Raw Signal (Duration: {duration:.2f}s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # 子图 2: 最后 5 秒特写
    plt.subplot(2, 1, 2)
    # 取最后 5 秒的点数
    last_seconds = 5
    last_samples = int(last_seconds * fs)
    if len(sig) > last_samples:
        snippet = sig[-last_samples:]
        snippet_time = time_axis[-last_samples:]
    else:
        snippet = sig
        snippet_time = time_axis

    plt.plot(snippet_time, snippet, color='red')
    plt.title(f"Zoom in: Last {last_seconds} seconds")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    save_name = f"check_raw_bdf_{session_id}.png"
    plt.tight_layout()
    plt.savefig(save_name)
    print(f"\nSaved plot to: {save_name}")
    print("-" * 30)
    print("分析指南:")
    print("1. 看图2(红色)：最后几秒是否有巨大的尖峰/漂移？")
    print(f"2. 你的视频长度是 30秒。")
    print(f"3. BDF长度是 {duration:.2f}秒。")
    print(f"   如果 BDF长度 > 30秒，且尖峰出现在第 30秒以后，")
    print(f"   说明你的【尾部对齐】策略把多余的垃圾时间包含进来了。")

if __name__ == "__main__":
    # 修改这里为你的数据集根目录
    DATA_ROOT = "/dataset/MAHNOB-HCI" 
    
    # 修改为你出问题的那个 Session ID (比如 10)
    SESSION_ID = "10" 
    
    check_raw_bdf(SESSION_ID, DATA_ROOT)
