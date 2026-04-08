# diagnostic_labels_fixed.py
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def check_label_values():
    """检查预处理标签的实际数值"""
    # 使用你提供的路径
    base_path = "/root/autodl-tmp/Code/rPPG-Toolbox/data/MAHNOB-HCI/PreprocessedData/MAHNOB-HCI_SizeW72_SizeH72_ClipLength180_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse"
    
    print(f"Checking path: {base_path}")
    
    # 查找标签文件
    label_files = glob.glob(os.path.join(base_path, "*_label*.npy"))
    
    print(f"Found {len(label_files)} label files")
    
    if len(label_files) == 0:
        print("No label files found! Checking for input files...")
        input_files = glob.glob(os.path.join(base_path, "*_input*.npy"))
        print(f"Found {len(input_files)} input files")
        return
    
    # 随机选择几个标签文件（按修改时间排序，获取最新的）
    label_files = sorted(label_files, key=os.path.getmtime, reverse=True)[:20]
    
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    all_labels = []
    
    for i, label_file in enumerate(label_files[:12]):
        try:
            label = np.load(label_file)
            all_labels.append(label)
            
            # 绘制标签信号
            ax = axes[i]
            ax.plot(label[:180])  # 显示整个chunk
            ax.set_title(f"Label {i+1}: {os.path.basename(label_file)}")
            ax.set_xlabel("Time (frames)")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)
            
            # 打印统计信息
            print(f"\n{'='*50}")
            print(f"Label {i+1}: {os.path.basename(label_file)}")
            print(f"  Shape: {label.shape}")
            print(f"  Range: [{label.min():.6f}, {label.max():.6f}]")
            print(f"  Mean: {label.mean():.6f}, Std: {label.std():.6f}")
            print(f"  First 5 values: {label[:5]}")
            
            # 计算潜在心率
            hr_estimate = estimate_heart_rate(label, fs=30)
            print(f"  Estimated HR: {hr_estimate:.1f} bpm")
            
            # 检查是否都是接近0的值
            if abs(label.mean()) < 0.01 and abs(label.std() - 1.0) < 0.1:
                print(f"  ⚠️  This label appears to be normalized (mean~0, std~1)")
            
        except Exception as e:
            print(f"Error loading {label_file}: {e}")
    
    plt.tight_layout()
    plt.savefig("label_analysis.png", dpi=150)
    plt.show()
    
    # 分析所有标签
    if all_labels:
        all_labels_concat = np.concatenate(all_labels)
        print(f"\n{'='*60}")
        print(f"=== Overall Statistics (from {len(all_labels)} samples) ===")
        print(f"Total values: {len(all_labels_concat):,}")
        print(f"Global range: [{all_labels_concat.min():.6f}, {all_labels_concat.max():.6f}]")
        print(f"Global mean: {all_labels_concat.mean():.6f}")
        print(f"Global std: {all_labels_concat.std():.6f}")
        print(f"Global variance: {all_labels_concat.var():.6f}")
        
        # 检查是否都是接近0的值
        mean_abs = np.mean(np.abs(all_labels_concat))
        print(f"Mean absolute value: {mean_abs:.6f}")
        
        if np.allclose(all_labels_concat.mean(), 0, atol=0.01) and np.allclose(all_labels_concat.std(), 1, atol=0.1):
            print("⚠️  ALL labels appear to be normalized (mean~0, std~1)")
            print("This means labels are NOT in BPM units!")
        else:
            print("Labels are NOT normalized to mean=0, std=1")
        
        # 检查是否有异常值
        q1, q3 = np.percentile(all_labels_concat, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = all_labels_concat[(all_labels_concat < lower_bound) | (all_labels_concat > upper_bound)]
        print(f"Outliers (>1.5*IQR): {len(outliers)} ({len(outliers)/len(all_labels_concat)*100:.2f}%)")
        
        # 直方图
        plt.figure(figsize=(10, 6))
        plt.hist(all_labels_concat, bins=100, alpha=0.7, edgecolor='black')
        plt.title("Distribution of All Label Values")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        plt.savefig("label_distribution.png", dpi=150)
        plt.show()

def estimate_heart_rate(signal, fs=30):
    """从信号估计心率"""
    if len(signal) < fs * 2:
        return 0
    
    try:
        # 去趋势
        signal_detrended = signal - np.mean(signal)
        
        # FFT分析
        fft = np.abs(np.fft.rfft(signal_detrended))
        freqs = np.fft.rfftfreq(len(signal_detrended), d=1/fs)
        
        # 只关心0.7-4Hz范围（42-240 bpm）
        mask = (freqs >= 0.7) & (freqs <= 4)
        if np.sum(mask) == 0:
            return 0
        
        # 找到最大振幅的频率
        main_freq = freqs[mask][np.argmax(fft[mask])]
        hr_bpm = main_freq * 60
        
        return hr_bpm
    except:
        return 0

def check_specific_label_file():
    """检查特定的标签文件"""
    specific_file = "/root/autodl-tmp/Code/rPPG-Toolbox/data/MAHNOB-HCI/PreprocessedData/MAHNOB-HCI_SizeW72_SizeH72_ClipLength180_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse/session_950_label9.npy"
    
    print(f"\n{'='*60}")
    print(f"Checking specific file: {os.path.basename(specific_file)}")
    
    try:
        label = np.load(specific_file)
        
        print(f"Shape: {label.shape}")
        print(f"First 20 values: {label[:20]}")
        print(f"Last 20 values: {label[-20:]}")
        print(f"Min: {label.min():.6f}, Max: {label.max():.6f}")
        print(f"Mean: {label.mean():.6f}, Std: {label.std():.6f}")
        
        # 绘制详细分析
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 1. 时域信号
        axes[0, 0].plot(label)
        axes[0, 0].set_title(f"Time Domain: {os.path.basename(specific_file)}")
        axes[0, 0].set_xlabel("Time (frames)")
        axes[0, 0].set_ylabel("Value")
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 直方图
        axes[0, 1].hist(label, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title("Value Distribution")
        axes[0, 1].set_xlabel("Value")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. FFT分析
        fs = 30
        fft = np.abs(np.fft.rfft(label))
        freqs = np.fft.rfftfreq(len(label), d=1/fs)
        
        # 找到主频
        mask = (freqs >= 0.7) & (freqs <= 4)
        if np.sum(mask) > 0:
            main_freq = freqs[mask][np.argmax(fft[mask])]
            hr_bpm = main_freq * 60
        else:
            main_freq = 0
            hr_bpm = 0
        
        axes[1, 0].plot(freqs[:len(freqs)//2], fft[:len(fft)//2])
        axes[1, 0].axvline(main_freq, color='red', linestyle='--', label=f'Peak: {hr_bpm:.1f} bpm')
        axes[1, 0].set_title(f"Frequency Domain (HR: {hr_bpm:.1f} bpm)")
        axes[1, 0].set_xlabel("Frequency (Hz)")
        axes[1, 0].set_ylabel("Amplitude")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 自相关
        autocorr = np.correlate(label, label, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        axes[1, 1].plot(autocorr[:200])
        axes[1, 1].set_title("Autocorrelation")
        axes[1, 1].set_xlabel("Lag")
        axes[1, 1].set_ylabel("Correlation")
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("specific_label_analysis.png", dpi=150)
        plt.show()
        
        print(f"\nEstimated HR from FFT: {hr_bpm:.1f} bpm")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("MAHNOB-HCI Label Analysis Tool")
    print("="*60)
    
    # 检查所有标签
    check_label_values()
    
    # 检查特定文件
    check_specific_label_file()