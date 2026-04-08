import pickle
import cv2
import numpy as np
import os
from scipy import signal
import torch
from matplotlib import pyplot as plt
import csv
from collections import deque
import time


class HRVEmotionProcessor:
    def __init__(self, window_size=180, overlap=30, fs=30):
        """
        初始化HRV情绪处理器

        Args:
            window_size: 分析窗口大小（帧数）
            overlap: 窗口重叠大小（帧数）
            fs: 采样频率
        """
        self.window_size = window_size
        self.overlap = overlap
        self.fs = fs
        self.step_size = window_size - overlap

        # 个性化阈值（将在在线学习中更新）
        self.calm_threshold = 25.0  # RMSSD平静阈值
        self.low_arousal_threshold = 15.0  # 低唤醒阈值
        self.high_arousal_threshold = 8.0  # 高唤醒阈值

        # 状态跟踪
        self.current_state = "calm"
        self.hrv_history = deque(maxlen=10)  # 保存最近的HRV值
        self.baseline_rmssd = None
        self.baseline_established = False

        # 人脸检测器
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def save_hrv_to_csv(self, hrv_values, output_csv_path):
        """保存 HRV 值到 CSV 文件"""
        with open(output_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Window_Start', 'Window_End', 'RMSSD', 'SDNN', 'State', 'LF_HF_Ratio'])

            for idx, (start, end, rmssd, sdnn, state, lf_hf) in enumerate(hrv_values):
                writer.writerow([start, end, rmssd, sdnn, state, lf_hf])

    def get_all_subjects(self, data):
        """获取pickle文件中所有的subject名称"""
        if 'predictions' in data:
            return list(data['predictions'].keys())
        else:
            # 如果数据结构不同，尝试其他方式获取subject
            subjects = []
            for key in data.keys():
                if key.startswith('subject'):
                    subjects.append(key)
            return subjects

    def find_video_file_for_subject(self, subject_name, base_video_dir):
        """根据subject名称查找对应的视频文件"""
        # 实现视频文件查找逻辑
        # 这里需要根据您的文件结构来调整
        possible_paths = [
            f"{base_video_dir}/{subject_name}/vid.avi",
            f"{base_video_dir}/{subject_name}/video.avi",
            # 添加其他可能的路径模式
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None

    def concatenate_rppg_signal(self, pred_dict):
        """将多个rPPG信号片段拼接成一个长信号"""
        pred_signal = np.concatenate([
            v.numpy().flatten() if isinstance(v, torch.Tensor) else np.array(v).flatten()
            for k, v in sorted(pred_dict.items())
        ])
        return pred_signal

    def load_pickle(self, file_path):
        """加载pickle文件"""
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data

    def compute_hrv_features(self, rppg_signal):
        """计算HRV特征（RMSSD, SDNN, LF/HF Ratio）- 带调试信息"""

        print(f"\n{'=' * 50}")
        print(f"HRV特征计算调试信息")
        print(f"{'=' * 50}")

        # 1. 信号基本信息
        print(f" 输入信号信息:")
        print(f"   - 信号长度: {len(rppg_signal)} 个采样点")
        print(f"   - 信号时长: {len(rppg_signal) / self.fs:.2f} 秒")
        print(f"   - 采样频率: {self.fs} Hz")
        print(f"   - 信号范围: [{np.min(rppg_signal):.4f}, {np.max(rppg_signal):.4f}]")
        print(f"   - 信号均值: {np.mean(rppg_signal):.4f}")
        print(f"   - 信号标准差: {np.std(rppg_signal):.4f}")

        # 检查信号长度
        if len(rppg_signal) < 100:
            print(f"❌ 信号过短，无法进行HRV分析")
            return None, None, None

        try:
            # 2. 滤波前信号特征
            print(f"\n 滤波前分析:")
            original_peaks, _ = signal.find_peaks(rppg_signal, distance=int(self.fs * 0.5))
            print(f"   - 原始信号峰值数: {len(original_peaks)}")

            # 带通滤波 (0.75-2.5 Hz)
            print(f"\n 信号滤波:")
            bandpass_filter = signal.butter(2, [0.75, 2.5], btype='bandpass', fs=self.fs)
            filtered_signal = signal.filtfilt(*bandpass_filter, rppg_signal)

            print(f"   - 滤波后信号范围: [{np.min(filtered_signal):.4f}, {np.max(filtered_signal):.4f}]")
            print(f"   - 滤波后信号标准差: {np.std(filtered_signal):.4f}")

            # 3. 峰值检测
            print(f"\n 峰值检测:")
            peaks, properties = signal.find_peaks(
                filtered_signal,
                distance=int(self.fs * 0.5),  # 最低0.5秒间隔
                height=np.mean(filtered_signal) + 0.5 * np.std(filtered_signal),  # 最小高度阈值
                prominence=0.1  # 突出度阈值
            )

            print(f"   - 检测到峰值数量: {len(peaks)}")
            print(f"   - 峰值位置: {peaks[:10]}{'...' if len(peaks) > 10 else ''}")  # 显示前10个峰值位置
            print(f"   - 峰值高度范围: [{np.min(properties['peak_heights']):.4f}, {np.max(properties['peak_heights']):.4f}]")

            if len(peaks) < 2:
                print(f"❌ 峰值数量不足，需要至少2个峰值，当前只有{len(peaks)}个")
                return None, None, None

            # 4. IBI计算
            print(f"\n 心跳间期(IBI)分析:")
            ibi = np.diff(peaks) / self.fs  # 转换为秒

            print(f"   - IBI数量: {len(ibi)}")
            print(f"   - IBI范围: [{np.min(ibi):.4f}, {np.max(ibi):.4f}] 秒")
            print(f"   - IBI均值: {np.mean(ibi):.4f} 秒")
            print(f"   - 对应心率范围: [{60 / np.max(ibi):.1f}, {60 / np.min(ibi):.1f}] BPM")

            # 检查IBI合理性
            if np.max(ibi) > 3.0 or np.min(ibi) < 0.3:
                print(f"⚠️  IBI数值异常，可能峰值检测有误")

            # 5. RMSSD计算
            print(f"\n RMSSD计算:")
            successive_diffs = np.diff(ibi)
            print(f"   - 连续IBI差值数量: {len(successive_diffs)}")
            print(f"   - 差值范围: [{np.min(successive_diffs):.4f}, {np.max(successive_diffs):.4f}]")

            rmssd = np.sqrt(np.mean(successive_diffs ** 2))
            print(f"   - RMSSD原始值: {rmssd:.6f} 秒")
            print(f"   - RMSSD转换为毫秒: {rmssd * 1000:.2f} ms")

            # 6. SDNN计算
            print(f"\n SDNN计算:")
            sdnn = np.std(ibi)
            print(f"   - SDNN原始值: {sdnn:.6f} 秒")
            print(f"   - SDNN转换为毫秒: {sdnn * 1000:.2f} ms")

            # 7. 频域分析
            print(f"\n 频域分析:")
            if len(ibi) >= 64:
                # 使用Lomb-Scargle周期图处理非均匀采样的IBI数据
                freqs = np.linspace(0.003, 0.4, 200)  # 扩展频段范围

                # 时间点（使用累积时间）
                time_points = np.cumsum(ibi[:-1])

                power = signal.lombscargle(time_points, ibi[1:], freqs, normalize=True)

                # 定义频段
                vlf_band = (0.003, 0.04)
                lf_band = (0.04, 0.15)
                hf_band = (0.15, 0.4)

                # 计算各频段功率
                vlf_power = np.trapz(power[(freqs >= vlf_band[0]) & (freqs < vlf_band[1])],
                                     freqs[(freqs >= vlf_band[0]) & (freqs < vlf_band[1])])
                lf_power = np.trapz(power[(freqs >= lf_band[0]) & (freqs < lf_band[1])],
                                    freqs[(freqs >= lf_band[0]) & (freqs < lf_band[1])])
                hf_power = np.trapz(power[(freqs >= hf_band[0]) & (freqs < hf_band[1])],
                                    freqs[(freqs >= hf_band[0]) & (freqs < hf_band[1])])

                total_power = vlf_power + lf_power + hf_power

                print(f"   - VLF功率: {vlf_power:.4f}")
                print(f"   - LF功率: {lf_power:.4f}")
                print(f"   - HF功率: {hf_power:.4f}")
                print(f"   - 总功率: {total_power:.4f}")

                lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0
                print(f"   - LF/HF比率: {lf_hf_ratio:.4f}")

                # 计算归一化功率
                if total_power > 0:
                    lf_nu = lf_power / (lf_power + hf_power) * 100
                    hf_nu = hf_power / (lf_power + hf_power) * 100
                    print(f"   - LF归一化: {lf_nu:.1f} nu")
                    print(f"   - HF归一化: {hf_nu:.1f} nu")
            else:
                print(f"   - IBI数据点不足，跳过频域分析")
                lf_hf_ratio = 0

            print(f"\n HRV特征计算完成:")
            print(f"   - RMSSD: {rmssd * 1000:.2f} ms")
            print(f"   - SDNN: {sdnn * 1000:.2f} ms")
            print(f"   - LF/HF: {lf_hf_ratio:.4f}")
            print(f"{'=' * 50}")

            return rmssd, sdnn, lf_hf_ratio

        except Exception as e:
            print(f"\n❌ HRV特征计算错误: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

    def detect_faces_from_video_segment(self, video_path, start_frame, end_frame):
        """从视频片段中检测人脸"""
        cap = cv2.VideoCapture(video_path)
        faces = []

        # 设置起始帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for frame_idx in range(start_frame, min(end_frame, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
            ret, frame = cap.read()
            if not ret:
                break

            # 每隔5帧检测一次以提高效率
            if frame_idx % 5 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detected_faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )

                if len(detected_faces) > 0:
                    largest_face = max(detected_faces, key=lambda x: x[2] * x[3])
                    x, y, w, h = largest_face
                    faces.append(frame[y:y + h, x:x + w])

        cap.release()
        return faces

    def update_personalized_thresholds(self, current_rmssd):
        """在线学习更新个性化阈值 - 带调试接口"""
        if not self.baseline_established:
            # 初始基线建立
            if self.baseline_rmssd is None:
                self.baseline_rmssd = current_rmssd
            else:
                # 平滑更新基线
                self.baseline_rmssd = 0.9 * self.baseline_rmssd + 0.1 * current_rmssd

            # 收集足够数据后建立基线
            if len(self.hrv_history) >= 5:
                self.baseline_established = True
                # 基于基线设置个性化阈值 - calm阈值保持0.8不变
                self.calm_threshold = self.baseline_rmssd * 0.8  # calm阈值保持不变
                self.low_arousal_threshold = self.baseline_rmssd * 0.7  # 只调整这个
                self.high_arousal_threshold = self.baseline_rmssd * 0.4  # 只调整这个

                print(f"✅ 基线建立完成: RMSSD={self.baseline_rmssd * 1000:.2f}ms")
                print(f"✅ 个性化阈值设置:")
                print(f"   - Calm (0.8): {self.calm_threshold * 1000:.2f}ms")
                print(f"   - Low (0.7): {self.low_arousal_threshold * 1000:.2f}ms")
                print(f"   - High (0.4): {self.high_arousal_threshold * 1000:.2f}ms")
        else:
            # 持续更新基线（适应长期变化）
            self.baseline_rmssd = 0.99 * self.baseline_rmssd + 0.01 * current_rmssd

    def set_threshold_ratios(self, calm_ratio=0.8, low_ratio=0.7, high_ratio=0.4):
        """设置阈值系数（调试接口）"""
        if self.baseline_established:
            self.calm_threshold = self.baseline_rmssd * calm_ratio
            self.low_arousal_threshold = self.baseline_rmssd * low_ratio
            self.high_arousal_threshold = self.baseline_rmssd * high_ratio

            print(f" 手动调整阈值系数:")
            print(f"   - Calm ratio: {calm_ratio} -> {self.calm_threshold * 1000:.2f}ms")
            print(f"   - Low ratio: {low_ratio} -> {self.low_arousal_threshold * 1000:.2f}ms")
            print(f"   - High ratio: {high_ratio} -> {self.high_arousal_threshold * 1000:.2f}ms")
        else:
            print("️ 基线尚未建立，无法调整阈值")

    def get_current_thresholds(self):
        """获取当前阈值信息（调试接口）"""
        return {
            'baseline_rmssd_ms': self.baseline_rmssd * 1000 if self.baseline_rmssd else None,
            'calm_threshold_ms': self.calm_threshold * 1000,
            'low_arousal_threshold_ms': self.low_arousal_threshold * 1000,
            'high_arousal_threshold_ms': self.high_arousal_threshold * 1000,
            'baseline_established': self.baseline_established
        }

    def print_threshold_info(self):
        """打印当前阈值信息（调试接口）"""
        thresholds = self.get_current_thresholds()
        print(f"\n 当前阈值信息:")
        print(f"   - 基线建立: {thresholds['baseline_established']}")
        if thresholds['baseline_established']:
            print(f"   - 基线RMSSD: {thresholds['baseline_rmssd_ms']:.2f}ms")
            print(f"   - Calm阈值: {thresholds['calm_threshold_ms']:.2f}ms")
            print(f"   - Low阈值: {thresholds['low_arousal_threshold_ms']:.2f}ms")
            print(f"   - High阈值: {thresholds['high_arousal_threshold_ms']:.2f}ms")

    def determine_emotional_state(self, rmssd, lf_hf_ratio):
        """根据HRV特征确定情绪状态 - 带详细调试信息"""
        if rmssd is None:
            return "unknown"

        # 记录HRV历史
        self.hrv_history.append(rmssd)

        # 更新个性化阈值
        self.update_personalized_thresholds(rmssd)

        if not self.baseline_established:
            return "calibrating"

        rmssd_ms = rmssd * 1000

        # 基于个性化阈值判断状态
        if rmssd >= self.calm_threshold:
            state = "calm"
            reason = f"RMSSD({rmssd_ms:.2f}ms) >= Calm阈值({self.calm_threshold * 1000:.2f}ms)"
        elif rmssd >= self.low_arousal_threshold:
            state = "low_arousal"
            reason = f"Low阈值({self.low_arousal_threshold * 1000:.2f}ms) <= RMSSD({rmssd_ms:.2f}ms) < Calm阈值({self.calm_threshold * 1000:.2f}ms)"
        else:
            state = "high_arousal"
            reason = f"RMSSD({rmssd_ms:.2f}ms) < Low阈值({self.low_arousal_threshold * 1000:.2f}ms)"

        print(f" 状态判断: {state} | {reason}")
        return state

    def save_faces(self, faces, output_path, state, window_idx):
        """保存人脸图片"""
        os.makedirs(output_path, exist_ok=True)
        for idx, face in enumerate(faces):
            cv2.imwrite(f"{output_path}/{state}_window{window_idx}_face{idx}.jpg", face)

    def process_rppg_with_sliding_window(self, pickle_file, base_video_dir, output_path, output_csv_dir):
        """处理所有subject的rPPG信号"""
        # 加载数据
        data = self.load_pickle(pickle_file)

        # 获取所有subject
        subjects = self.get_all_subjects(data)
        print(f"找到 {len(subjects)} 个subject: {subjects}")

        # 为每个subject创建独立的处理器实例
        all_results = {}
        for subject in subjects:
            print(f"\n{'=' * 60}")
            print(f"处理 Subject: {subject}")
            print(f"{'=' * 60}")

            # 为每个subject创建新的处理器实例（重置状态）
            subject_processor = HRVEmotionProcessor(self.window_size, self.overlap, self.fs)

            # 查找对应的视频文件
            video_file = self.find_video_file_for_subject(subject, base_video_dir)

            # 处理当前subject
            subject_results = subject_processor.process_single_subject(data, subject, video_file, output_path,
                                                                       output_csv_dir)
            all_results[subject] = subject_results

        return all_results

    def process_single_subject(self, data, subject_name, video_file, output_path, output_csv_dir):
        """处理单个subject的数据"""
        # 创建subject专用输出目录
        subject_output_path = os.path.join(output_path, subject_name)
        subject_csv_path = os.path.join(output_csv_dir, f"{subject_name}_HRV_values.csv")

        # 获取当前subject的rPPG信号
        if subject_name in data['predictions']:
            subject_predictions = data['predictions'][subject_name]
            rppg_signal = self.concatenate_rppg_signal(subject_predictions)
        else:
            print(f"❌ 在数据中找不到 {subject_name}")
            return None

        # 原有的滑动窗口处理逻辑（针对单个subject）
        total_frames = len(rppg_signal)
        hrv_results = []

        print(f"处理 {subject_name}: {total_frames} 帧，窗口大小 {self.window_size}")

        # 滑动窗口处理（保持原有逻辑）
        for start_idx in range(0, total_frames - self.window_size + 1, self.step_size):
            end_idx = start_idx + self.window_size
            window_signal = rppg_signal[start_idx:end_idx]

            # 计算HRV特征
            rmssd, sdnn, lf_hf_ratio = self.compute_hrv_features(window_signal)

            if rmssd is not None:
                # 确定情绪状态
                state = self.determine_emotional_state(rmssd, lf_hf_ratio)

                # 记录结果
                hrv_results.append((start_idx, end_idx, rmssd, sdnn, state, lf_hf_ratio))

                # 根据状态触发人脸抓拍
                if state in ["low_arousal", "high_arousal"] and video_file:
                    faces = self.detect_faces_from_video_segment(video_file, start_idx, end_idx)
                    if faces:
                        self.save_faces(faces, os.path.join(subject_output_path, state), state, start_idx)

        # 保存HRV结果到subject专用CSV
        self.save_hrv_to_csv(hrv_results, subject_csv_path)

        # 打印统计信息
        states = [result[4] for result in hrv_results]
        state_counts = {state: states.count(state) for state in set(states)}
        print(f"{subject_name} 状态分布: {state_counts}")

        return hrv_results


# 使用示例
if __name__ == "__main__":
    pickle_file = './runs/exp/UBFC-rPPG_SizeW72_SizeH72_ClipLength180_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse/saved_test_outputs/PURE_TSCAN_UBFC-rPPG_outputs.pickle'
    base_video_dir = '/root/autodl-tmp/Code/rPPG-Toolbox/data/UBFC-rPPG'  # 视频文件根目录
    output_path = './output'
    output_csv_dir = './output/csv'  # CSV文件输出目录

    # 确保主输出目录存在
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(output_csv_dir, exist_ok=True)

    processor = HRVEmotionProcessor(window_size=180, overlap=60, fs=30)
    # 在处理过程中随时调整阈值
    processor.print_threshold_info()  # 查看当前阈值
    # 或者使用更宽松的阈值
    processor.set_threshold_ratios(calm_ratio=0.8, low_ratio=0.5, high_ratio=0.2)

    results = processor.process_rppg_with_sliding_window(
        pickle_file, base_video_dir, output_path, output_csv_dir
    )
