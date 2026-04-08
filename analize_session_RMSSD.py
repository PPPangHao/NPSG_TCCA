import pickle
import numpy as np
import os
import csv
from scipy import signal
import torch

class HRVEmotionProcessor:
    def __init__(self, fs=30):
        self.fs = fs  # 采样频率

    def compute_rmssd(self, rppg_signal):
        """计算 RMSSD (Root Mean Square of Successive Differences)"""
        if len(rppg_signal) < 2:
            return None

        # 计算连续采样点之间的差值
        diff_signal = np.diff(rppg_signal)
        # 计算RMSSD
        rmssd = np.sqrt(np.mean(diff_signal ** 2))
        return rmssd

    def load_pickle(self, file_path):
        """加载pickle文件"""
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data

    def save_rmssd_to_csv(self, rmssd_values, output_csv_path):
        """保存 RMSSD 值到 CSV 文件"""
        with open(output_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['session_id', 'rmssd'])

            for sess_id, rmssd in rmssd_values:
                writer.writerow([sess_id, rmssd])

    def process_rppg_and_save_rmssd(self, pickle_file, output_csv_path):
        """处理所有 subject 的 rPPG 信号并计算 RMSSD"""
        data = self.load_pickle(pickle_file)

        # 获取所有 subject
        subjects = data['predictions'].keys()
        rmssd_values = []

        # 遍历所有 subject 计算 RMSSD
        for subject in subjects:
            rppg_signal = self.concatenate_rppg_signal(data['predictions'][subject])
            rmssd = self.compute_rmssd(rppg_signal)

            if rmssd is not None:
                rmssd_values.append((subject, rmssd))

        # 保存 RMSSD 值到 CSV
        self.save_rmssd_to_csv(rmssd_values, output_csv_path)

    def concatenate_rppg_signal(self, pred_dict):
        """将多个 rPPG 信号片段拼接成一个长信号"""
        pred_signal = np.concatenate([
            v.numpy().flatten() if isinstance(v, torch.Tensor) else np.array(v).flatten()
            for k, v in sorted(pred_dict.items())
        ])
        return pred_signal


# 使用示例
if __name__ == "__main__":
    pickle_file = './accurate_pickle/accurate.pickle'  # 替换为你的 pickle 文件路径
    output_csv_path = './output/rmssd_values.csv'  # 输出的 CSV 文件路径

    processor = HRVEmotionProcessor(fs=30)  # 假设采样频率为30Hz
    processor.process_rppg_and_save_rmssd(pickle_file, output_csv_path)

    print(f"RMSSD values have been saved to {output_csv_path}")

