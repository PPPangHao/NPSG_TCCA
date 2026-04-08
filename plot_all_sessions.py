#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch  # for torch.Tensor support
from scipy.signal import butter, filtfilt


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def bandpass_filter(x, fs=30.0):
    ny = fs / 2
    b, a = butter(4, [0.75/ny, 2.5/ny], btype="bandpass")
    return filtfilt(b, a, x)
    
# --------------------------------------------
# 从一个 chunk 中提取一维信号
# --------------------------------------------
def extract_signal(obj):
    """从 chunk 中提取一维波形，自动适配各种结构"""

    # torch tensor
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().astype(float).flatten()

    # numpy array or list
    if isinstance(obj, (list, np.ndarray)):
        return np.array(obj).astype(float).flatten()

    # dict: 取常见字段
    if isinstance(obj, dict):

        # 尝试常见字段
        for k in ["bvp", "pred", "prediction", "signal", "data", "value"]:
            if k in obj:
                return extract_signal(obj[k])

        # fallback: 查找第一个 array-like
        for v in obj.values():
            if isinstance(v, (list, np.ndarray, torch.Tensor)):
                return extract_signal(v)

        raise ValueError(f"无法从 dict 中提取信号: {obj}")

    # 单个 HR 数字
    if isinstance(obj, (int, float)):
        return np.ones(300) * float(obj)

    raise ValueError(f"未知类型 {type(obj)} 内容={obj}")


# --------------------------------------------
# 自动从各种结构中提取 chunk 列表
# --------------------------------------------
def extract_chunks(obj):
    """
    自动判断 predictions[session] 的结构，
    并提取 chunk 列表。
    """

    # 1. 本身就是 list → chunk list
    if isinstance(obj, list):
        return obj

    # 2. dict 情况
    if isinstance(obj, dict):

        # 2.1 常见字段包含 chunk list
        common = ["chunks", "predictions", "labels", "bvp_chunks", "data"]
        for k in common:
            if k in obj and isinstance(obj[k], list):
                return obj[k]

        # 2.2 数字 key：0,1,2,... 才视为 chunk dict
        keys = list(obj.keys())
        # 判断 key 是否全是数字/数字字符串
        def is_int_like(x):
            try:
                int(x)
                return True
            except:
                return False

        if all(is_int_like(k) for k in keys):
            sorted_keys = sorted(keys, key=lambda x: int(x))
            return [obj[k] for k in sorted_keys]

        # 2.3 fallback: 取第一个 list（适合 data={"input":[...], "label":[...]})
        for v in obj.values():
            if isinstance(v, list):
                return v

        raise ValueError(f"无法识别 chunk 列表结构：{obj}")

    raise ValueError(f"不支持类型：{type(obj)}")


# --------------------------------------------
# 拼接所有 chunk 为完整波形
# --------------------------------------------
def concat_chunks(chunk_list):
    signals = []
    for chunk in chunk_list:
        sig = extract_signal(chunk)
        signals.append(sig)
    if len(signals) == 0:
        return np.array([])
    return np.concatenate(signals)


# --------------------------------------------
# 绘制一个 Session 的完整 rPPG + GT
# --------------------------------------------
def plot_session(pred_chunks, label_chunks, sid, save_dir):
    pred_full = concat_chunks(pred_chunks)
    label_full = concat_chunks(label_chunks)

    min_len = min(len(pred_full), len(label_full))
    pred_full = pred_full[:min_len]
    label_full = label_full[:min_len]

    t = np.arange(min_len)

    plt.figure(figsize=(12, 5))
    plt.plot(t, pred_full, color="red", linewidth=1, label="Predicted")
    plt.plot(t, label_full, color="blue", linewidth=1, alpha=0.6, label="Ground Truth")
    plt.title(f"Session {sid} - Full Signal")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)

    ensure_dir(save_dir)
    out = os.path.abspath(os.path.join(save_dir, f"session_{sid}.png"))
    plt.savefig(out, dpi=200)
    plt.close()

    print(f"[Saved] {out}")


# --------------------------------------------
# 主函数
# --------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle", required=True)
    parser.add_argument("--save_dir", default="plots_full")
    args = parser.parse_args()

    print(f"[INFO] Loading pickle: {args.pickle}")
    data = pickle.load(open(args.pickle, "rb"))

    predictions = data["predictions"]
    labels = data["labels"]

    for sid in sorted(predictions.keys(), key=lambda x: int(x)):
        sid_str = str(sid)

        try:
            pred_chunks = extract_chunks(predictions[sid_str])
            label_chunks = extract_chunks(labels[sid_str])
        except Exception as e:
            print(f"[WARN] Session {sid_str} 解析失败: {e}")
            continue

        plot_session(pred_chunks, label_chunks, sid_str, args.save_dir)

    print("\n[INFO] 所有 session 绘制完成！")


if __name__ == "__main__":
    main()
