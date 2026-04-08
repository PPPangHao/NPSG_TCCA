import os
import csv
import pickle
import numpy as np
from stable_hrv_arousal import StableHRVArousalExtractor
import torch


# ----------------------------------------------------------------------
# 读取你的 pickle 并合并 rPPG chunk
# ----------------------------------------------------------------------
def load_rppg_from_pickle(pickle_path, fs=30):
    """
    加载并合并每个 session 的 rPPG 信号，顺便打印每个 session 的长度。
    fs: rPPG 采样率，用于换算秒。
    """
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    predictions = data["predictions"]
    sessions = {}

    print("\n===== RPPG 信号长度统计（合并后） =====")

    for session_id, chunks in predictions.items():
        merged = []

        # 逐 chunk 合并
        for k in sorted(chunks.keys(), key=lambda x: int(x)):
            arr = chunks[k]

            # 处理 torch.Tensor / list / numpy
            if isinstance(arr, torch.Tensor):
                arr = arr.cpu().numpy()
            arr = np.array(arr).flatten()

            merged.append(arr)

        # 合并成一个 session 的完整信号
        if merged:
            signal = np.concatenate(merged)
        else:
            signal = np.array([])

        sessions[session_id] = signal

        # 打印每个 session 的长度（帧 & 秒）
        length_frames = len(signal)
        length_seconds = length_frames / fs

        print(f"Session {session_id}: {length_frames} frames ({length_seconds:.2f} sec)")

    print("===== 统计完成 =====\n")

    return sessions


# ----------------------------------------------------------------------
# 保存 CSV
# ----------------------------------------------------------------------
def save_session_csv(session_id, results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{session_id}.csv")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["session_id", "start_frame", "end_frame", "rmssd_ms", "arousal_state"])

        for start, end, rmssd, state in results:
            rmssd_ms = "" if rmssd is None else round(rmssd * 1000, 3)
            writer.writerow([session_id, start, end, rmssd_ms, state])

    print(f"[Saved] {out_path}")


def save_global_csv(all_results, out_dir):
    out_path = os.path.join(out_dir, "all_sessions_hrv.csv")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["session_id", "start_frame", "end_frame", "rmssd_ms", "arousal_state"])

        for session_id, session_results in all_results.items():
            for start, end, rmssd, state in session_results:
                rmssd_ms = "" if rmssd is None else round(rmssd * 1000, 3)
                writer.writerow([session_id, start, end, rmssd_ms, state])

    print(f"[Saved Global CSV] {out_path}")


# ----------------------------------------------------------------------
# 主函数
# ----------------------------------------------------------------------
def main():
    pickle_path = "./accurate_pickle/accurate.pickle"           # TODO: 修改为你的 pickle 路径
    output_dir = "./hrv_output"                 # 保存输出的目录

    print("Loading rPPG pickle...")
    sessions_rppg = load_rppg_from_pickle(pickle_path)
    print(f"Loaded {len(sessions_rppg)} sessions")

    extractor = StableHRVArousalExtractor(fs=30, window_sec=30, overlap_sec=15)

    all_results = {}

    for session_id, signal in sessions_rppg.items():
        print(f"\n=== Processing session {session_id} ===")
        if len(signal) < 200:
            print("  Skipped (signal too short)\n")
            continue

        results = extractor.process(signal)
        all_results[session_id] = results

        save_session_csv(session_id, results, output_dir)

    save_global_csv(all_results, output_dir)

    print("\nProcessing completed!")


if __name__ == "__main__":
    main()

