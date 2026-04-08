#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pickle
import pandas as pd
import os
from typing import Any, Dict


def filter_pickle_by_sessions(csv_path: str,
                              pickle_path: str,
                              out_pickle_path: str) -> None:
    """
    根据 CSV 中 Is_Accurate==1 的 Session，
    从 pickle 的 predictions / labels 中筛选对应 session 另存。

    约定：
      - CSV 至少包含列：Session, Is_Accurate
      - pickle 结构类似：
            {
                "predictions": {session_id: ...},
                "labels":      {session_id: ...},
                "label_type":  "HR"    # 其他字段原样保留
            }
        其中 session_id 可以是 str 或 int，脚本会自动用 str 对齐。
    """

    # 1. 读取 CSV
    print(f"[INFO] Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path, engine="python", sep=None)  # 自动识别分隔符

    if "Session" not in df.columns or "Is_Accurate" not in df.columns:
        raise ValueError("CSV 中必须包含列：'Session' 和 'Is_Accurate'")

    # 2. 过滤出准确的 Session（Is_Accurate == 1）
    acc_df = df[df["Is_Accurate"] == 1]
    if acc_df.empty:
        print("[WARN] 没有 Is_Accurate == 1 的样本，退出。")
        return

    # 把 Session 转成字符串集合，方便和 pickle 里的 key 对齐
    accurate_sessions = set(acc_df["Session"].astype(str).tolist())
    print(f"[INFO] 准确样本数量: {len(accurate_sessions)}")
    print(f"[DEBUG] 准确 Session 示例: {list(accurate_sessions)[:10]}")

    # 3. 读取原始 pickle
    print(f"[INFO] Loading pickle: {pickle_path}")
    with open(pickle_path, "rb") as f:
        pk_data: Dict[str, Any] = pickle.load(f)

    if "predictions" not in pk_data or "labels" not in pk_data:
        raise ValueError("pickle 中必须包含 'predictions' 和 'labels' 字段")

    preds = pk_data["predictions"]
    labels = pk_data["labels"]

    if not isinstance(preds, dict) or not isinstance(labels, dict):
        raise TypeError("当前脚本假定 predictions / labels 为 dict[session_id] 形式，请确认结构。")

    # 4. 只保留准确 Session 的数据
    new_preds = {}
    new_labels = {}

    for k, v in preds.items():
        # k 可能是 int，也可能是 str；统一转成 str 做比较
        if str(k) in accurate_sessions:
            new_preds[k] = v

    for k, v in labels.items():
        if str(k) in accurate_sessions:
            new_labels[k] = v

    print(f"[INFO] 原始预测条目数: {len(preds)}")
    print(f"[INFO] 原始标签条目数: {len(labels)}")
    print(f"[INFO] 筛选后预测条目数: {len(new_preds)}")
    print(f"[INFO] 筛选后标签条目数: {len(new_labels)}")

    # 5. 组装新的 pickle 数据结构
    new_data = dict(pk_data)  # 复制一份整体结构
    new_data["predictions"] = new_preds
    new_data["labels"] = new_labels

    # 6. 保存新 pickle
    out_dir = os.path.dirname(out_pickle_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(out_pickle_path, "wb") as f:
        pickle.dump(new_data, f)

    print(f"[INFO] 筛选后的 pickle 已保存到: {out_pickle_path}")


def main():
    parser = argparse.ArgumentParser(description="根据 CSV 中准确 Session 筛选 rPPG pickle")
    parser.add_argument("--csv",
                        required=True,
                        help="包含 Session、Is_Accurate 的 CSV 文件路径")
    parser.add_argument("--pickle",
                        required=True,
                        help="原始 rPPG 预测结果 pickle 路径")
    parser.add_argument("--out",
                        required=True,
                        help="筛选后的 pickle 输出路径")
    args = parser.parse_args()

    filter_pickle_by_sessions(args.csv, args.pickle, args.out)


if __name__ == "__main__":
    main()

