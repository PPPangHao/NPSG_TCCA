import os
import numpy as np

def check_input_npy(root_dir):
    print(f"Checking directory: {root_dir}")

    if not os.path.exists(root_dir):
        print("Directory does not exist!")
        return

    npy_files = [f for f in os.listdir(root_dir) if f.lower().endswith(".npy") and "input" in f.lower()]

    if not npy_files:
        print("No input*.npy files found.")
        return

    print(f"Found {len(npy_files)} candidate input npy files:\n")
    for name in npy_files:
        npy_path = os.path.join(root_dir, name)
        try:
            arr = np.load(npy_path)
        except Exception as e:
            print(f"Failed to load {name}: {e}")
            continue

        print("-" * 60)
        print(f"File: {name}")
        print(f"Shape: {arr.shape}")
        print(f"Dtype: {arr.dtype}")
        print(f"Min: {arr.min()}, Max: {arr.max()}")

        # 判断是否 72×72
        if len(arr.shape) == 4:
            T, H, W, C = arr.shape
            print(f"Detected sequence: T={T}, H={H}, W={W}, C={C}")
            if H == 72 and W == 72:
                print(">>> This looks like MTCNN 72×72 face inputs (good).")
        elif len(arr.shape) == 3:
            H, W, C = arr.shape
            print(f"Detected image: H={H}, W={W}, C={C}")
            if H == 72 and W == 72:
                print(">>> This is likely a cropped 72×72 face image.")
        else:
            print("Unknown shape format.")

    print("-" * 60)
    print("Scan finished.")


if __name__ == "__main__":
    # 修改为你的 rPPG 提取输入路径，例如：
    # "/dataset/MAHNOB-HCI/rppg_extract/session_104/input"
    ROOT = "/root/autodl-tmp/Code/rPPG-Toolbox/data/MAHNOB-HCI_2/MAHNOB-HCI_SizeW72_SizeH72_ClipLength160_DataTypeRaw_DataAugNone_LabelTypeStandardized_Crop_faceTrue_BackendY5F_Large_boxTrue_Large_size1.5_Dyamic_DetTrue_det_len30_Median_face_boxFalse"

    check_input_npy(ROOT)

