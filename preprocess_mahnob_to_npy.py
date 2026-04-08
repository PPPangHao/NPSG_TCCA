import os
import glob
import pickle
import numpy as np
import xml.etree.ElementTree as ET
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# ----------------------------------------------------------
#  ResNet18 Feature Extractor (输出 512 维特征)
# ----------------------------------------------------------
def build_resnet18(device="cuda"):
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Identity()        # 512-dim
    model = model.to(device)
    model.eval()
    return model


# ----------------------------------------------------------
#   提取 (T, 3, 224,224) 和 (T,512) 特征
# ----------------------------------------------------------
IMAGE_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])


def extract_video_features(video_path, target_len, feature_net, device="cuda"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return None, None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    idxs = np.linspace(0, max(0, total-1), target_len).astype(int)

    imgs = []
    feats = []

    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        tensor = IMAGE_TF(pil).unsqueeze(0).to(device)   # (1,3,224,224)
        with torch.no_grad():
            feat = feature_net(tensor).cpu().numpy()[0]  # (512,)

        imgs.append(tensor.cpu().numpy()[0])             # (3,224,224)
        feats.append(feat)

    cap.release()
    return np.array(imgs), np.array(feats)


# ----------------------------------------------------------
#   rPPG 合并 + 重采样到 target_len
# ----------------------------------------------------------
def merge_and_resample_rppg(pred_dict, target_len):
    merged = []
    for k in sorted(pred_dict.keys(), key=lambda x: int(x)):
        v = pred_dict[k]
        if torch.is_tensor(v):
            v = v.cpu().numpy()
        merged.append(np.array(v).flatten())

    if not merged:
        return np.zeros(target_len)

    sig = np.concatenate(merged)

    x_old = np.linspace(0, 1, len(sig))
    x_new = np.linspace(0, 1, target_len)
    sig_rs = np.interp(x_new, x_old, sig)

    # 标准化
    if np.std(sig_rs) > 1e-6:
        sig_rs = (sig_rs - np.mean(sig_rs)) / np.std(sig_rs)

    return sig_rs.astype(np.float32)


# ----------------------------------------------------------
#   XML 标签 → Valence 二分类 (H=1 / L=0)
# ----------------------------------------------------------
def read_valence_label(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    valence = int(root.attrib.get("feltVlnc", 5))
    label = 1 if valence >= 5 else 0
    return label


# ----------------------------------------------------------
#   主函数：生成 NPY
# ----------------------------------------------------------
def preprocess_all(
    sessions_root,
    rppg_pkl,
    out_dir="./preprocessed",
    target_len=300,
    device="cuda"
):
    os.makedirs(out_dir, exist_ok=True)

    # load rppg pickle
    with open(rppg_pkl, "rb") as f:
        data = pickle.load(f)

    predictions = data["predictions"]     # {sess_id: {chunk: sig}}

    # load feature net
    feature_net = build_resnet18(device=device)

    for sess_id in tqdm(predictions.keys(), desc="Processing Sessions"):
        sid = str(sess_id)
        sess_dir = os.path.join(sessions_root, sid)

        # ---- 找 AVI ----
        avi_list = glob.glob(os.path.join(sess_dir, "*.avi"))
        if not avi_list:
            continue
        avi_path = avi_list[0]

        # ---- XML 标签 ----
        xml_path = os.path.join(sess_dir, "session.xml")
        if not os.path.exists(xml_path):
            continue

        # ---- 视频特征 ----
        imgs, feats = extract_video_features(avi_path, target_len, feature_net, device)
        if imgs is None:
            continue

        # ---- rPPG ----
        rppg = merge_and_resample_rppg(predictions[sess_id], target_len)

        # ---- 标签 ----
        label = read_valence_label(xml_path)

        # ---- 存 ----
        np.save(os.path.join(out_dir, f"{sid}_video.npy"), imgs)
        np.save(os.path.join(out_dir, f"{sid}_facefeat.npy"), feats)
        np.save(os.path.join(out_dir, f"{sid}_rppg.npy"), rppg)
        np.save(os.path.join(out_dir, f"{sid}_label.npy"), np.array(label))

    print("✅ Finished preprocessing MAHNOB → NPY!")


if __name__ == "__main__":
    preprocess_all(
        sessions_root="/dataset/MAHNOB-HCI/Sessions",
        rppg_pkl="./accurate_pickle/accurate.pickle",
        out_dir="./preprocessed",
        target_len=300,
        device="cuda"
    )

