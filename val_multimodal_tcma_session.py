import os
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import transforms
import torchvision.models as models

from deepseek_1DcNN_gpt import MultiModalTCMAClassification
from deepseek_1DcNN_gpt_train_2class import MAHNOBChunkNPYDataset, build_feature_extractor
# ↑↑↑ 确保 import 路径正确（和 train 用同一份 Dataset）


# =========================================================
# Session-level Validation
# =========================================================
@torch.no_grad()
def validate_session_level(
    npy_root,
    rppg_pickle_path,
    sessions_root,
    model_ckpt,
    face_feature_model_path=None,
    batch_size=32,
    device="cuda",
    sample_m=15,
    out_pickle="./session_predictions.pkl"
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # ---------- transform ----------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # ---------- Dataset ----------
    dataset = MAHNOBChunkNPYDataset(
        npy_root=npy_root,
        rppg_pickle_path=rppg_pickle_path,
        sessions_root=sessions_root,
        transform=transform,
        min_len=30
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # ---------- Models ----------
    feature_extractor = build_feature_extractor(
        face_feature_model_path, device=device
    )

    model = MultiModalTCMAClassification(num_classes=2, dropout=0.0)
    ckpt = torch.load(model_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # ---------- Buffers ----------
    session_probs = defaultdict(list)
    session_preds = defaultdict(list)
    session_gt = {}

    # ---------- Inference ----------
    for batch in tqdm(loader, desc="Running inference"):
        videos = batch["video"]          # (B,T,3,H,W)
        rppg = batch["rppg"]             # (B,1,T)
        labels = batch["label"]          # (B,)
        session_ids = batch["session_id"]

        B, T, C, H, W = videos.shape

        # ---- sample m frames ----
        if T >= sample_m:
            idxs = np.linspace(0, T - 1, sample_m).astype(int)
            videos_sub = torch.stack(
                [videos[b, idxs] for b in range(B)], dim=0
            )
        else:
            videos_sub = videos
            sample_m = T

        B2, m, C2, H2, W2 = videos_sub.shape
        videos_sub = videos_sub.view(B2 * m, C2, H2, W2).to(device)
        feats = feature_extractor(videos_sub)
        feats = feats.view(B2, m, -1)

        rppg = rppg.to(device)
        logits = model(rppg, feats)
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        # ---- collect per session ----
        for i in range(B):
            sid = session_ids[i]
            session_probs[sid].append(probs[i].cpu().numpy())
            session_preds[sid].append(preds[i].item())
            session_gt[sid] = labels[i].item()

    # ---------- Session aggregation ----------
    results = {}
    correct = 0

    for sid in session_probs.keys():
        probs = np.stack(session_probs[sid], axis=0)   # (N_chunk,2)
        mean_prob = probs.mean(axis=0)
        session_pred = int(mean_prob.argmax())
        gt = session_gt[sid]

        if session_pred == gt:
            correct += 1

        results[sid] = {
            "gt": gt,
            "pred": session_pred,
            "chunk_preds": session_preds[sid],
            "chunk_probs": probs.tolist(),
            "mean_prob": mean_prob.tolist(),
        }

    acc = correct / len(results)

    # ---------- save ----------
    with open(out_pickle, "wb") as f:
        pickle.dump(results, f)

    print("\n============== Session-level Validation ==============")
    print(f"Total sessions : {len(results)}")
    print(f"Valence ACC    : {acc:.4f}")
    print(f"Saved to       : {out_pickle}")
    print("======================================================")

    return acc, results


# =========================================================
# main
# =========================================================
if __name__ == "__main__":
    NPY_ROOT = "./data/MAHNOB-HCI_2/MAHNOB-HCI_SizeW72_SizeH72_ClipLength160_DataTypeRaw_DataAugNone_LabelTypeStandardized_Crop_faceTrue_BackendY5F_Large_boxTrue_Large_size1.5_Dyamic_DetTrue_det_len30_Median_face_boxFalse"
    RPPG_PICKLE_PATH = "./accurate_pickle/accurate.pickle"
    SESSIONS_ROOT = "/dataset/MAHNOB-HCI/Sessions"

    MODEL_CKPT = "./output/multimodal_tcma_mahnob_best_2cls.pth"
    FACE_FEATURE_MODEL_PATH = "./weight/face_feature_net.pth"

    validate_session_level(
        npy_root=NPY_ROOT,
        rppg_pickle_path=RPPG_PICKLE_PATH,
        sessions_root=SESSIONS_ROOT,
        model_ckpt=MODEL_CKPT,
        face_feature_model_path=FACE_FEATURE_MODEL_PATH,
        batch_size=32,
        device="cuda",
        sample_m=15,
        out_pickle="./output/session_valence_predictions.pkl"
    )

