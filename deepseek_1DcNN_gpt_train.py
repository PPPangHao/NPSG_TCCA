import os
import glob
import pickle
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision.models as models
from torchvision import transforms

from deepseek_1DcNN_gpt import MultiModalTCMAClassification  # 你的多模态模型


# =========================================================
# Dataset：基于 input*.npy + accurate.pickle + session.xml
# 每个样本 = 一个 (session_id, chunk_id)
#   - face_npy:  session_inputK.npy  -> (T_face, 72, 72, 3)
#   - rppg:      predictions[session_id][K] -> (T_rppg,)
#   - label:     从 /dataset/MAHNOB-HCI/Sessions/session_id/session.xml 解析
# =========================================================
class MAHNOBChunkNPYDataset(Dataset):
    def __init__(self,
                 npy_root: str,
                 rppg_pickle_path: str,
                 sessions_root: str,
                 transform=None,
                 min_len: int = 30):
        """
        npy_root:  存放 1052_input0.npy / 1052_label0.npy ... 的目录
        rppg_pickle_path: accurate.pickle，结构为 predictions[session_id][chunk_id]
        sessions_root: /dataset/MAHNOB-HCI/Sessions
        transform: 图像预处理（Resize + Normalize 等）
        min_len: 保留的最小序列长度（face 和 rPPG 裁剪后的长度至少要这么长）
        """
        super().__init__()
        self.npy_root = npy_root
        self.sessions_root = sessions_root
        self.transform = transform
        self.min_len = min_len

        # 1) 加载 rPPG pickle
        with open(rppg_pickle_path, "rb") as f:
            data = pickle.load(f)

        if "predictions" not in data:
            raise ValueError("Pickle 文件中未找到 'predictions' 键，请确认结构。")

        self.predictions = data["predictions"]  # dict: {session_id: {chunk_id: rppg_array}}

        # 2) 建立 (session_id, chunk_id) 到 npy 的映射
        #    文件名格式示例： 1052_input0.npy / 1052_label0.npy
        npy_files = glob.glob(os.path.join(npy_root, "*_input*.npy"))
        print(f"Found {len(npy_files)} input npy files in {npy_root}")

        # session_id -> {chunk_id -> face_npy_path}
        self.face_map = {}
        for fpath in npy_files:
            base = os.path.basename(fpath)
            # 形如：1052_input0.npy
            try:
                sid_part, rest = base.split("_input")
                chunk_part = rest.split(".")[0]  # "0"
            except ValueError:
                continue
            sid = sid_part.strip()
            chunk_id = chunk_part.strip()
            if sid not in self.face_map:
                self.face_map[sid] = {}
            self.face_map[sid][chunk_id] = fpath

        # 3) 生成样本列表：只保留 (session_id, chunk_id) 三方都存在的
        #    - rPPG 存在
        #    - 对应的 inputK.npy 存在
        #    - 对应 session.xml 存在
        self.samples = []  # list of dict: {session_id, chunk_id, face_path, xml_path}
        for sess_key in self.predictions.keys():
            sid = str(sess_key)  # pickle 里可能是 int，也可能是 str
            if sid not in self.face_map:
                continue

            # Sessions/<sid>/session.xml
            xml_path = os.path.join(self.sessions_root, sid, "session.xml")
            if not os.path.exists(xml_path):
                # 有的会是 SessionXXX 命名，可按需加一条尝试
                # xml_path2 = os.path.join(self.sessions_root, f"Session{sid}", "session.xml")
                # if not os.path.exists(xml_path2): ...
                continue

            # chunk 级别匹配
            pred_chunks = self.predictions[sess_key]
            for chunk_id, rppg_seq in pred_chunks.items():
                # chunk_id 可能是 int（0,1,2...），也可能是 str("0","1","2")
                chunk_id_str = str(chunk_id)

                if chunk_id_str not in self.face_map[sid]:
                    continue

                face_path = self.face_map[sid][chunk_id_str]

                # 简单检查 rPPG 长度
                if isinstance(rppg_seq, torch.Tensor):
                    rppg_len = rppg_seq.numel()
                else:
                    rppg_len = np.asarray(rppg_seq).size

                if rppg_len < self.min_len:
                    continue

                self.samples.append({
                    "session_id": sid,
                    "pickle_key": sess_key,  # 原始 key，用来索引 predictions
                    "chunk_id": chunk_id_str,
                    "face_path": face_path,
                    "xml_path": xml_path,
                })

        print(f"Loaded {len(self.samples)} face+rPPG chunk pairs")

        # 4 类情绪编码
        self.label_map = {"LV-LA": 0, "LV-HA": 1, "HV-LA": 2, "HV-HA": 3}

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _load_face_npy(face_path: str):
        """
        face_path: 1052_input0.npy
        返回: face_clip (T,72,72,3), dtype float32
        """
        arr = np.load(face_path)  # (T,72,72,3)
        arr = arr.astype(np.float32)
        return arr  # (T,H,W,C)

    def _load_rppg_chunk(self, pickle_key, chunk_id_str):
        pred_dict = self.predictions[pickle_key]

        # chunk_id 可能是 int，也可能是 str，这里统一转换成 str 来查找
        if chunk_id_str in pred_dict:
            seq = pred_dict[chunk_id_str]
        elif chunk_id_str.isdigit() and int(chunk_id_str) in pred_dict:
            seq = pred_dict[int(chunk_id_str)]
        else:
            raise KeyError(f"Chunk id {chunk_id_str} not found in predictions[{pickle_key}]. "
                           f"Available keys: {list(pred_dict.keys())}")

        # 转 numpy
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()
        seq = np.asarray(seq, dtype=np.float32).flatten()
        return seq

    @staticmethod
    def _normalize_rppg(x: np.ndarray):
        x_min = x.min()
        x_max = x.max()
        if x_max - x_min < 1e-6:
            return np.zeros_like(x, dtype=np.float32)
        return (x - x_min) / (x_max - x_min)

    @staticmethod
    def _parse_label_from_xml(xml_path: str, label_map):
        """
        从 session.xml 中解析 feltArsl, feltVlnc -> 4 类：
          1-4: Low
          5-9: High
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        arousal = int(root.attrib.get("feltArsl", 5))  # 1~9
        valence = int(root.attrib.get("feltVlnc", 5))  # 1~9

        A = "H" if arousal >= 5 else "L"
        V = "H" if valence >= 5 else "L"

        if V == "L" and A == "L":
            cls = "LV-LA"
        elif V == "L" and A == "H":
            cls = "LV-HA"
        elif V == "H" and A == "L":
            cls = "HV-LA"
        else:
            cls = "HV-HA"

        return label_map[cls]

    def __getitem__(self, idx):
        item = self.samples[idx]
        sid = item["session_id"]
        pickle_key = item["pickle_key"]
        chunk_id = item["chunk_id"]
        face_path = item["face_path"]
        xml_path = item["xml_path"]

        # -------- 1) 加载 face npy --------
        face_np = self._load_face_npy(face_path)  # (T_face,72,72,3)
        T_face = face_np.shape[0]

        # -------- 2) 加载 rPPG chunk --------
        rppg_seq = self._load_rppg_chunk(pickle_key, chunk_id)  # (T_rppg,)
        T_rppg = rppg_seq.shape[0]

        # 对齐长度：取两者最小值
        T = min(T_face, T_rppg)
        if T < self.min_len:
            # 极端情况很短，可以直接截成 min_len 或者抛异常。
            # 这里简单截到 T（因为构建 samples 时已经筛过）
            pass

        face_np = face_np[:T]  # (T,72,72,3)
        rppg_seq = rppg_seq[:T]  # (T,)

        # rPPG 按论文公式归一化到 [0,1]
        rppg_norm = self._normalize_rppg(rppg_seq)

        # -------- 3) 图像 transform --------
        img_tensors = []
        for t in range(T):
            frame = face_np[t]  # (72,72,3)，一般为 RGB 或 BGR，这里默认当作 RGB
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            img_pil = Image.fromarray(frame)
            if self.transform is not None:
                img_tensor = self.transform(img_pil)  # (3,H,W)
            else:
                img_tensor = transforms.ToTensor()(img_pil)
            img_tensors.append(img_tensor)
        # (T,3,H,W)
        video_tensor = torch.stack(img_tensors, dim=0)

        # -------- 4) rPPG -> (1,T) tensor --------
        rppg_tensor = torch.from_numpy(rppg_norm).float().unsqueeze(0)  # (1,T)

        # -------- 5) label from xml --------
        label = self._parse_label_from_xml(xml_path, self.label_map)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            "session_id": sid,
            "chunk_id": chunk_id,
            "video": video_tensor,  # (T,3,H,W)
            "rppg": rppg_tensor,  # (1,T)
            "label": label_tensor,  # scalar
        }


# =========================================================
# ResNet50 特征提取器：输出 2048-d 特征，fc 置为 Identity
# 可以加载你自己训练好的 face_feature_net.pth（FER+ 微调）
# =========================================================
def build_feature_extractor(feature_model_path: str = None, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    print("========== Building ResNet50 Feature Extractor ==========")
    model = models.resnet50(weights=None)

    if feature_model_path and os.path.exists(feature_model_path):
        print(f"Loading checkpoint: {feature_model_path}")
        ckpt = torch.load(feature_model_path, map_location="cpu")

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt

        base_sd = model.state_dict()
        new_sd = {}
        for k, v in state_dict.items():
            k_clean = k
            if k_clean.startswith("backbone."):
                k_clean = k_clean[len("backbone."):]
            if k_clean.startswith("module."):
                k_clean = k_clean[len("module."):]

            # 跳过 fc
            if k_clean.startswith("fc."):
                print(f"[Skip] {k} (fc layer)")
                continue

            if k_clean in base_sd and base_sd[k_clean].shape == v.shape:
                new_sd[k_clean] = v
            else:
                print(f"[Skip] {k} (not used or shape mismatch)")

        print(f"Loading filtered keys: {len(new_sd)} keys")
        missing, unexpected = model.load_state_dict(new_sd, strict=False)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
    else:
        print("Using untrained ResNet50 as feature extractor.")

    model.fc = nn.Identity()
    model.to(device)
    model.eval()
    print("========== ResNet50 Feature Extractor Ready ==========")
    return model


# =========================================================
# 训练：ResNet50 + MultiModalTCMAClassification
# Face 路径：从整段 T 帧里抽 m=15 帧送 ResNet（论文设置）
# =========================================================
def train_multimodal(
        npy_root: str,
        rppg_pickle_path: str,
        sessions_root: str,
        feature_model_path: str = None,
        save_last_path: str = "./multimodal_tcma_mahnob_last.pth",
        save_best_path: str = "./multimodal_tcma_mahnob_best.pth",
        num_epochs: int = 20,
        batch_size: int = 8,
        lr: float = 1e-4,
        device: str = "cuda",
        val_ratio: float = 0.2,
        sample_m: int = 15,  # 论文中从 n 帧抽 m=15 帧进 ResNet
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # 图像预处理：升采样到 224x224 + 标准 ImageNet 归一化
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # Dataset
    full_dataset = MAHNOBChunkNPYDataset(
        npy_root=npy_root,
        rppg_pickle_path=rppg_pickle_path,
        sessions_root=sessions_root,
        transform=transform,
        min_len=30
    )
    if len(full_dataset) == 0:
        raise RuntimeError("Dataset is empty. 请检查 npy_root / rppg_pickle_path / sessions_root 是否正确。")

    # Train / Val 划分（chunk 级别）
    val_size = max(1, int(len(full_dataset) * val_ratio))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"Train samples: {train_size}, Val samples: {val_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    # 特征提取器和多模态模型
    feature_extractor = build_feature_extractor(feature_model_path, device=device)
    multimodal_model = MultiModalTCMAClassification(num_classes=4, dropout=0.3)
    multimodal_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(multimodal_model.parameters(), lr=lr)

    best_val_acc = 0.0

    print("Start training...")
    for epoch in range(num_epochs):
        # ==================== Train ====================
        multimodal_model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", ncols=120)
        for batch in pbar:
            videos = batch["video"]  # (B,T,3,H,W)
            rppg_seq = batch["rppg"]  # (B,1,T_rppg)
            labels = batch["label"].to(device)

            B, T, C, H, W = videos.shape

            # ---- 从 T 帧里抽 sample_m 帧进入 ResNet（论文：m=15）----
            if T >= sample_m:
                # 用 numpy 生成统一 index
                idxs = np.linspace(0, T - 1, sample_m).astype(int)
                # 按 batch 手动抽
                videos_sub = []
                for b in range(B):
                    videos_sub.append(videos[b, idxs])  # (m,3,H,W)
                videos_sub = torch.stack(videos_sub, dim=0)  # (B,m,3,H,W)
            else:
                # 帧数不够时，直接用全部帧
                videos_sub = videos
                sample_m = T

            B2, m, C2, H2, W2 = videos_sub.shape

            # ResNet 特征提取：在主进程 + GPU
            videos_sub = videos_sub.view(B2 * m, C2, H2, W2).to(device, non_blocking=True)
            with torch.no_grad():
                feats = feature_extractor(videos_sub)  # (B*m, 2048)
            feats = feats.view(B2, m, -1)  # (B,m,2048)

            # rPPG
            rppg_seq = rppg_seq.to(device, non_blocking=True)  # (B,1,T_rppg)

            # 前向 + 反向
            optimizer.zero_grad()
            outputs = multimodal_model(rppg_seq, feats)  # (B,4)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * B2
            _, pred = outputs.max(1)
            train_correct += (pred == labels).sum().item()
            train_total += B2

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{train_correct / max(train_total, 1):.3f}"
            })

        avg_train_loss = train_loss_sum / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)
        print(f"[Epoch {epoch + 1}] Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # ==================== Val ====================
        multimodal_model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]", ncols=120)
            for batch in pbar_val:
                videos = batch["video"]  # (B,T,3,H,W)
                rppg_seq = batch["rppg"]  # (B,1,T_rppg)
                labels = batch["label"].to(device)

                B, T, C, H, W = videos.shape

                if T >= sample_m:
                    idxs = np.linspace(0, T - 1, sample_m).astype(int)
                    videos_sub = []
                    for b in range(B):
                        videos_sub.append(videos[b, idxs])
                    videos_sub = torch.stack(videos_sub, dim=0)
                else:
                    videos_sub = videos
                    sample_m = T

                B2, m, C2, H2, W2 = videos_sub.shape
                videos_sub = videos_sub.view(B2 * m, C2, H2, W2).to(device, non_blocking=True)
                feats = feature_extractor(videos_sub)  # (B*m,2048)
                feats = feats.view(B2, m, -1)

                rppg_seq = rppg_seq.to(device, non_blocking=True)

                outputs = multimodal_model(rppg_seq, feats)
                loss = criterion(outputs, labels)

                val_loss_sum += loss.item() * B2
                _, pred = outputs.max(1)
                val_correct += (pred == labels).sum().item()
                val_total += B2

                pbar_val.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{val_correct / max(val_total, 1):.3f}"
                })

        avg_val_loss = val_loss_sum / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        print(f"[Epoch {epoch + 1}] Val  Loss: {avg_val_loss:.4f}, Val  Acc: {val_acc:.4f}")

        # 保存 last
        torch.save({"model_state_dict": multimodal_model.state_dict()},
                   save_last_path)

        # 保存 best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state_dict": multimodal_model.state_dict()},
                       save_best_path)
            print(f"*** New best model saved (val_acc={val_acc:.4f}) ***")

    print("Training finished.")
    print(f"Last model saved to: {save_last_path}")
    print(f"Best model saved to: {save_best_path}, best val acc = {best_val_acc:.4f}")


# =========================================================
# main
# =========================================================
if __name__ == "__main__":
    # 这些路径按你的实际情况改
    NPY_ROOT = "./data/MAHNOB-HCI_2/MAHNOB-HCI_SizeW72_SizeH72_ClipLength160_DataTypeRaw_DataAugNone_LabelTypeStandardized_Crop_faceTrue_BackendY5F_Large_boxTrue_Large_size1.5_Dyamic_DetTrue_det_len30_Median_face_boxFalse"
    RPPG_PICKLE_PATH = "./accurate_pickle/accurate.pickle"
    SESSIONS_ROOT = "/dataset/MAHNOB-HCI/Sessions"
    FACE_FEATURE_MODEL_PATH = "./weight/face_feature_net.pth"  # 或 None

    train_multimodal(
        npy_root=NPY_ROOT,
        rppg_pickle_path=RPPG_PICKLE_PATH,
        sessions_root=SESSIONS_ROOT,
        feature_model_path=FACE_FEATURE_MODEL_PATH,
        save_last_path="./multimodal_tcma_mahnob_last.pth",
        save_best_path="./multimodal_tcma_mahnob_best.pth",
        num_epochs=80,
        batch_size=8,
        lr=1e-4,
        device="cuda",
        val_ratio=0.2,
        sample_m=15,  # 论文使用 m=15 帧
    )
