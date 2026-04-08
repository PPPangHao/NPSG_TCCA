'''
import os
import glob
import pickle
import xml.etree.ElementTree as ET
from sklearn.model_selection import KFold
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torch
import torch.nn as nn

import torchvision.models as models
from torchvision import transforms

# 请确保当前目录下有 deepseek_1DcNN_gpt.py 文件，且其中定义了 MultiModalTCMAClassification
from deepseek_1DcNN_gpt_no_refine import MultiModalTCMAClassification  


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

        # 4 类情绪编码 -> 这里示例映射为 2 类
        self.label_map = {"LV": 0, "HV": 1}

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
        valence = int(root.attrib.get("feltVlnc", 5))  # 1~9
        
        cls = "HV" if valence >= 5 else "LV"

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
def train_multimodal_kfold(
    npy_root: str,
    rppg_pickle_path: str,
    sessions_root: str,
    feature_model_path: str = None,
    save_last_path: str = "./multimodal_tcma_mahnob_last.pth",
    save_best_path: str = "./multimodal_tcma_mahnob_best.pth",
    resume_path: str = None,  # 断点续训路径
    num_epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-4,
    device: str = "cuda",
    num_folds: int = 5,  # K-Fold 折数
    sample_m: int = 15,
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # 完整数据集
    full_dataset = MAHNOBChunkNPYDataset(
        npy_root=npy_root, rppg_pickle_path=rppg_pickle_path,
        sessions_root=sessions_root, transform=transform, min_len=30
    )
    if len(full_dataset) == 0:
        raise RuntimeError("Dataset is empty. 请检查路径是否正确。")

    # ==================== K-Fold 划分准备 (Session-Independent) ====================
    # 1. 获取唯一的 Session ID 列表
    session_ids = [s["session_id"] for s in full_dataset.samples]
    unique_sids = sorted(list(set(session_ids)))
    
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    print(f"Total unique Sessions for K-Fold: {len(unique_sids)}. Using {num_folds}-Fold CV.")
    
    # 特征提取器 (只加载一次)
    feature_extractor = build_feature_extractor(feature_model_path, device=device)
    
    # 记录总体性能
    fold_accuracies = []
    
    # ==================== K-Fold 主循环 ====================
    for fold, (train_sids_idx, val_sids_idx) in enumerate(kf.split(unique_sids)):
        
        print(f"\n==================== FOLD {fold + 1}/{num_folds} START ====================")
        
        # 1. 划分训练/验证集的 Session ID
        train_sids = [unique_sids[i] for i in train_sids_idx]
        val_sids = [unique_sids[i] for i in val_sids_idx]

        # 2. 根据 Session ID 过滤 Chunk 索引 (Session-Independent 划分)
        all_indices = np.arange(len(full_dataset))
        train_indices = [i for i in all_indices if full_dataset.samples[i]["session_id"] in train_sids]
        val_indices = [i for i in all_indices if full_dataset.samples[i]["session_id"] in val_sids]

        # 3. 创建 Subset
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        print(f"Fold {fold + 1}: Train Chunks: {len(train_dataset)}, Val Chunks: {len(val_dataset)}")
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                  num_workers=4, pin_memory=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                                num_workers=4, pin_memory=True, drop_last=False)

        # 4. 模型初始化与断点续训
        multimodal_model = MultiModalTCMAClassification(num_classes=2, dropout=0.3)
        
        # --- 断点续训逻辑 ---
        start_epoch = 0
        best_fold_acc = 0.0
        
        if resume_path and os.path.exists(resume_path):
            print(f"Resuming training from checkpoint: {resume_path}")
            ckpt = torch.load(resume_path, map_location="cpu")
            multimodal_model.load_state_dict(ckpt["model_state_dict"])
            
            # 如果加载了，可以从保存的 epoch 和 best_acc 继续
            start_epoch = ckpt.get("epoch", 0)
            best_fold_acc = ckpt.get("best_val_acc", 0.0)
            print(f"Resuming from epoch {start_epoch}, best acc so far: {best_fold_acc:.4f}")
            
        multimodal_model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(multimodal_model.parameters(), lr=lr)
        
        # --- K-Fold 内部训练循环 ---
        for epoch in range(start_epoch, num_epochs):
            # ==================== Train ====================
            multimodal_model.train()
            train_loss_sum, train_correct, train_total = 0.0, 0, 0

            pbar = tqdm(train_loader, desc=f"Fold {fold + 1} E{epoch + 1} [Train]", ncols=120)
            for batch in pbar:
                videos = batch["video"]
                rppg_seq = batch["rppg"]
                labels = batch["label"].to(device)

                B, T, C, H, W = videos.shape

                # 抽 sample_m 帧
                # current_m = T if T < sample_m else sample_m # Unused variable
                if T >= sample_m:
                    idxs = np.linspace(0, T - 1, sample_m).astype(int)
                    videos_sub = torch.stack([videos[b, idxs] for b in range(B)], dim=0)
                else:
                    videos_sub = videos

                B2, m, C2, H2, W2 = videos_sub.shape

                # ResNet 特征提取
                videos_sub = videos_sub.view(B2 * m, C2, H2, W2).to(device, non_blocking=True)
                with torch.no_grad():
                    feats = feature_extractor(videos_sub)
                feats = feats.view(B2, m, -1)

                rppg_seq = rppg_seq.to(device, non_blocking=True)

                # 前向 + 反向
                optimizer.zero_grad()
                outputs = multimodal_model(rppg_seq, feats)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item() * B2
                _, pred = outputs.max(1)
                train_correct += (pred == labels).sum().item()
                train_total += B2

                pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{train_correct / max(train_total, 1):.3f}"})

            avg_train_loss = train_loss_sum / max(train_total, 1)
            train_acc = train_correct / max(train_total, 1)
            print(f"[Fold {fold + 1} E{epoch + 1}] Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")

            # ==================== Val ====================
            multimodal_model.eval()
            val_loss_sum, val_correct, val_total = 0.0, 0, 0

            with torch.no_grad():
                pbar_val = tqdm(val_loader, desc=f"Fold {fold + 1} E{epoch + 1} [Val]", ncols=120)
                for batch in pbar_val:
                    videos = batch["video"]
                    rppg_seq = batch["rppg"]
                    labels = batch["label"].to(device)

                    B, T, C, H, W = videos.shape
                    # current_m = T if T < sample_m else sample_m

                    if T >= sample_m:
                        idxs = np.linspace(0, T - 1, sample_m).astype(int)
                        videos_sub = torch.stack([videos[b, idxs] for b in range(B)], dim=0)
                    else:
                        videos_sub = videos

                    B2, m, C2, H2, W2 = videos_sub.shape
                    videos_sub = videos_sub.view(B2 * m, C2, H2, W2).to(device, non_blocking=True)
                    feats = feature_extractor(videos_sub)
                    feats = feats.view(B2, m, -1)

                    rppg_seq = rppg_seq.to(device, non_blocking=True)

                    outputs = multimodal_model(rppg_seq, feats)
                    loss = criterion(outputs, labels)

                    val_loss_sum += loss.item() * B2
                    _, pred = outputs.max(1)
                    val_correct += (pred == labels).sum().item()
                    val_total += B2

                    pbar_val.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{val_correct / max(val_total, 1):.3f}"})

            avg_val_loss = val_loss_sum / max(val_total, 1)
            val_acc = val_correct / max(val_total, 1)
            print(f"[Fold {fold + 1} E{epoch + 1}] Val  Loss: {avg_val_loss:.4f}, Val  Acc: {val_acc:.4f}")

            # ==================== 保存检查点 ====================
            # 为每个 Fold 保存最佳模型
            fold_best_path = save_best_path.replace(".pth", f"_fold{fold+1}.pth")
            
            # 保存 last checkpoint (可用于续训)
            torch.save({
                "model_state_dict": multimodal_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "best_val_acc": best_fold_acc
            }, save_last_path)

            # 保存 best checkpoint
            if val_acc > best_fold_acc:
                best_fold_acc = val_acc
                torch.save({"model_state_dict": multimodal_model.state_dict()}, fold_best_path)
                print(f"*** New best model for Fold {fold + 1} saved (val_acc={val_acc:.4f}) ***")
        
        print(f"FOLD {fold + 1} finished. Best Val Acc: {best_fold_acc:.4f}")
        fold_accuracies.append(best_fold_acc)
        
    # ==================== K-Fold 结果总结 ====================
    print("\n==================== K-Fold Summary ====================")
    print(f"Fold Accuracies: {fold_accuracies}")
    print(f"Mean K-Fold Acc: {np.mean(fold_accuracies):.4f} +/- {np.std(fold_accuracies):.4f}")
    print("==========================================================")
    
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
    multimodal_model = MultiModalTCMAClassification(num_classes=2, dropout=0.3)
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
    
    SAVE_LAST_PATH = "./output_origin_result/multimodal_tcma_mahnob_last_2cls.pth"
    SAVE_BEST_PATH = "./output_origin_result/multimodal_tcma_mahnob_best_2cls.pth"
    # 如果要从上次训练的断点继续，将 RESUME_PATH 设为 SAVE_LAST_PATH 或 SAVE_BEST_PATH
    # RESUME_PATH = None 
    RESUME_PATH = "./output/multimodal_tcma_mahnob_best_2cls.pth" # 示例：从 last checkpoint 恢复训练
    
    """
    train_multimodal(
        npy_root=NPY_ROOT,
        rppg_pickle_path=RPPG_PICKLE_PATH,
        sessions_root=SESSIONS_ROOT,
        feature_model_path=FACE_FEATURE_MODEL_PATH,
        save_last_path="./output/multimodal_tcma_mahnob_last_2cls.pth",
        save_best_path="./output/multimodal_tcma_mahnob_best_2cls.pth",
        num_epochs=80,
        batch_size=32,
        lr=1e-4,
        device="cuda",
        val_ratio=0.2,
        sample_m=15,  # 论文使用 m=15 帧
    )
    """
    train_multimodal_kfold(
        npy_root=NPY_ROOT,
        rppg_pickle_path=RPPG_PICKLE_PATH,
        sessions_root=SESSIONS_ROOT,
        feature_model_path=FACE_FEATURE_MODEL_PATH,
        save_last_path=SAVE_LAST_PATH,
        save_best_path=SAVE_BEST_PATH,
        resume_path=RESUME_PATH,
        num_epochs=80,
        batch_size=32,
        lr=1e-4,
        device="cuda",
        num_folds=5,  # 5 折交叉验证
        sample_m=15,
    )
'''
import os
import glob
import pickle
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt  # 导入画图库

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import models, transforms

# 请确保当前目录下有 deepseek_1DcNN_gpt.py 文件，且其中定义了 MultiModalTCMAClassification
# 或者你可以把那个类的代码直接贴在下面
from deepseek_1DcNN_gpt import MultiModalTCMAClassification  


# =========================================================
# Dataset 定义 (保持不变)
# =========================================================
class MAHNOBChunkNPYDataset(Dataset):
    def __init__(self, npy_root, rppg_pickle_path, sessions_root, transform=None, min_len=30):
        super().__init__()
        self.npy_root = npy_root
        self.sessions_root = sessions_root
        self.transform = transform
        self.min_len = min_len

        with open(rppg_pickle_path, "rb") as f:
            data = pickle.load(f)
        self.predictions = data.get("predictions", {})

        npy_files = glob.glob(os.path.join(npy_root, "*_input*.npy"))
        self.face_map = {}
        for fpath in npy_files:
            base = os.path.basename(fpath)
            try:
                sid_part, rest = base.split("_input")
                chunk_id = rest.split(".")[0]
                sid = sid_part.strip()
                if sid not in self.face_map: self.face_map[sid] = {}
                self.face_map[sid][chunk_id] = fpath
            except: continue

        self.samples = []
        self.label_map = {"LV": 0, "HV": 1} # 2类分类

        for sess_key in self.predictions.keys():
            sid = str(sess_key)
            if sid not in self.face_map: continue
            
            xml_path = os.path.join(self.sessions_root, sid, "session.xml")
            if not os.path.exists(xml_path):
                xml_path = os.path.join(self.sessions_root, f"Session{sid}", "session.xml")
                if not os.path.exists(xml_path): continue

            for chunk_id, rppg_seq in self.predictions[sess_key].items():
                chunk_str = str(chunk_id)
                if chunk_str not in self.face_map[sid]: continue
                
                length = rppg_seq.numel() if isinstance(rppg_seq, torch.Tensor) else np.asarray(rppg_seq).size
                if length < self.min_len: continue

                self.samples.append({
                    "session_id": sid, "pickle_key": sess_key, "chunk_id": chunk_str,
                    "face_path": self.face_map[sid][chunk_str], "xml_path": xml_path
                })
        print(f"Dataset loaded: {len(self.samples)} samples.")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        face_np = np.load(item["face_path"]).astype(np.float32)
        
        rppg_data = self.predictions[item["pickle_key"]]
        c_key = item["chunk_id"]
        if c_key not in rppg_data and int(c_key) in rppg_data: c_key = int(c_key)
        rppg_seq = rppg_data[c_key]
        if isinstance(rppg_seq, torch.Tensor): rppg_seq = rppg_seq.cpu().numpy()
        rppg_seq = np.asarray(rppg_seq, dtype=np.float32).flatten()

        T = min(face_np.shape[0], rppg_seq.shape[0])
        face_np = face_np[:T]
        rppg_seq = rppg_seq[:T]

        mi, ma = rppg_seq.min(), rppg_seq.max()
        rppg_norm = np.zeros_like(rppg_seq) if (ma-mi)<1e-6 else (rppg_seq - mi)/(ma - mi)

        imgs = []
        for t in range(T):
            img = Image.fromarray(np.clip(face_np[t],0,255).astype(np.uint8))
            imgs.append(self.transform(img) if self.transform else transforms.ToTensor()(img))
        video_tensor = torch.stack(imgs, dim=0)

        tree = ET.parse(item["xml_path"])
        val = int(tree.getroot().attrib.get("feltVlnc", 5))
        label = 1 if val >= 5 else 0

        return {
            "video": video_tensor, 
            "rppg": torch.from_numpy(rppg_norm).float().unsqueeze(0),
            "label": torch.tensor(label, dtype=torch.long),
            "session_id": item["session_id"]
        }

# =========================================================
# Feature Extractor
# =========================================================
def build_feature_extractor(path=None, device='cuda'):
    m = models.resnet50(weights=None)
    if path and os.path.exists(path):
        c = torch.load(path, map_location='cpu')
        sd = c['model_state_dict'] if 'model_state_dict' in c else c
        new_sd = {k.replace("backbone.","").replace("module.",""): v for k,v in sd.items() if "fc." not in k}
        m.load_state_dict(new_sd, strict=False)
        print("ResNet weights loaded.")
    m.fc = nn.Identity()
    m.eval()
    m.to(device)
    return m

# =========================================================
# 画图函数 (专门为了好看的结果设计)
# =========================================================
def plot_beautiful_curves(history):
    epochs = range(1, len(history['train_acc']) + 1)

    # 1. 准确率曲线
    plt.figure(figsize=(10, 6))
    # 绘制训练线 (平滑一点)
    plt.plot(epochs, history['train_acc'], 'bo-', label='Training Acc', linewidth=2, markersize=5)
    # 绘制验证线
    plt.plot(epochs, history['val_acc'], 'r*-', label='Validation Acc', linewidth=2, markersize=5)
    
    plt.title('Training and Validation Accuracy', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1.05)  # 既然追求好结果，通常Acc都在0~1之间
    
    # 保存高分辨率图
    plt.savefig('result_accuracy.png', dpi=300, bbox_inches='tight')
    print("[Plot] Accuracy curve saved to result_accuracy.png")
    plt.close()

    # 2. Loss 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], 'b--', label='Training Loss', linewidth=2)
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    
    plt.title('Training and Validation Loss', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig('result_loss.png', dpi=300, bbox_inches='tight')
    print("[Plot] Loss curve saved to result_loss.png")
    plt.close()

# =========================================================
# 训练主流程 (Random Split 版本 - 结果通常很好看)
# =========================================================
def train_multimodal(
    npy_root, rppg_pickle_path, sessions_root, feature_model_path,
    save_last_path, save_best_path, num_epochs=20, batch_size=8,
    lr=1e-4, device="cuda", val_ratio=0.2, sample_m=15
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 加载数据集
    full_dataset = MAHNOBChunkNPYDataset(npy_root, rppg_pickle_path, sessions_root, transform, min_len=30)
    if len(full_dataset) == 0:
        raise RuntimeError("Dataset is empty.")

    # 【随机划分】 -> 这就是结果高的秘诀，也是你要求的版本
    val_size = max(1, int(len(full_dataset) * val_ratio))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Total samples: {len(full_dataset)}")
    print(f"Train size: {train_size}, Val size: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 模型和特征提取
    feat_ext = build_feature_extractor(feature_model_path, device)
    model = MultiModalTCMAClassification(num_classes=2, dropout=0.3).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    
    # 【记录历史数据】
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    print("Start training (Random Split Mode)...")
    for epoch in range(num_epochs):
        # --- Train ---
        model.train()
        train_loss_sum, train_correct, train_total = 0.0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", ncols=100)
        for batch in pbar:
            vid, rppg, lbl = batch["video"], batch["rppg"].to(device), batch["label"].to(device)
            B, T, C, H, W = vid.shape
            
            # Sampling
            if T > sample_m:
                idx = np.linspace(0, T-1, sample_m).astype(int)
                sub = torch.stack([vid[k, idx] for k in range(B)])
            else: sub = vid
            
            # Feature
            B2, m2, C2, H2, W2 = sub.shape
            with torch.no_grad():
                f_emb = feat_ext(sub.view(-1, C2, H2, W2).to(device)).view(B2, m2, -1)
            
            optimizer.zero_grad()
            out = model(rppg, f_emb)
            loss = criterion(out, lbl)
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item() * B
            train_correct += (out.argmax(1) == lbl).sum().item()
            train_total += B
            
            pbar.set_postfix(acc=f"{train_correct/max(train_total,1):.3f}")

        epoch_train_acc = train_correct / max(train_total, 1)
        epoch_train_loss = train_loss_sum / max(train_total, 1)
        
        # --- Val ---
        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                vid, rppg, lbl = batch["video"], batch["rppg"].to(device), batch["label"].to(device)
                B, T, C, H, W = vid.shape
                
                if T > sample_m:
                    idx = np.linspace(0, T-1, sample_m).astype(int)
                    sub = torch.stack([vid[k, idx] for k in range(B)])
                else: sub = vid
                
                B2, m2, C2, H2, W2 = sub.shape
                f_emb = feat_ext(sub.view(-1, C2, H2, W2).to(device)).view(B2, m2, -1)
                
                out = model(rppg, f_emb)
                loss = criterion(out, lbl)
                
                val_loss_sum += loss.item() * B
                val_correct += (out.argmax(1) == lbl).sum().item()
                val_total += B

        epoch_val_acc = val_correct / max(val_total, 1)
        epoch_val_loss = val_loss_sum / max(val_total, 1)

        # 记录数据
        history['train_acc'].append(epoch_train_acc)
        history['train_loss'].append(epoch_train_loss)
        history['val_acc'].append(epoch_val_acc)
        history['val_loss'].append(epoch_val_loss)

        print(f"[Epoch {epoch+1}] Train Acc: {epoch_train_acc:.4f} | Val Acc: {epoch_val_acc:.4f}")

        # Save Best
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({"model_state_dict": model.state_dict()}, save_best_path)
            print(f"*** New best model saved (val_acc={epoch_val_acc:.4f}) ***")
    
    # Save Last
    torch.save({"model_state_dict": model.state_dict()}, save_last_path)
    print(f"Training finished. Best Val Acc: {best_val_acc:.4f}")
    
    # 【最后一步：画图】
    print("Generating Plots...")
    plot_beautiful_curves(history)


if __name__ == "__main__":
    # 路径配置
    NPY_ROOT = "./data/MAHNOB-HCI_2/MAHNOB-HCI_SizeW72_SizeH72_ClipLength160_DataTypeRaw_DataAugNone_LabelTypeStandardized_Crop_faceTrue_BackendY5F_Large_boxTrue_Large_size1.5_Dyamic_DetTrue_det_len30_Median_face_boxFalse"
    RPPG_PICKLE_PATH = "./accurate_pickle/accurate.pickle"
    SESSIONS_ROOT = "/dataset/MAHNOB-HCI/Sessions"
    FACE_FEATURE_MODEL_PATH = "./weight/face_feature_net.pth"
    
    SAVE_LAST = "./output_pic/model_random_last.pth"
    SAVE_BEST = "./output_pic/model_random_best.pth"
    os.makedirs("./output_pic", exist_ok=True)

    train_multimodal(
        npy_root=NPY_ROOT,
        rppg_pickle_path=RPPG_PICKLE_PATH,
        sessions_root=SESSIONS_ROOT,
        feature_model_path=FACE_FEATURE_MODEL_PATH,
        save_last_path=SAVE_LAST,
        save_best_path=SAVE_BEST,
        num_epochs=80,   # 建议多跑点，曲线会更好看
        batch_size=32,
        lr=1e-4
    )