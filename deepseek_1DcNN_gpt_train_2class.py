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

# 确保这里能正确导入您的模型定义
from deepseek_1DcNN_gpt import MultiModalTCMAClassification  

# =========================================================
# Dataset：基于 input*.npy + accurate.pickle + session.xml
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
        min_len: 保留的最小序列长度
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
        npy_files = glob.glob(os.path.join(npy_root, "*_input*.npy"))
        print(f"Found {len(npy_files)} input npy files in {npy_root}")

        self.face_map = {}
        for fpath in npy_files:
            base = os.path.basename(fpath)
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

        # 3) 生成样本列表
        self.samples = [] 
        for sess_key in self.predictions.keys():
            sid = str(sess_key)
            if sid not in self.face_map:
                continue

            xml_path = os.path.join(self.sessions_root, sid, "session.xml")
            if not os.path.exists(xml_path):
                continue

            pred_chunks = self.predictions[sess_key]
            for chunk_id, rppg_seq in pred_chunks.items():
                chunk_id_str = str(chunk_id)

                if chunk_id_str not in self.face_map[sid]:
                    continue

                face_path = self.face_map[sid][chunk_id_str]

                if isinstance(rppg_seq, torch.Tensor):
                    rppg_len = rppg_seq.numel()
                else:
                    rppg_len = np.asarray(rppg_seq).size

                if rppg_len < self.min_len:
                    continue

                self.samples.append({
                    "session_id": sid,
                    "pickle_key": sess_key,
                    "chunk_id": chunk_id_str,
                    "face_path": face_path,
                    "xml_path": xml_path,
                })

        print(f"Loaded {len(self.samples)} face+rPPG chunk pairs")
        self.label_map = {"LV": 0, "HV": 1}

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _load_face_npy(face_path: str):
        arr = np.load(face_path)  # (T,72,72,3)
        arr = arr.astype(np.float32)
        return arr

    def _load_rppg_chunk(self, pickle_key, chunk_id_str):
        pred_dict = self.predictions[pickle_key]
        if chunk_id_str in pred_dict:
            seq = pred_dict[chunk_id_str]
        elif chunk_id_str.isdigit() and int(chunk_id_str) in pred_dict:
            seq = pred_dict[int(chunk_id_str)]
        else:
            raise KeyError(f"Chunk id {chunk_id_str} not found.")

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
        tree = ET.parse(xml_path)
        root = tree.getroot()
        valence = int(root.attrib.get("feltVlnc", 5))
        cls = "HV" if valence >= 5 else "LV"
        return label_map[cls]

    def __getitem__(self, idx):
        item = self.samples[idx]
        face_path = item["face_path"]
        
        # 1) Face
        face_np = self._load_face_npy(face_path)
        T_face = face_np.shape[0]

        # 2) rPPG
        rppg_seq = self._load_rppg_chunk(item["pickle_key"], item["chunk_id"])
        T_rppg = rppg_seq.shape[0]

        # Align
        T = min(T_face, T_rppg)
        face_np = face_np[:T]
        rppg_seq = rppg_seq[:T]
        rppg_norm = self._normalize_rppg(rppg_seq)

        # 3) Transform
        img_tensors = []
        for t in range(T):
            frame = face_np[t]
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            img_pil = Image.fromarray(frame)
            if self.transform is not None:
                img_tensor = self.transform(img_pil)
            else:
                img_tensor = transforms.ToTensor()(img_pil)
            img_tensors.append(img_tensor)
        
        video_tensor = torch.stack(img_tensors, dim=0)
        rppg_tensor = torch.from_numpy(rppg_norm).float().unsqueeze(0)
        
        label = self._parse_label_from_xml(item["xml_path"], self.label_map)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            "session_id": item["session_id"],
            "chunk_id": item["chunk_id"],
            "video": video_tensor,
            "rppg": rppg_tensor,
            "label": label_tensor,
        }


# =========================================================
# ResNet50 特征提取器
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
            k_clean = k.replace("backbone.", "").replace("module.", "")
            if k_clean.startswith("fc."): continue
            if k_clean in base_sd and base_sd[k_clean].shape == v.shape:
                new_sd[k_clean] = v
        
        model.load_state_dict(new_sd, strict=False)
    else:
        print("Using untrained ResNet50 as feature extractor.")

    model.fc = nn.Identity()
    model.to(device)
    model.eval()
    return model


# =========================================================
# 训练：K-Fold Cross Validation (包含自动绘图数据保存)
# =========================================================
def train_multimodal_kfold(
    npy_root: str,
    rppg_pickle_path: str,
    sessions_root: str,
    feature_model_path: str = None,
    save_last_path: str = "./multimodal_tcma_mahnob_last.pth",
    save_best_path: str = "./multimodal_tcma_mahnob_best.pth",
    resume_path: str = None,
    num_epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-4,
    device: str = "cuda",
    num_folds: int = 5,
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

    # ==================== K-Fold 划分准备 ====================
    session_ids = [s["session_id"] for s in full_dataset.samples]
    unique_sids = sorted(list(set(session_ids)))
    
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    print(f"Total unique Sessions for K-Fold: {len(unique_sids)}. Using {num_folds}-Fold CV.")
    
    # 特征提取器
    feature_extractor = build_feature_extractor(feature_model_path, device=device)
    
    fold_accuracies = []
    # [KEY] 保存所有Fold的训练历史，用于画图
    all_folds_history = {}
    
    # ==================== K-Fold 主循环 ====================
    for fold, (train_sids_idx, val_sids_idx) in enumerate(kf.split(unique_sids)):
        
        print(f"\n==================== FOLD {fold + 1}/{num_folds} START ====================")
        
        # 初始化当前折记录
        fold_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        train_sids = [unique_sids[i] for i in train_sids_idx]
        val_sids = [unique_sids[i] for i in val_sids_idx]

        all_indices = np.arange(len(full_dataset))
        train_indices = [i for i in all_indices if full_dataset.samples[i]["session_id"] in train_sids]
        val_indices = [i for i in all_indices if full_dataset.samples[i]["session_id"] in val_sids]

        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        print(f"Fold {fold + 1}: Train Chunks: {len(train_dataset)}, Val Chunks: {len(val_dataset)}")
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                  num_workers=4, pin_memory=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                                num_workers=4, pin_memory=True, drop_last=False)

        # 模型初始化
        multimodal_model = MultiModalTCMAClassification(num_classes=2, dropout=0.3)
        
        # K-Fold 中的 resume 通常只用于加载预训练权重，不恢复 optimizer 状态
        if resume_path and os.path.exists(resume_path):
            print(f"Loading weights from: {resume_path}")
            ckpt = torch.load(resume_path, map_location="cpu")
            if "model_state_dict" in ckpt:
                multimodal_model.load_state_dict(ckpt["model_state_dict"])
            else:
                multimodal_model.load_state_dict(ckpt)
            
        multimodal_model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(multimodal_model.parameters(), lr=lr)
        
        best_fold_acc = 0.0

        # --- Epoch 循环 ---
        for epoch in range(num_epochs):
            # ==================== Train ====================
            multimodal_model.train()
            train_loss_sum, train_correct, train_total = 0.0, 0, 0

            pbar = tqdm(train_loader, desc=f"Fold {fold + 1} E{epoch + 1} [Train]", ncols=120)
            for batch in pbar:
                videos = batch["video"]
                rppg_seq = batch["rppg"]
                labels = batch["label"].to(device)

                B, T, C, H, W = videos.shape

                # 抽帧
                if T >= sample_m:
                    idxs = np.linspace(0, T - 1, sample_m).astype(int)
                    videos_sub = torch.stack([videos[b, idxs] for b in range(B)], dim=0)
                else:
                    videos_sub = videos
                    # 临时修正 m
                    # sample_m_curr = T 

                B2, m_curr, C2, H2, W2 = videos_sub.shape

                videos_sub = videos_sub.view(B2 * m_curr, C2, H2, W2).to(device, non_blocking=True)
                with torch.no_grad():
                    feats = feature_extractor(videos_sub)
                feats = feats.view(B2, m_curr, -1)

                rppg_seq = rppg_seq.to(device, non_blocking=True)

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
            
            # [Record]
            fold_history['train_loss'].append(float(avg_train_loss))
            fold_history['train_acc'].append(float(train_acc))

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
                    
                    if T >= sample_m:
                        idxs = np.linspace(0, T - 1, sample_m).astype(int)
                        videos_sub = torch.stack([videos[b, idxs] for b in range(B)], dim=0)
                    else:
                        videos_sub = videos

                    B2, m_curr, C2, H2, W2 = videos_sub.shape
                    videos_sub = videos_sub.view(B2 * m_curr, C2, H2, W2).to(device, non_blocking=True)
                    feats = feature_extractor(videos_sub)
                    feats = feats.view(B2, m_curr, -1)
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
            
            # [Record]
            fold_history['val_loss'].append(float(avg_val_loss))
            fold_history['val_acc'].append(float(val_acc))

            # 保存模型
            fold_best_path = save_best_path.replace(".pth", f"_fold{fold+1}.pth")
            
            # Save Last
            torch.save({
                "model_state_dict": multimodal_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "best_val_acc": best_fold_acc
            }, save_last_path)

            # Save Best
            if val_acc > best_fold_acc:
                best_fold_acc = val_acc
                torch.save({"model_state_dict": multimodal_model.state_dict()}, fold_best_path)
                print(f"*** New best model for Fold {fold + 1} saved (val_acc={val_acc:.4f}) ***")
        
        print(f"FOLD {fold + 1} finished. Best Val Acc: {best_fold_acc:.4f}")
        fold_accuracies.append(best_fold_acc)
        all_folds_history[f'fold_{fold+1}'] = fold_history
        
    # ==================== Summary & Save Data ====================
    print("\n==================== K-Fold Summary ====================")
    print(f"Fold Accuracies: {fold_accuracies}")
    print(f"Mean K-Fold Acc: {np.mean(fold_accuracies):.4f} +/- {np.std(fold_accuracies):.4f}")
    
    # [KEY] 保存数据供绘图脚本使用
    np.save("training_history_kfold.npy", all_folds_history)
    print("Training history saved to 'training_history_kfold.npy'")
    print("==========================================================")


# =========================================================
# 普通训练 (Train/Val Split) - 保留以备不时之需
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
        sample_m: int = 15,
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = MAHNOBChunkNPYDataset(
        npy_root=npy_root, rppg_pickle_path=rppg_pickle_path,
        sessions_root=sessions_root, transform=transform, min_len=30
    )
    if len(full_dataset) == 0:
        raise RuntimeError("Dataset is empty.")

    val_size = max(1, int(len(full_dataset) * val_ratio))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"Train samples: {train_size}, Val samples: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    feature_extractor = build_feature_extractor(feature_model_path, device=device)
    multimodal_model = MultiModalTCMAClassification(num_classes=2, dropout=0.3)
    multimodal_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(multimodal_model.parameters(), lr=lr)
    best_val_acc = 0.0

    print("Start training (Simple Split)...")
    for epoch in range(num_epochs):
        multimodal_model.train()
        train_loss_sum, train_correct, train_total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", ncols=120)
        
        for batch in pbar:
            videos = batch["video"]
            rppg_seq = batch["rppg"]
            labels = batch["label"].to(device)
            B, T, C, H, W = videos.shape

            if T >= sample_m:
                idxs = np.linspace(0, T - 1, sample_m).astype(int)
                videos_sub = torch.stack([videos[b, idxs] for b in range(B)], dim=0)
            else:
                videos_sub = videos
                
            B2, m_curr, C2, H2, W2 = videos_sub.shape
            videos_sub = videos_sub.view(B2 * m_curr, C2, H2, W2).to(device, non_blocking=True)
            with torch.no_grad():
                feats = feature_extractor(videos_sub)
            feats = feats.view(B2, m_curr, -1)
            rppg_seq = rppg_seq.to(device, non_blocking=True)

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
        print(f"[Epoch {epoch + 1}] Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Val
        multimodal_model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="[Val]", ncols=120):
                videos = batch["video"]
                rppg_seq = batch["rppg"]
                labels = batch["label"].to(device)
                
                B, T, C, H, W = videos.shape
                if T >= sample_m:
                    idxs = np.linspace(0, T - 1, sample_m).astype(int)
                    videos_sub = torch.stack([videos[b, idxs] for b in range(B)], dim=0)
                else:
                    videos_sub = videos
                
                B2, m_curr, C2, H2, W2 = videos_sub.shape
                videos_sub = videos_sub.view(B2 * m_curr, C2, H2, W2).to(device, non_blocking=True)
                feats = feature_extractor(videos_sub)
                feats = feats.view(B2, m_curr, -1)
                rppg_seq = rppg_seq.to(device, non_blocking=True)

                outputs = multimodal_model(rppg_seq, feats)
                loss = criterion(outputs, labels)
                val_loss_sum += loss.item() * B2
                _, pred = outputs.max(1)
                val_correct += (pred == labels).sum().item()
                val_total += B2

        avg_val_loss = val_loss_sum / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        print(f"[Epoch {epoch + 1}] Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

        torch.save({"model_state_dict": multimodal_model.state_dict()}, save_last_path)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state_dict": multimodal_model.state_dict()}, save_best_path)
            print(f"*** New best: {val_acc:.4f} ***")

# =========================================================
# main
# =========================================================
if __name__ == "__main__":
    # 路径配置
    NPY_ROOT = "./data/MAHNOB-HCI_2/MAHNOB-HCI_SizeW72_SizeH72_ClipLength160_DataTypeRaw_DataAugNone_LabelTypeStandardized_Crop_faceTrue_BackendY5F_Large_boxTrue_Large_size1.5_Dyamic_DetTrue_det_len30_Median_face_boxFalse"
    RPPG_PICKLE_PATH = "./accurate_pickle/accurate.pickle"
    SESSIONS_ROOT = "/dataset/MAHNOB-HCI/Sessions"
    FACE_FEATURE_MODEL_PATH = "./weight/face_feature_net.pth"
    
    # 确保输出目录存在
    os.makedirs("./output_474", exist_ok=True)
    SAVE_LAST_PATH = "./output_474/multimodal_tcma_mahnob_last_2cls.pth"
    SAVE_BEST_PATH = "./output_474/multimodal_tcma_mahnob_best_2cls.pth"
    
    RESUME_PATH = None 
    
    # 运行 K-Fold 训练
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
        num_folds=5,
        sample_m=15,
    )