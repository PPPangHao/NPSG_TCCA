# 增加混淆矩阵
import os
import glob
import pickle
import xml.etree.ElementTree as ET
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torch
import torch.nn as nn

import torchvision.models as models
from torchvision import transforms

# 绘图库
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# 请确保当前目录下有 deepseek_1DcNN_gpt.py 文件
from deepseek_1DcNN_gpt import MultiModalTCMAClassification

# =========================================================
#  辅助功能：绘制并保存混淆矩阵
# =========================================================
def plot_and_save_confusion_matrix(y_true, y_pred, classes, save_path, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

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

        self.predictions = data["predictions"]

        # 2) 建立 (session_id, chunk_id) 到 npy 的映射
        npy_files = glob.glob(os.path.join(npy_root, "*_input*.npy"))
        print(f"Found {len(npy_files)} input npy files in {npy_root}")

        self.face_map = {}
        for fpath in npy_files:
            base = os.path.basename(fpath)
            try:
                sid_part, rest = base.split("_input")
                chunk_part = rest.split(".")[0]
            except ValueError:
                continue
            sid = sid_part.strip()
            chunk_id = chunk_part.strip()
            if sid not in self.face_map:
                self.face_map[sid] = {}
            self.face_map[sid][chunk_id] = fpath

        # 3) 生成样本列表 (在此处进行标签过滤)
        self.samples = []
        skipped_count = 0 # 统计被过滤掉的样本数

        for sess_key in self.predictions.keys():
            sid = str(sess_key)
            if sid not in self.face_map:
                continue

            xml_path = os.path.join(self.sessions_root, sid, "session.xml")
            if not os.path.exists(xml_path):
                continue
            
            # --- 修改核心：提前读取分数进行过滤 ---
            raw_score = self._get_score_from_xml(xml_path)
            
            # 过滤逻辑：小于3的不加入 dataset
            # if raw_score < 3:
            #     skipped_count += 1
            #     continue 
            
            # 制作标签：3-9之间，以5为界做二分类
            # >= 5 -> High (1)
            # < 5 (且>=3) -> Low (0)
            label_int = 1 if raw_score >= 5 else 0
            # ------------------------------------

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
                    "label": label_int  # 直接存储处理好的标签
                })

        print(f"Loaded {len(self.samples)} face+rPPG chunk pairs.")
        print(f"Filtered out {skipped_count} sessions (score < 3).")
        
        self.label_map = {0: "Low (3-4)", 1: "High (5-9)"}

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _load_face_npy(face_path: str):
        arr = np.load(face_path)
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
    def _get_score_from_xml(xml_path: str):
        """
        读取 XML 中的 feltArsl (Arousal) 数值并返回整数。
        如果找不到，默认返回 5 (避免报错，可视情况修改)。
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            # 注意：这里假设标签是 'feltArsl'，如果是 valence 请改为 'feltVlnc'
            score = int(root.attrib.get("feltArsl", 5)) 
            return score
        except Exception as e:
            print(f"Error parsing XML {xml_path}: {e}")
            return 5

    def __getitem__(self, idx):
        item = self.samples[idx]
        sid = item["session_id"]
        pickle_key = item["pickle_key"]
        chunk_id = item["chunk_id"]
        face_path = item["face_path"]
        
        # 直接使用 __init__ 中计算好的 label
        label_int = item["label"]

        face_np = self._load_face_npy(face_path)
        T_face = face_np.shape[0]

        rppg_seq = self._load_rppg_chunk(pickle_key, chunk_id)
        T_rppg = rppg_seq.shape[0]

        T = min(T_face, T_rppg)
        face_np = face_np[:T]
        rppg_seq = rppg_seq[:T]

        rppg_norm = self._normalize_rppg(rppg_seq)

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
        
        if len(img_tensors) > 0:
            video_tensor = torch.stack(img_tensors, dim=0)
        else:
            # 异常处理：如果是空视频
            video_tensor = torch.zeros((1, 3, 224, 224))
            
        rppg_tensor = torch.from_numpy(rppg_norm).float().unsqueeze(0)
        label_tensor = torch.tensor(label_int, dtype=torch.long)

        return {
            "session_id": sid,
            "chunk_id": chunk_id,
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
            k_clean = k
            if k_clean.startswith("backbone."): k_clean = k_clean[len("backbone."):]
            if k_clean.startswith("module."): k_clean = k_clean[len("module."):]
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
# 训练：ResNet50 + MultiModalTCMAClassification
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
    mode: str = "fusion"
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 初始化 Dataset (内部会自动执行过滤 <3 的逻辑)
    full_dataset = MAHNOBChunkNPYDataset(
        npy_root=npy_root, rppg_pickle_path=rppg_pickle_path,
        sessions_root=sessions_root, transform=transform, min_len=30
    )
    if len(full_dataset) == 0:
        raise RuntimeError("Dataset is empty. Check paths or filtering logic.")

    # 设置混淆矩阵保存目录
    output_dir = os.path.dirname(save_best_path)
    cm_dir = os.path.join(output_dir, "confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)
    print(f"Confusion matrices will be saved to: {cm_dir}")

    session_ids = [s["session_id"] for s in full_dataset.samples]
    unique_sids = sorted(list(set(session_ids)))
    
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    print(f"Total unique Sessions (after filtering): {len(unique_sids)}. Using {num_folds}-Fold CV.")
    
    feature_extractor = build_feature_extractor(feature_model_path, device=device)
    fold_accuracies = []
    
    # ==================== K-Fold Loop ====================
    for fold, (train_sids_idx, val_sids_idx) in enumerate(kf.split(unique_sids)):
        
        print(f"\n==================== FOLD {fold + 1}/{num_folds} START ====================")
        
        train_sids = [unique_sids[i] for i in train_sids_idx]
        val_sids = [unique_sids[i] for i in val_sids_idx]

        all_indices = np.arange(len(full_dataset))
        train_indices = [i for i in all_indices if full_dataset.samples[i]["session_id"] in train_sids]
        val_indices = [i for i in all_indices if full_dataset.samples[i]["session_id"] in val_sids]

        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        print(f"Fold {fold + 1}: Train {len(train_dataset)}, Val {len(val_dataset)}")
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                  num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                                num_workers=4, pin_memory=True)

        multimodal_model = MultiModalTCMAClassification(num_classes=2, dropout=0.3)
        
        start_epoch = 0
        best_fold_acc = 0.0
        
        if resume_path and os.path.exists(resume_path):
            print(f"Resuming from: {resume_path}")
            ckpt = torch.load(resume_path, map_location="cpu")
            multimodal_model.load_state_dict(ckpt["model_state_dict"], strict=False)
            multimodal_model.to(device)

        multimodal_model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, multimodal_model.parameters()), lr=lr)
        
        for epoch in range(start_epoch, num_epochs):
            # ==================== Train ====================
            multimodal_model.train()
            train_loss_sum, train_correct, train_total = 0.0, 0, 0

            pbar = tqdm(train_loader, desc=f"Fold {fold+1} E{epoch+1} [Train]", ncols=120)
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
                
                B2, m, C2, H2, W2 = videos_sub.shape
                videos_sub = videos_sub.view(B2 * m, C2, H2, W2).to(device, non_blocking=True)
                with torch.no_grad():
                    feats = feature_extractor(videos_sub)
                feats = feats.view(B2, m, -1)

                rppg_seq = rppg_seq.to(device, non_blocking=True)

                optimizer.zero_grad()
                outputs = multimodal_model(rppg_seq, feats, mode=mode)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item() * B2
                _, pred = outputs.max(1)
                train_correct += (pred == labels).sum().item()
                train_total += B2
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{train_correct/max(train_total,1):.3f}"})

            train_acc = train_correct / max(train_total, 1)

            # ==================== Val ====================
            multimodal_model.eval()
            val_loss_sum, val_correct, val_total = 0.0, 0, 0
            
            all_preds = []
            all_labels = []

            with torch.no_grad():
                pbar_val = tqdm(val_loader, desc=f"Fold {fold+1} E{epoch+1} [Val]", ncols=120)
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

                    B2, m, C2, H2, W2 = videos_sub.shape
                    videos_sub = videos_sub.view(B2 * m, C2, H2, W2).to(device, non_blocking=True)
                    feats = feature_extractor(videos_sub)
                    feats = feats.view(B2, m, -1)

                    rppg_seq = rppg_seq.to(device, non_blocking=True)

                    outputs = multimodal_model(rppg_seq, feats, mode=mode)
                    loss = criterion(outputs, labels)

                    val_loss_sum += loss.item() * B2
                    _, pred = outputs.max(1)
                    val_correct += (pred == labels).sum().item()
                    val_total += B2
                    
                    all_preds.extend(pred.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                    pbar_val.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{val_correct/max(val_total,1):.3f}"})

            avg_val_loss = val_loss_sum / max(val_total, 1)
            val_acc = val_correct / max(val_total, 1)
            print(f"[Fold {fold+1} E{epoch+1}] Val Acc: {val_acc:.4f}")

            # 保存混淆矩阵
            cm_filename = f"cm_fold{fold+1}_epoch{epoch+1}_acc{val_acc:.3f}.png"
            cm_path = os.path.join(cm_dir, cm_filename)
            plot_and_save_confusion_matrix(
                all_labels, 
                all_preds, 
                classes=['Low', 'High'], 
                save_path=cm_path,
                title=f'CM Fold {fold+1} Epoch {epoch+1}'
            )

            fold_best_path = save_best_path.replace(".pth", f"_fold{fold+1}.pth")
            torch.save({
                "model_state_dict": multimodal_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "best_val_acc": best_fold_acc
            }, save_last_path)

            if val_acc > best_fold_acc:
                best_fold_acc = val_acc
                torch.save({"model_state_dict": multimodal_model.state_dict()}, fold_best_path)
                print(f"*** New best (val_acc={val_acc:.4f}) ***")
        
        print(f"FOLD {fold + 1} finished. Best: {best_fold_acc:.4f}")
        fold_accuracies.append(best_fold_acc)
        
    print(f"\nMean K-Fold Acc: {np.mean(fold_accuracies):.4f}")

if __name__ == "__main__":
    NPY_ROOT = "./data/MAHNOB-HCI_2/MAHNOB-HCI_SizeW72_SizeH72_ClipLength160_DataTypeRaw_DataAugNone_LabelTypeStandardized_Crop_faceTrue_BackendY5F_Large_boxTrue_Large_size1.5_Dyamic_DetTrue_det_len30_Median_face_boxFalse"
#    RPPG_PICKLE_PATH = "./accurate_pickle/active_sessions_dump.pickle"
    RPPG_PICKLE_PATH = "./accurate_pickle/accurate.pickle"
    SESSIONS_ROOT = "/dataset/MAHNOB-HCI/Sessions"
    FACE_FEATURE_MODEL_PATH = "./weight/face_feature_net.pth"
    
    SAVE_LAST_PATH = "./output_TCMA_arousal_5/multimodal_tcma_mahnob_last_2cls_filtered.pth"
    SAVE_BEST_PATH = "./output_TCMA_arousal_5/multimodal_tcma_mahnob_best_2cls_filtered.pth"
    
    # 记得按需修改 RESUME_PATH
#    RESUME_PATH = "./TCMA_TCCA_weight/TCCA_filiter_5_0.7724.pth"
    RESUME_PATH = "./output_TCMA_arousal_5/multimodal_tcma_mahnob_best_2cls_filtered_fold1.pth"
    # RESUME_PATH = None

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
        lr=e-5,
        device="cuda",
        num_folds=5,
        sample_m=15,
        mode="TCMA"
    )