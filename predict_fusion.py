# import os
# import pickle
# import numpy as np
# import xml.etree.ElementTree as ET
# from tqdm import tqdm
# from collections import defaultdict

# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from torchvision import transforms

# # --- 导入您的自定义模块 ---
# from deepseek_1DcNN_gpt_no_refine  import MultiModalTCMAClassification
# from deepseek_1DcNN_gpt_train_2class import MAHNOBChunkNPYDataset, build_feature_extractor

# def check_data_root(root_path, session_id=5):
#     """检查根路径和单个文件的可访问性"""
#     if not os.path.exists(root_path):
#         print(f"FATAL ERROR: SESSIONS_ROOT path does NOT exist: {root_path}")
#         return False
    
#     # 尝试检查一个示例 session 文件
#     example_path = os.path.join(root_path, str(session_id), "session.xml")
#     if not os.path.exists(example_path):
#         print(f"WARNING: SESSIONS_ROOT exists, but example file not found at: {example_path}")
#         return False
    
#     print(f"SUCCESS: Data root and example file check passed.")
#     return True

# # =========================================================
# # 1. Arousal-STM 处理器 (恢复为二分类版)
# # =========================================================
# class ArousalSTMProcessor:
#     def __init__(self, fs=30, window_sec=6, step_sec=1):
#         self.fs = fs
#         self.window_size = int(fs * window_sec)
#         self.step_size = int(fs * step_sec)
        
#         # --- STM 超参数 ---
#         # 如果您之前搜索过最佳阈值，请在这里填入
#         self.THETA_AROUSAL = 0.05       # 产生方波的阈值
#         self.LAMBDA_A = 0.05            # 衰减率
#         self.THETA_POOL_A = 0.5         # 最终触发判定阈值 (0: Low, 1: High)

#     def compute_windowed_rmssd(self, rppg_signal):
#         rmssd_series = []
#         L = len(rppg_signal)
#         if L < self.window_size:
#             return np.array([])
        
#         for i in range(0, L - self.window_size + 1, self.step_size):
#             window = rppg_signal[i : i + self.window_size]
#             diff_signal = np.diff(window)
#             rmssd = np.sqrt(np.mean(diff_signal ** 2))
#             rmssd_series.append(rmssd)
#         return np.array(rmssd_series)

#     def run_stm(self, rppg_signal):
#         """
#         运行 STM 并返回 Session 级别的 Arousal 二分类预测
#         0: Low Arousal
#         1: High Arousal
#         """
#         # 1. 计算时序 RMSSD
#         rmssd_series = self.compute_windowed_rmssd(rppg_signal)
        
#         if rmssd_series.size == 0:
#             return 0, 0.0  
        
#         # 2. STM 逻辑
#         # A. 计算残差
#         rmssd_mean = np.mean(rmssd_series)
#         rmssd_res = np.abs(rmssd_series - rmssd_mean)
        
#         # B. 方波提取
#         S_A = (rmssd_res > self.THETA_AROUSAL).astype(float)
        
#         # C. 状态累积 (I_A)
#         I_A = np.zeros_like(S_A)
#         time_step_val = self.step_size / self.fs
#         decay_factor = np.exp(-self.LAMBDA_A * time_step_val)
        
#         I_A[0] = S_A[0]
#         for t in range(1, len(S_A)):
#             I_A[t] = S_A[t] + I_A[t-1] * decay_factor
            
#         # 3. Session 级判定 (二分类)
#         avg_intensity = np.mean(I_A)
#         prediction = 1 if avg_intensity > self.THETA_POOL_A else 0
            
#         return prediction, avg_intensity

#     @staticmethod
#     def get_ground_truth_from_xml(xml_path):
#         """解析 XML 获取 Arousal 和 Valence 真值"""
#         if not os.path.exists(xml_path):
#             return None, None
            
#         try:
#             tree = ET.parse(xml_path)
#             root = tree.getroot() 
#             aro_val_str = root.get('feltArsl')
#             val_val_str = root.get('feltVlnc')
            
#             aro_val = float(aro_val_str) if aro_val_str else None
#             val_val = float(val_val_str) if val_val_str else None
            
#             if aro_val is None or val_val is None:
#                 return None, None
            
#             return aro_val, val_val
#         except Exception as e:
#             print(f"XML PARSE ERROR: {xml_path}. Error: {e}")
#             return None, None

# # =========================================================
# # 2. 联合验证函数 (Valence Net + Arousal STM 2-Class)
# # =========================================================
# @torch.no_grad()
# def validate_combined_system(
#     npy_root,
#     rppg_pickle_path,
#     sessions_root,
#     model_ckpt,
#     face_feature_model_path=None,
#     batch_size=32,
#     device="cuda",
#     sample_m=15,
#     out_pickle="./four_class_predictions.pkl" # 改回4分类命名
# ):
#     device = torch.device(device if torch.cuda.is_available() else "cpu")

#     # ---------- 1. 准备 STM 处理器 ----------
#     stm_processor = ArousalSTMProcessor()
    
#     print(f"Loading full rPPG data from {rppg_pickle_path} for STM...")
#     with open(rppg_pickle_path, 'rb') as f:
#         full_rppg_data = pickle.load(f)
    
#     session_rppg_map = {}
#     for sid, chunks in full_rppg_data['predictions'].items():
#         if isinstance(chunks, list):
#             flat_chunks = []
#             for c in chunks:
#                 c_np = c.numpy() if hasattr(c, 'numpy') else np.array(c)
#                 flat_chunks.append(c_np.flatten())
#             session_rppg_map[int(sid)] = np.concatenate(flat_chunks)
#         else:
#             session_rppg_map[int(sid)] = chunks

#     # ---------- 2. 准备 Valence 模型与数据 ----------
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])

#     dataset = MAHNOBChunkNPYDataset(
#         npy_root=npy_root,
#         rppg_pickle_path=rppg_pickle_path,
#         sessions_root=sessions_root,
#         transform=transform,
#         min_len=30
#     )

#     loader = DataLoader(
#         dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
#     )

#     feature_extractor = build_feature_extractor(face_feature_model_path, device=device)
#     model = MultiModalTCMAClassification(num_classes=2, dropout=0.0)
#     ckpt = torch.load(model_ckpt, map_location="cpu")
#     model.load_state_dict(ckpt["model_state_dict"])
#     model.to(device)
#     model.eval()

#     # ---------- 3. 推理循环 (Valence) ----------
#     session_valence_probs = defaultdict(list)

#     for batch in tqdm(loader, desc="Valence Network Inference"):
#         videos = batch["video"]
#         rppg = batch["rppg"]
#         session_ids = batch["session_id"]

#         B, T, C, H, W = videos.shape
#         if T >= sample_m:
#             idxs = np.linspace(0, T - 1, sample_m).astype(int)
#             videos_sub = torch.stack([videos[b, idxs] for b in range(B)], dim=0)
#         else:
#             videos_sub = videos
#             sample_m = T

#         B2, m, C2, H2, W2 = videos_sub.shape
#         videos_sub = videos_sub.view(B2 * m, C2, H2, W2).to(device)
#         feats = feature_extractor(videos_sub)
#         feats = feats.view(B2, m, -1)

#         rppg = rppg.to(device)
#         logits = model(rppg, feats)
#         probs = F.softmax(logits, dim=1)
        
#         for i in range(B):
#             sid = int(session_ids[i])
#             session_valence_probs[sid].append(probs[i].cpu().numpy())

#     # ---------- 4. Session 级聚合与分类评估 (2-Arousal * 2-Valence) ----------
#     results = {}
    
#     correct_valence = 0
#     correct_arousal = 0
#     correct_four_class = 0 
#     total_samples = 0

#     # 遍历所有被处理过的 Session
#     for sid in session_valence_probs.keys():
        
#         # --- A. Valence 预测 (网络) ---
#         v_probs = np.stack(session_valence_probs[sid], axis=0)
#         v_mean_prob = v_probs.mean(axis=0)
#         pred_valence = int(v_mean_prob.argmax()) 
        
#         # --- B. Arousal 预测 (STM 二分类) ---
#         if sid in session_rppg_map:
#             full_signal = session_rppg_map[sid]
#             pred_arousal, stm_intensity = stm_processor.run_stm(full_signal)
#         else:
#             print(f"Warning: Session {sid} not found in rPPG pickle.")
#             pred_arousal = 0 
#             stm_intensity = 0.0

#         # --- C. 获取真值并映射 ---
#         xml_path = os.path.join(sessions_root, str(sid), "session.xml")
#         gt_aro_score, gt_val_score = stm_processor.get_ground_truth_from_xml(xml_path)
        
#         if gt_aro_score is None or gt_val_score is None:
#             continue 
            
#         # 1. Valence 真值 (2分类)
#         # 假设 < 5 为 Low (0), >= 5 为 High (1)
#         gt_valence_binary = 1 if gt_val_score >= 5 else 0

#         # 2. Arousal 真值 (2分类) [恢复为 >7 High]
#         # <= 7: Low Arousal (0)
#         # > 7 : High Arousal (1)
#         gt_arousal_binary = 1 if gt_aro_score > 7 else 0

#         # --- D. 统计 ---
#         total_samples += 1
        
#         # Valence Acc
#         if pred_valence == gt_valence_binary:
#             correct_valence += 1
            
#         # Arousal Acc (二分类)
#         if pred_arousal == gt_arousal_binary:
#             correct_arousal += 1
            
#         # --- E. 4分类映射 ---
#         # 0: LA-LV (0,0)
#         # 1: LA-HV (0,1)
#         # 2: HA-LV (1,0)
#         # 3: HA-HV (1,1)
#         pred_four = pred_arousal * 2 + pred_valence
#         gt_four = gt_arousal_binary * 2 + gt_valence_binary
        
#         if pred_four == gt_four:
#             correct_four_class += 1
            
#         results[sid] = {
#             "gt_scores": (gt_aro_score, gt_val_score),
#             "gt_binary": (gt_arousal_binary, gt_valence_binary),
#             "pred_binary": (pred_arousal, pred_valence),
#             "gt_four": gt_four,
#             "pred_four": pred_four,
#             "stm_intensity": stm_intensity,
#             "val_prob": v_mean_prob.tolist()
#         }

#     # ---------- 5. 输出结果 ----------
#     acc_v = correct_valence / total_samples if total_samples > 0 else 0
#     acc_a = correct_arousal / total_samples if total_samples > 0 else 0
#     acc_4 = correct_four_class / total_samples if total_samples > 0 else 0

#     print("\n============== Combined System Validation (4-Class) ==============")
#     print(f"Total Sessions Processed : {total_samples}")
#     print(f"Valence Accuracy (Net)   : {acc_v:.4f}")
#     print(f"Arousal Accuracy (STM)   : {acc_a:.4f} (Threshold > 7)")
#     print(f"4-Class Accuracy         : {acc_4:.4f}")
#     print(f"Saved results to         : {out_pickle}")
#     print("================================================================")
    
#     with open(out_pickle, "wb") as f:
#         pickle.dump(results, f)

#     return acc_4, results

# # =========================================================
# # Main Execution
# # =========================================================
# if __name__ == "__main__":
#     # 配置您的路径
#     NPY_ROOT = "./data/MAHNOB-HCI_2/MAHNOB-HCI_SizeW72_SizeH72_ClipLength160_DataTypeRaw_DataAugNone_LabelTypeStandardized_Crop_faceTrue_BackendY5F_Large_boxTrue_Large_size1.5_Dyamic_DetTrue_det_len30_Median_face_boxFalse"
#     RPPG_PICKLE_PATH = "./accurate_pickle/accurate.pickle"
#     SESSIONS_ROOT = "/dataset/MAHNOB-HCI/Sessions"
    
#     check_data_root(SESSIONS_ROOT, session_id=1042) 
    
#     # MODEL_CKPT = "./output/multimodal_tcma_mahnob_best_2cls.pth"
#     MODEL_CKPT = "./output_origin_result/multimodal_tcma_mahnob_best_2cls_fold1.pth"
#     FACE_FEATURE_MODEL_PATH = "./weight/face_feature_net.pth"

#     validate_combined_system(
#         npy_root=NPY_ROOT,
#         rppg_pickle_path=RPPG_PICKLE_PATH,
#         sessions_root=SESSIONS_ROOT,
#         model_ckpt=MODEL_CKPT,
#         face_feature_model_path=FACE_FEATURE_MODEL_PATH,
#         batch_size=32,
#         device="cuda", 
#         sample_m=15,
#         out_pickle="./output/final_four_class_predictions.pkl"
#     )

import os
import pickle
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.model_selection import KFold
from torch.utils.data import Subset

# --- 导入您的自定义模块 ---
from deepseek_1DcNN_gpt_no_refine  import MultiModalTCMAClassification
from deepseek_1DcNN_gpt_train_2class import MAHNOBChunkNPYDataset, build_feature_extractor

def check_data_root(root_path, session_id=5):
    """检查根路径和单个文件的可访问性"""
    if not os.path.exists(root_path):
        print(f"FATAL ERROR: SESSIONS_ROOT path does NOT exist: {root_path}")
        return False
    
    # 尝试检查一个示例 session 文件
    example_path = os.path.join(root_path, str(session_id), "session.xml")
    if not os.path.exists(example_path):
        print(f"WARNING: SESSIONS_ROOT exists, but example file not found at: {example_path}")
        return False
    
    print(f"SUCCESS: Data root and example file check passed.")
    return True

# =========================================================
# 1. Arousal-STM 处理器 (恢复为二分类版)
# =========================================================
class ArousalSTMProcessor:
    def __init__(self, fs=30, window_sec=6, step_sec=1):
        self.fs = fs
        self.window_size = int(fs * window_sec)
        self.step_size = int(fs * step_sec)
        
        # --- STM 超参数 ---
        # 如果您之前搜索过最佳阈值，请在这里填入
        self.THETA_AROUSAL = 0.05       # 产生方波的阈值
        self.LAMBDA_A = 0.05            # 衰减率
        self.THETA_POOL_A = 0.5         # 最终触发判定阈值 (0: Low, 1: High)

    def compute_windowed_rmssd(self, rppg_signal):
        rmssd_series = []
        L = len(rppg_signal)
        if L < self.window_size:
            return np.array([])
        
        for i in range(0, L - self.window_size + 1, self.step_size):
            window = rppg_signal[i : i + self.window_size]
            diff_signal = np.diff(window)
            rmssd = np.sqrt(np.mean(diff_signal ** 2))
            rmssd_series.append(rmssd)
        return np.array(rmssd_series)

    def run_stm(self, rppg_signal):
        """
        运行 STM 并返回 Session 级别的 Arousal 二分类预测
        0: Low Arousal
        1: High Arousal
        """
        # 1. 计算时序 RMSSD
        rmssd_series = self.compute_windowed_rmssd(rppg_signal)
        
        if rmssd_series.size == 0:
            return 0, 0.0  
        
        # 2. STM 逻辑
        # A. 计算残差
        rmssd_mean = np.mean(rmssd_series)
        rmssd_res = np.abs(rmssd_series - rmssd_mean)
        
        # B. 方波提取
        S_A = (rmssd_res > self.THETA_AROUSAL).astype(float)
        
        # C. 状态累积 (I_A)
        I_A = np.zeros_like(S_A)
        time_step_val = self.step_size / self.fs
        decay_factor = np.exp(-self.LAMBDA_A * time_step_val)
        
        I_A[0] = S_A[0]
        for t in range(1, len(S_A)):
            I_A[t] = S_A[t] + I_A[t-1] * decay_factor
            
        # 3. Session 级判定 (二分类)
        avg_intensity = np.mean(I_A)
        prediction = 1 if avg_intensity > self.THETA_POOL_A else 0
            
        return prediction, avg_intensity

    @staticmethod
    def get_ground_truth_from_xml(xml_path):
        """解析 XML 获取 Arousal 和 Valence 真值"""
        if not os.path.exists(xml_path):
            return None, None
            
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot() 
            aro_val_str = root.get('feltArsl')
            val_val_str = root.get('feltVlnc')
            
            aro_val = float(aro_val_str) if aro_val_str else None
            val_val = float(val_val_str) if val_val_str else None
            
            if aro_val is None or val_val is None:
                return None, None
            
            return aro_val, val_val
        except Exception as e:
            print(f"XML PARSE ERROR: {xml_path}. Error: {e}")
            return None, None

# =========================================================
# 2. 联合验证函数 (Valence Net + Arousal STM 2-Class)
# =========================================================
@torch.no_grad()
def validate_combined_system(
    npy_root,
    rppg_pickle_path,
    sessions_root,
    model_ckpt,
    face_feature_model_path=None,
    batch_size=32,
    device="cuda",
    sample_m=15,
    num_folds=5,
    fold_idx=0,
    random_state=42,
    out_pickle="./val_four_class_predictions.pkl"
):
    """
    在指定 fold 的 validation sessions 上评估：
    - Valence (Net)
    - Arousal (STM)
    - 4-Class 准确率

    ⚠️ 随机方式与训练完全一致（KFold + shuffle + random_state）
    """

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # =========================================================
    # 1. STM 处理器 & rPPG 全量加载
    # =========================================================
    stm_processor = ArousalSTMProcessor()

    with open(rppg_pickle_path, "rb") as f:
        full_rppg_data = pickle.load(f)

    session_rppg_map = {}
    for sid, chunks in full_rppg_data["predictions"].items():
        if isinstance(chunks, list):
            flat = []
            for c in chunks:
                c = c.numpy() if hasattr(c, "numpy") else np.asarray(c)
                flat.append(c.flatten())
            session_rppg_map[int(sid)] = np.concatenate(flat)
        else:
            session_rppg_map[int(sid)] = chunks

    # =========================================================
    # 2. 构建 Dataset（全量）
    # =========================================================
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    full_dataset = MAHNOBChunkNPYDataset(
        npy_root=npy_root,
        rppg_pickle_path=rppg_pickle_path,
        sessions_root=sessions_root,
        transform=transform,
        min_len=30
    )

    # =========================================================
    # 3. Session 级 KFold（关键：与训练一致）
    # =========================================================
    # 1. 收集所有 chunk 对应的 session_id
    all_chunk_session_ids = []
    for i in range(len(full_dataset)):
        all_chunk_session_ids.append(int(full_dataset[i]["session_id"]))

    # 2. 去重得到 session 列表
    all_sessions = sorted(set(all_chunk_session_ids))


    kf = KFold(
        n_splits=num_folds,
        shuffle=True,
        random_state=random_state
    )

    folds = list(kf.split(all_sessions))
    _, val_session_idx = folds[fold_idx]
    val_sessions = set(np.array(all_sessions)[val_session_idx])

    # =========================================================
    # 4. 构建 Val 子集（chunk 级，但 session 已固定）
    # =========================================================
    val_indices = [
        i for i, sid in enumerate(all_chunk_session_ids)
        if sid in val_sessions
    ]

    val_dataset = Subset(full_dataset, val_indices)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"[VAL] Fold {fold_idx} | Sessions: {len(val_sessions)} | Chunks: {len(val_indices)}")

    # =========================================================
    # 5. 加载模型
    # =========================================================
    feature_extractor = build_feature_extractor(
        face_feature_model_path, device=device
    )

    model = MultiModalTCMAClassification(num_classes=2, dropout=0.0)
    ckpt = torch.load(model_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    # =========================================================
    # 6. Valence 网络推理（Session 聚合）
    # =========================================================
    session_valence_probs = defaultdict(list)

    for batch in tqdm(val_loader, desc="Valence Inference (VAL)"):
        videos = batch["video"]
        rppg = batch["rppg"].to(device)
        session_ids = batch["session_id"]

        B, T, C, H, W = videos.shape

        if T >= sample_m:
            idxs = np.linspace(0, T - 1, sample_m).astype(int)
            videos = torch.stack([videos[b, idxs] for b in range(B)])
        else:
            sample_m = T

        videos = videos.view(B * sample_m, C, H, W).to(device)
        feats = feature_extractor(videos)
        feats = feats.view(B, sample_m, -1)

        logits = model(rppg, feats)
        probs = F.softmax(logits, dim=1)

        for i in range(B):
            sid = int(session_ids[i])
            session_valence_probs[sid].append(probs[i].cpu().numpy())

    # =========================================================
    # 7. Session 级评估
    # =========================================================
    correct_v, correct_a, correct_4 = 0, 0, 0
    total = 0
    results = {}

    for sid in session_valence_probs.keys():

        # ---- Valence ----
        v_probs = np.stack(session_valence_probs[sid])
        pred_val = int(v_probs.mean(axis=0).argmax())

        # ---- Arousal (STM) ----
        pred_aro, intensity = stm_processor.run_stm(
            session_rppg_map.get(sid, np.array([]))
        )

        # ---- GT ----
        xml_path = os.path.join(sessions_root, str(sid), "session.xml")
        gt_aro, gt_val = stm_processor.get_ground_truth_from_xml(xml_path)
        if gt_aro is None:
            continue

        gt_val_bin = 1 if gt_val >= 5 else 0
        gt_aro_bin = 1 if gt_aro > 7 else 0

        total += 1

        if pred_val == gt_val_bin:
            correct_v += 1
        if pred_aro == gt_aro_bin:
            correct_a += 1

        pred_4 = pred_aro * 2 + pred_val
        gt_4 = gt_aro_bin * 2 + gt_val_bin
        if pred_4 == gt_4:
            correct_4 += 1

        results[sid] = {
            "pred": (pred_aro, pred_val),
            "gt": (gt_aro_bin, gt_val_bin),
            "pred_4": pred_4,
            "gt_4": gt_4,
            "stm_intensity": intensity,
            "valence_prob": v_probs.mean(axis=0).tolist()
        }

    # =========================================================
    # 8. 输出
    # =========================================================
    acc_v = correct_v / total if total else 0
    acc_a = correct_a / total if total else 0
    acc_4 = correct_4 / total if total else 0

    print("\n========== VALIDATION RESULT ==========")
    print(f"Fold                : {fold_idx}")
    print(f"Total Val Sessions  : {total}")
    print(f"Valence Acc         : {acc_v:.4f}")
    print(f"Arousal Acc         : {acc_a:.4f}")
    print(f"4-Class Acc         : {acc_4:.4f}")
    print("======================================")

    with open(out_pickle, "wb") as f:
        pickle.dump(results, f)

    return acc_4, results

# =========================================================
# Main Execution
# =========================================================
if __name__ == "__main__":
    # 配置您的路径
    NPY_ROOT = "./data/MAHNOB-HCI_2/MAHNOB-HCI_SizeW72_SizeH72_ClipLength160_DataTypeRaw_DataAugNone_LabelTypeStandardized_Crop_faceTrue_BackendY5F_Large_boxTrue_Large_size1.5_Dyamic_DetTrue_det_len30_Median_face_boxFalse"
    RPPG_PICKLE_PATH = "./accurate_pickle/accurate.pickle"
    SESSIONS_ROOT = "/dataset/MAHNOB-HCI/Sessions"
    
    check_data_root(SESSIONS_ROOT, session_id=1042) 
    
    MODEL_CKPT = "./output/multimodal_tcma_mahnob_best_2cls.pth"
    # MODEL_CKPT = "./output_origin_result/multimodal_tcma_mahnob_best_2cls_fold1.pth"
    FACE_FEATURE_MODEL_PATH = "./weight/face_feature_net.pth"

    validate_combined_system(
        npy_root=NPY_ROOT,
        rppg_pickle_path=RPPG_PICKLE_PATH,
        sessions_root=SESSIONS_ROOT,
        model_ckpt=MODEL_CKPT,
        face_feature_model_path=FACE_FEATURE_MODEL_PATH,
        batch_size=32,
        device="cuda", 
        sample_m=15,
        num_folds=5,
        fold_idx=0,   # fold1
        out_pickle="./output/final_four_class_predictions.pkl"
    )
