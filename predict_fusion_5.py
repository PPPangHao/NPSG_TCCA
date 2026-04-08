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

# --- 导入您的自定义模块 ---
from deepseek_1DcNN_gpt import MultiModalTCMAClassification
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
# 1. Arousal-STM 处理器 (升级为三分类版)
# =========================================================
class ArousalSTMProcessor:
    def __init__(self, fs=30, window_sec=6, step_sec=1):
        self.fs = fs
        self.window_size = int(fs * window_sec)
        self.step_size = int(fs * step_sec)
        
        # --- STM 超参数 (需要根据三分类逻辑调整) ---
        self.THETA_AROUSAL = 0.05       # 产生方波的阈值 (Sensitivity)
        self.LAMBDA_A = 0.05            # 衰减率 (Decay)
        
        # --- 双阈值逻辑 ---
        # 强度 I_A <= THETA_CALM : Calm (Class 0)
        # THETA_CALM < I_A <= THETA_HA : Low Arousal (Class 1)
        # I_A > THETA_HA : High Arousal (Class 2)
        
        # 注意：这些阈值需要您使用之前提供的“最佳阈值搜索代码”针对新标签重新搜索
        self.THETA_CALM = 0.2           # 区分 Calm 和 Low Arousal 的阈值
        self.THETA_HA = 0.6             # 区分 Low Arousal 和 High Arousal 的阈值

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

    def run_stm_ternary(self, rppg_signal):
        """
        运行 STM 并返回 Session 级别的 Arousal 三分类预测
        0: Calm (1-3)
        1: Low Arousal (4-7)
        2: High Arousal (8-9)
        """
        # 1. 计算时序 RMSSD
        rmssd_series = self.compute_windowed_rmssd(rppg_signal)
        
        if rmssd_series.size == 0:
            return 0, 0.0  
        
        # 2. STM 逻辑
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
            
        # 3. Session 级判定 (三分类)
        avg_intensity = np.mean(I_A)
        
        if avg_intensity <= self.THETA_CALM:
            prediction = 0 # Calm
        elif avg_intensity <= self.THETA_HA:
            prediction = 1 # Low Arousal
        else:
            prediction = 2 # High Arousal
            
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
# 2. 联合验证函数 (Valence Net + Arousal STM 3-Class)
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
    out_pickle="./five_class_predictions.pkl" # 改名体现5分类
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # ---------- 1. 准备 STM 处理器 ----------
    stm_processor = ArousalSTMProcessor()
    
    print(f"Loading full rPPG data from {rppg_pickle_path} for STM...")
    with open(rppg_pickle_path, 'rb') as f:
        full_rppg_data = pickle.load(f)
    
    session_rppg_map = {}
    for sid, chunks in full_rppg_data['predictions'].items():
        if isinstance(chunks, list):
            flat_chunks = []
            for c in chunks:
                c_np = c.numpy() if hasattr(c, 'numpy') else np.array(c)
                flat_chunks.append(c_np.flatten())
            session_rppg_map[int(sid)] = np.concatenate(flat_chunks)
        else:
            session_rppg_map[int(sid)] = chunks

    # ---------- 2. 准备 Valence 模型与数据 ----------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = MAHNOBChunkNPYDataset(
        npy_root=npy_root,
        rppg_pickle_path=rppg_pickle_path,
        sessions_root=sessions_root,
        transform=transform,
        min_len=30
    )

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    feature_extractor = build_feature_extractor(face_feature_model_path, device=device)
    model = MultiModalTCMAClassification(num_classes=2, dropout=0.0)
    ckpt = torch.load(model_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # ---------- 3. 推理循环 (Valence) ----------
    session_valence_probs = defaultdict(list)

    for batch in tqdm(loader, desc="Valence Network Inference"):
        videos = batch["video"]
        rppg = batch["rppg"]
        # labels = batch["label"] # 这里的 label 是 Valence 的 GT
        session_ids = batch["session_id"]

        B, T, C, H, W = videos.shape
        if T >= sample_m:
            idxs = np.linspace(0, T - 1, sample_m).astype(int)
            videos_sub = torch.stack([videos[b, idxs] for b in range(B)], dim=0)
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
        
        for i in range(B):
            sid = int(session_ids[i])
            session_valence_probs[sid].append(probs[i].cpu().numpy())

    # ---------- 4. Session 级聚合与分类评估 (3-Arousal * 2-Valence) ----------
    results = {}
    
    correct_valence = 0
    correct_arousal = 0
    correct_final_class = 0 # 5分类准确率
    total_samples = 0

    # 遍历所有被处理过的 Session
    for sid in session_valence_probs.keys():
        
        # --- A. Valence 预测 (网络) ---
        # 0: Low Valence, 1: High Valence
        v_probs = np.stack(session_valence_probs[sid], axis=0)
        v_mean_prob = v_probs.mean(axis=0)
        pred_valence = int(v_mean_prob.argmax()) 
        
        # --- B. Arousal 预测 (STM 三分类) ---
        # 0: Calm, 1: Low Arousal, 2: High Arousal
        if sid in session_rppg_map:
            full_signal = session_rppg_map[sid]
            pred_arousal, stm_intensity = stm_processor.run_stm_ternary(full_signal)
        else:
            print(f"Warning: Session {sid} not found in rPPG pickle.")
            pred_arousal = 0 
            stm_intensity = 0.0

        # --- C. 获取真值并映射 ---
        xml_path = os.path.join(sessions_root, str(sid), "session.xml")
        gt_aro_score, gt_val_score = stm_processor.get_ground_truth_from_xml(xml_path)
        
        if gt_aro_score is None or gt_val_score is None:
            continue 
            
        # 1. Valence 真值 (2分类)
        # 假设 < 5 为 Low (0), >= 5 为 High (1)
        gt_valence_label = 1 if gt_val_score >= 5 else 0

        # 2. Arousal 真值 (3分类) [修改核心]
        # 1-3: Calm (0)
        # 4-7: Low Arousal (1)
        # 8-9: High Arousal (2)
        if gt_aro_score <= 4:
            gt_arousal_label = 0
        elif gt_aro_score <= 7:
            gt_arousal_label = 1
        else:
            gt_arousal_label = 2

        # --- D. 统计 ---
        total_samples += 1
        
        # Valence Acc
        if pred_valence == gt_valence_label:
            correct_valence += 1
            
        # Arousal Acc (三分类)
        if pred_arousal == gt_arousal_label:
            correct_arousal += 1
            
        # --- E. 5分类映射 ---
        # 逻辑：
        # 如果 Arousal 是 Calm (0) -> Class 0 (Neutral/Calm)
        # 如果 Arousal 是 LA (1):
        #    Valence Low (0) -> Class 1 (LA-LV)
        #    Valence High (1) -> Class 2 (LA-HV)
        # 如果 Arousal 是 HA (2):
        #    Valence Low (0) -> Class 3 (HA-LV)
        #    Valence High (1) -> Class 4 (HA-HV)
        
        def get_5_class(aro, val):
            if aro == 0: return 0
            if aro == 1: return 1 + val # 1或2
            if aro == 2: return 3 + val # 3或4
            return 0

        pred_final = get_5_class(pred_arousal, pred_valence)
        gt_final = get_5_class(gt_arousal_label, gt_valence_label)
        
        if pred_final == gt_final:
            correct_final_class += 1
            
        results[sid] = {
            "gt_scores": (gt_aro_score, gt_val_score),
            "gt_labels": (gt_arousal_label, gt_valence_label), # (0-2, 0-1)
            "pred_labels": (pred_arousal, pred_valence),       # (0-2, 0-1)
            "gt_5class": gt_final,
            "pred_5class": pred_final,
            "stm_intensity": stm_intensity,
            "val_prob": v_mean_prob.tolist()
        }

    # ---------- 5. 输出结果 ----------
    acc_v = correct_valence / total_samples if total_samples > 0 else 0
    acc_a = correct_arousal / total_samples if total_samples > 0 else 0
    acc_final = correct_final_class / total_samples if total_samples > 0 else 0

    print("\n============== Combined System Validation (5-Class) ==============")
    print(f"Total Sessions Processed : {total_samples}")
    print(f"Valence Accuracy (2-class): {acc_v:.4f}")
    print(f"Arousal Accuracy (3-class): {acc_a:.4f} (Calm/Low/High)")
    print(f"Final 5-Class Accuracy    : {acc_final:.4f}")
    print(f"Saved results to          : {out_pickle}")
    print("================================================================")
    
    with open(out_pickle, "wb") as f:
        pickle.dump(results, f)

    return acc_final, results

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
        out_pickle="./output/final_five_class_predictions.pkl"
    )