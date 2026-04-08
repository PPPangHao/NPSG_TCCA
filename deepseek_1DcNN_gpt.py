import os
import math
import pickle
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


# ===================== #
#   小工具：打印 shape   #
# ===================== #
def _log(name, x, enabled=True):
    if not enabled:
        return
    if isinstance(x, torch.Tensor):
        print(f"[{name}] shape={tuple(x.shape)} dtype={x.dtype} device={x.device}")
    else:
        print(f"[{name}] (not tensor) type={type(x)}")


# ===================== #
#      rPPG Blocks      #
# ===================== #
class RPPGBlock1(nn.Module):
    """
    RppgBlock1
    Conv1D[filter=16,kernel_size=20,stride=1]+BN1D+ReLU+Dropout(0.3)
    MaxPool1d[2]
    Conv1D[filter=32,kernel_size=10,stride=1]+BN1D+ReLU+Dropout(0.3)
    MaxPool1d[2]
    input:  (B,1,T)
    output: (B,32,Tr1)
    """

    def __init__(self, in_channels=1, dropout=0.3, debug=False):
        super().__init__()
        self.debug = debug
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=20, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(16)
        self.drop1 = nn.Dropout(dropout)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=10, stride=1, padding=0)
        self.bn2 = nn.BatchNorm1d(32)
        self.drop2 = nn.Dropout(dropout)
        self.pool2 = nn.MaxPool1d(2)

    def forward(self, x):
        _log("RPPGBlock1/in", x, self.debug)  # (B,1,T)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.drop1(x)
        x = self.pool1(x)
        _log("RPPGBlock1/after_pool1", x, self.debug)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.drop2(x)
        x = self.pool2(x)
        _log("RPPGBlock1/after_pool2", x, self.debug)
        return x


class RPPGBlock2(nn.Module):
    """
    rppgBlock2
    Conv1D[filter=64,kernel_size=3,stride=1]+BN1D+ReLU+Dropout(0.3)
    MaxPool1d[2]
    Conv1D[filter=128,kernel_size=10,stride=1]+BN1D+ReLU+Dropout(0.3)
    MaxPool1d[2]
    input:  (B,64,T)
    output: (B,128,Tr2)
    """

    def __init__(self, in_channels=64, dropout=0.3, debug=False):
        super().__init__()
        self.debug = debug
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.drop1 = nn.Dropout(dropout)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=10, stride=1, padding=0)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(dropout)
        self.pool2 = nn.MaxPool1d(2)

    def forward(self, x):
        _log("RPPGBlock2/in", x, self.debug)  # (B,64,T)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.drop1(x)
        x = self.pool1(x)
        _log("RPPGBlock2/after_pool1", x, self.debug)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.drop2(x)
        x = self.pool2(x)
        _log("RPPGBlock2/after_pool2", x, self.debug)
        return x


# ===================== #
#      Face Blocks      #
# ===================== #
class FaceBlock1(nn.Module):
    """
    Faceblock1
    Conv1D[filter=256,kernel_size=3,stride=1]+BN1D+ReLU+Dropout(0.3)
    Conv1D[filter=64,kernel_size=3,stride=1]+BN1D+ReLU+Dropout(0.3)

    输入期望: (B,T,2048) 或 (B,2048,T)
    输出:     (B,64,T)
    """

    def __init__(self, in_channels=2048, dropout=0.3, debug=False):
        super().__init__()
        self.debug = debug
        self.conv1 = nn.Conv1d(in_channels, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(256, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        _log("FaceBlock1/in_raw", x, self.debug)
        if x.dim() == 3 and x.shape[2] == 2048:
            # (B,T,2048) -> (B,2048,T)
            x = x.permute(0, 2, 1).contiguous()
        _log("FaceBlock1/in(B,2048,T)", x, self.debug)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.drop2(x)
        _log("FaceBlock1/out(B,64,T)", x, self.debug)
        return x


class FaceBlock2(nn.Module):
    """
    FaceBlock2
    Conv1D[filter=128,kernel_size=3,stride=1]+BN1D+ReLU+Dropout(0.3)
    然后做全局平均池化得到 (B,128) 的向量
    """

    def __init__(self, in_channels=64, dropout=0.3, debug=False):
        super().__init__()
        self.debug = debug
        self.conv1 = nn.Conv1d(in_channels, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(dropout)
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # -> (B,128,1)

    def forward(self, x):
        _log("FaceBlock2/in(B,64,T)", x, self.debug)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.drop1(x)
        _log("FaceBlock2/conv_out(B,128,T)", x, self.debug)

        x = self.global_pool(x)  # (B,128,1)
        _log("FaceBlock2/gap(B,128,1)", x, self.debug)
        x = x.view(x.size(0), -1)  # (B,128)
        _log("FaceBlock2/out(B,128)", x, self.debug)
        return x


# ===================== #
#         TCMA          #
# ===================== #
class TCMA_R2F(nn.Module):
    """
    rPPG -> Face 方向的跨模态注意力:
    - rPPG token 作为 Query
    - Face token 作为 Key/Value
    输入:
      Xf_tokens: (B,Nf,Cf=64)
      Xr_tokens: (B,Nr,Cr=32)
    输出:
      fused_rppg_tokens: (B,Nr,d_v) 这里 d_v=64, 方便接 RPPGBlock2
    """

    def __init__(self,
                 dim_face_token=64,
                 dim_rppg_token=32,
                 d_k=64,
                 d_v=64,
                 n_heads=4,
                 post_layers=1,
                 debug=False):
        super().__init__()
        self.debug = debug
        assert d_k % n_heads == 0 and d_v % n_heads == 0
        self.n_heads = n_heads
        self.head_k = d_k // n_heads
        self.head_v = d_v // n_heads

        self.Wq_r = nn.Linear(dim_rppg_token, d_k)
        self.Wk_f = nn.Linear(dim_face_token, d_k)
        self.Wv_f = nn.Linear(dim_face_token, d_v)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_v,
            nhead=n_heads,
            dim_feedforward=4 * d_v,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=post_layers)

        self.rppg_to_dv = nn.Linear(dim_rppg_token, d_v)
        self.fusion_proj = nn.Linear(d_v + d_v, d_v)

    def forward(self, Xf_tokens, Xr_tokens):
        _log("TCMA_R2F/face_tokens(B,Nf,64)", Xf_tokens, self.debug)
        _log("TCMA_R2F/rppg_tokens(B,Nr,32)", Xr_tokens, self.debug)

        B, Nf, Cf = Xf_tokens.shape
        B2, Nr, Cr = Xr_tokens.shape
        assert B == B2

        K = self.Wk_f(Xf_tokens)  # (B,Nf,d_k)
        V = self.Wv_f(Xf_tokens)  # (B,Nf,d_v)
        Q = self.Wq_r(Xr_tokens)  # (B,Nr,d_k)

        Qh = Q.view(B, Nr, self.n_heads, self.head_k).permute(0, 2, 1, 3)  # (B,H,Nr,dk/H)
        Kh = K.view(B, Nf, self.n_heads, self.head_k).permute(0, 2, 3, 1)  # (B,H,dk/H,Nf)
        Vh = V.view(B, Nf, self.n_heads, self.head_v).permute(0, 2, 1, 3)  # (B,H,Nf,dv/H)

        scores = torch.matmul(Qh, Kh) / math.sqrt(self.head_k)  # (B,H,Nr,Nf)
        attn = torch.softmax(scores, dim=-1)
        att = torch.matmul(attn, Vh)  # (B,H,Nr,dv/H)
        att = att.permute(0, 2, 1, 3).contiguous().view(B, Nr, -1)  # (B,Nr,d_v)

        att_out = self.transformer(att)  # (B,Nr,d_v)
        rppg_proj = self.rppg_to_dv(Xr_tokens)  # (B,Nr,d_v)
        fused = torch.cat([att_out, rppg_proj], dim=-1)  # (B,Nr,2*d_v)
        fused = self.fusion_proj(fused)  # (B,Nr,d_v)
        _log("TCMA_R2F/fused(B,Nr,d_v)", fused, self.debug)
        return fused


class TCMA_F2R(nn.Module):
    """
    Face -> rPPG 方向的跨模态注意力:
    - Face token 作为 Query
    - rPPG token 作为 Key/Value
    输入:
      Xf_tokens: (B,Nf,64)
      Xr_tokens: (B,Nr,32)
    输出:
      fused_face_tokens: (B,Nf,d_v=64) 方便接 FaceBlock2
    """

    def __init__(self,
                 dim_face_token=64,
                 dim_rppg_token=32,
                 d_k=64,
                 d_v=64,
                 n_heads=4,
                 post_layers=1,
                 debug=False):
        super().__init__()
        self.debug = debug
        assert d_k % n_heads == 0 and d_v % n_heads == 0
        self.n_heads = n_heads
        self.head_k = d_k // n_heads
        self.head_v = d_v // n_heads

        self.Wq_f = nn.Linear(dim_face_token, d_k)
        self.Wk_r = nn.Linear(dim_rppg_token, d_k)
        self.Wv_r = nn.Linear(dim_rppg_token, d_v)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_v,
            nhead=n_heads,
            dim_feedforward=4 * d_v,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=post_layers)

        self.face_to_dv = nn.Linear(dim_face_token, d_v)
        self.fusion_proj = nn.Linear(d_v + d_v, d_v)

    def forward(self, Xf_tokens, Xr_tokens):
        _log("TCMA_F2R/face_tokens(B,Nf,64)", Xf_tokens, self.debug)
        _log("TCMA_F2R/rppg_tokens(B,Nr,32)", Xr_tokens, self.debug)

        B, Nf, Cf = Xf_tokens.shape
        B2, Nr, Cr = Xr_tokens.shape
        assert B == B2

        Q = self.Wq_f(Xf_tokens)  # (B,Nf,d_k)
        K = self.Wk_r(Xr_tokens)  # (B,Nr,d_k)
        V = self.Wv_r(Xr_tokens)  # (B,Nr,d_v)

        Qh = Q.view(B, Nf, self.n_heads, self.head_k).permute(0, 2, 1, 3)
        Kh = K.view(B, Nr, self.n_heads, self.head_k).permute(0, 2, 3, 1)
        Vh = V.view(B, Nr, self.n_heads, self.head_v).permute(0, 2, 1, 3)

        scores = torch.matmul(Qh, Kh) / math.sqrt(self.head_k)  # (B,H,Nf,Nr)
        attn = torch.softmax(scores, dim=-1)
        att = torch.matmul(attn, Vh)  # (B,H,Nf,dv/H)
        att = att.permute(0, 2, 1, 3).contiguous().view(B, Nf, -1)  # (B,Nf,d_v)

        att_out = self.transformer(att)  # (B,Nf,d_v)
        face_proj = self.face_to_dv(Xf_tokens)  # (B,Nf,d_v)
        fused = torch.cat([att_out, face_proj], dim=-1)  # (B,Nf,2*d_v)
        fused = self.fusion_proj(fused)  # (B,Nf,d_v)
        _log("TCMA_F2R/fused(B,Nf,d_v)", fused, self.debug)
        return fused


# ===================== #
#    新模块: 特征精炼    #
# ===================== #
class RefineBlock(nn.Module):
    """
    这是一个轻量级的通道注意力模块 (SE-style)。
    用于在 TCMA 融合后，对特征进行“二次筛选”和“精炼”。
    
    输入: (B, C, T)
    输出: (B, C, T)
    """
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1) # 全局池化 -> (B, C, 1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, t = x.size()
        # 1. 压缩空间信息
        y = self.avg_pool(x).view(b, c) 
        # 2. 计算通道权重
        y = self.fc(y).view(b, c, 1)
        # 3. 重新加权原始特征
        return x * y.expand_as(x)
# ===================== #
#    Multi-modal Net    #
# ===================== #
class MultiModalTCMAClassification(nn.Module):
    """
    你期望的整体结构：
    video -> ResNet -> FaceBlock1 -> TCMA_F2R -> FaceBlock2 -> pooling
    rppg -> RPPGBlock1 -> TCMA_R2F -> RPPGBlock2 -> pooling
    然后 concat -> 全连接分类
    """

    def __init__(self,
                 num_classes=4,
                 dropout=0.3,
                 debug=False):
        super().__init__()
        self.debug = debug

        # blocks
        self.face_block1 = FaceBlock1(in_channels=2048, dropout=dropout, debug=debug)
        self.face_block2 = FaceBlock2(in_channels=64, dropout=dropout, debug=debug)

        self.rppg_block1 = RPPGBlock1(in_channels=1, dropout=dropout, debug=debug)
        self.rppg_block2 = RPPGBlock2(in_channels=64, dropout=dropout, debug=debug)

        # TCMA
        self.tcma_r2f = TCMA_R2F(dim_face_token=64, dim_rppg_token=32,
                                 d_k=64, d_v=64, n_heads=4, post_layers=1, debug=debug)
        self.tcma_f2r = TCMA_F2R(dim_face_token=64, dim_rppg_token=32,
                                 d_k=64, d_v=64, n_heads=4, post_layers=1, debug=debug)
        
        # ==========================================
        #  在这里加入你的新模块
        # ==========================================
        # TCMA输出维度是 d_v=64，所以输入通道是 64
        self.refine_r = RefineBlock(in_channels=64, reduction=4)  # <--- 新增
        self.refine_f = RefineBlock(in_channels=64, reduction=4)  # <--- 新增

        # 最后的分类头：rPPG分支得到 128 维，Face 分支得到 128 维，共 256 -> 128 -> num_classes
        self.fc_head = nn.Sequential(
            nn.Linear(128 + 128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    def forward(self, rppg_seq, face_feats, mode='TCMA'):
        """
        mode: 'fusion' (TCCA), 'TCMA', 'face_only', 'rppg_only', 'concat_only'
        """
        B = rppg_seq.size(0)

        # ==========================================
        # 1. 基础特征提取 (所有模式通用)
        # ==========================================
        # ---- Face path ----
        face_feat1 = self.face_block1(face_feats)  # (B,64,T)
        face_tokens = face_feat1.permute(0, 2, 1)  # (B,T,64)

        # ---- rPPG path ----
        r_feat1 = self.rppg_block1(rppg_seq)       # (B,32,T)
        r_tokens = r_feat1.permute(0, 2, 1)        # (B,T,32)

        # ---- 时间维对齐 (所有模式通用) ----
        if face_tokens.size(1) != r_tokens.size(1):
            target_len = max(face_tokens.size(1), r_tokens.size(1))
            face_tokens = F.interpolate(
                face_tokens.permute(0, 2, 1), size=target_len, mode="linear", align_corners=False
            ).permute(0, 2, 1)
            r_tokens = F.interpolate(
                r_tokens.permute(0, 2, 1), size=target_len, mode="linear", align_corners=False
            ).permute(0, 2, 1)

        # ==========================================
        # 2. 分支逻辑 (消融实验核心)
        # ==========================================
        
        # --- 模式 A & B: 单模态 (Face Only / rPPG Only) ---
        if mode == 'face_only':
            # 跳过 TCMA，跳过 Refine
            # face_tokens: (B, T, 64) -> permute -> (B, 64, T)
            face_in = face_tokens.permute(0, 2, 1) 
            f_vec = self.face_block2(face_in) # (B, 128)
            
            # rPPG 部分全置 0
            zero_r = torch.zeros_like(f_vec).to(f_vec.device)
            fused = torch.cat([zero_r, f_vec], dim=-1)
            
        elif mode == 'rppg_only':
            # 跳过 TCMA，跳过 Refine
            # r_tokens: (B, T, 32)
            # 问题: RPPGBlock2 需要输入 64 通道，但这里只有 32
            # 解决: 在通道维拼接自己，凑成 64 (32+32)
            r_tokens_up = torch.cat([r_tokens, r_tokens], dim=2) # (B, T, 64)
            r_in = r_tokens_up.permute(0, 2, 1) # (B, 64, T)
            
            r_feat2 = self.rppg_block2(r_in)
            r_vec = F.adaptive_avg_pool1d(r_feat2, 1).view(B, -1) # (B, 128)
            
            # Face 部分全置 0
            zero_f = torch.zeros_like(r_vec).to(r_vec.device)
            fused = torch.cat([r_vec, zero_f], dim=-1)

        # --- 模式 C: 简单拼接 (Concat Only) ---
        elif mode == 'concat_only':
            # 1. Face 路 (无交互)
            face_in = face_tokens.permute(0, 2, 1) # (B, 64, T)
            f_vec = self.face_block2(face_in)      # (B, 128)
            
            # 2. rPPG 路 (无交互，需升维)
            r_tokens_up = torch.cat([r_tokens, r_tokens], dim=2) # (B, T, 64)
            r_in = r_tokens_up.permute(0, 2, 1)    # (B, 64, T)
            r_feat2 = self.rppg_block2(r_in)
            r_vec = F.adaptive_avg_pool1d(r_feat2, 1).view(B, -1) # (B, 128)
            
            # 3. 直接拼接
            fused = torch.cat([r_vec, f_vec], dim=-1)

        # --- 模式 D & E: 融合模式 (TCMA / TCCA) ---
        else: # mode == 'fusion' (TCCA) or mode == 'TCMA'
            # 1. 运行 TCMA 交互
            fused_r_tokens = self.tcma_r2f(face_tokens, r_tokens) # (B, T, 64)
            fused_f_tokens = self.tcma_f2r(face_tokens, r_tokens) # (B, T, 64)
            
            # 准备进入 Block2
            r_in = fused_r_tokens.permute(0, 2, 1) # (B, 64, T)
            f_in = fused_f_tokens.permute(0, 2, 1) # (B, 64, T)

            # 2. 是否运行 RefineBlock (TCCA 特有)
            if mode == 'fusion': 
                r_in = self.refine_r(r_in)
                f_in = self.refine_f(f_in)
            
            # 3. 后半段特征提取
            r_feat2 = self.rppg_block2(r_in)
            r_vec = F.adaptive_avg_pool1d(r_feat2, 1).view(B, -1)
            
            f_vec = self.face_block2(f_in)
            
            # 4. 拼接
            fused = torch.cat([r_vec, f_vec], dim=-1)

        # ==========================================
        # 3. 分类头 (所有模式通用)
        # ==========================================
        logits = self.fc_head(fused)
        return logits
        
        
# ===================== #
#  特征提取 / 推理类    #
# ===================== #
class VideoMultimodalFeatureExtractor:
    """
    读取：
      - rppg_pickle_path: 里面有 { 'predictions': {...}, 'labels': {...}, ...}
      - 视频：从 video_path 读取帧
    调用 MultiModalTCMAClassification 输出情绪分类
    """

    def __init__(self,
                 face_model_path: str,
                 rppg_pickle_path: str,
                 multimodal_model_path: str = None,
                 device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.face_model_path = face_model_path
        self.rppg_pickle_path = rppg_pickle_path

        # 加载 rPPG 数据
        self.rppg_data = self._load_rppg_data()

        # 加载 ResNet 特征提取器
        self.resnet = self._load_feature_extractor()

        # 加载多模态网络
        self.multimodal_model = self._load_multimodal_model(multimodal_model_path)

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        print(f"\n[Init] Multimodal feature extractor on {self.device}")
        print(f"[Init] Loaded rPPG data for {len(self.rppg_data.get('predictions', {}))} sessions\n")

    def _load_rppg_data(self) -> Dict[str, Any]:
        try:
            with open(self.rppg_pickle_path, "rb") as f:
                data = pickle.load(f)
            print(f"[rPPG] Loaded pickle: {self.rppg_pickle_path}")
            print(f"[rPPG] Keys: {list(data.keys())}")
            return data
        except Exception as e:
            print(f"[rPPG] Error loading pickle: {e}")
            return {"predictions": {}, "labels": {}}

    def _load_feature_extractor(self) -> nn.Module:
        """
        加载 ResNet50 backbone（你自己的face_feature_net.pth），
        输出为 2048 维向量。
        """
        model = models.resnet50(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 8)  # 你之前的分类头

        if os.path.exists(self.face_model_path):
            ckpt = torch.load(self.face_model_path, map_location='cpu')
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                state_dict = ckpt['model_state_dict']
            else:
                state_dict = ckpt

            new_state = {}
            for k, v in state_dict.items():
                if k.startswith('backbone.'):
                    k = k[len('backbone.'):]
                if k.startswith('module.'):
                    k = k[len('module.'):]
                new_state[k] = v

            model.load_state_dict(new_state, strict=False)
            print(f"[Face] Loaded feature extractor weights from {self.face_model_path}")
        else:
            print(f"[Face] Warning: face_model_path {self.face_model_path} not found, use random init.")

        # 去掉最后的 FC，保留 backbone 输出 (B,2048)
        model = nn.Sequential(*list(model.children())[:-1])  # (B,2048,1,1)
        model.eval()
        model.to(self.device)
        return model

    def _load_multimodal_model(self, model_path: str) -> nn.Module:
        # 1. 实例化你的新模型 (带 RefineBlock)
        model = MultiModalTCMAClassification(num_classes=4, dropout=0.3, debug=False)

        if model_path and os.path.exists(model_path):
            print(f"[MM] Loading weights from {model_path}...")
            
            # 加载 checkpoint
            ckpt = torch.load(model_path, map_location=self.device)
            state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt

            # =======================================================
            # 关键步骤：strict=False
            # =======================================================
            # 这告诉 PyTorch：
            # 1. 遇到能匹配的层 (TCMA, ResNet, etc.) -> 加载权重
            # 2. 遇到新模型里有但旧权重里没有的层 (refine_r, refine_f) -> 跳过，保持随机初始化
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

            print(f"[MM] Weights loaded with strict=False.")
            print(f"     > Missing keys (New layers initialized randomly): {len(missing_keys)}")
            
            # 打印出来确认一下，你应该会看到 refine_r 和 refine_f 在这里面
            for k in missing_keys:
                if 'refine' in k:
                    print(f"       - {k}")
            
        else:
            print(f"[MM] No weights found at {model_path}, using random init.")

        model.eval() # 或者 .train()
        model.to(self.device)
        return model

    # ---------- rPPG 部分 ----------
    def get_rppg_signal_for_session(self, session_id: str) -> Tuple[np.ndarray, bool]:
        preds = self.rppg_data.get('predictions', {})

        # 尝试多种 key
        possible_keys = [session_id, f"Session{session_id}", f"s{session_id}", f"session_{session_id}"]
        for k in possible_keys:
            if k in preds:
                pred_dict = preds[k]
                # pred_dict 可能是 {clip_id: np.array/torch.tensor}
                segs = []
                for seg_key in sorted(pred_dict.keys()):
                    sig = pred_dict[seg_key]
                    if isinstance(sig, torch.Tensor):
                        sig = sig.detach().cpu().numpy()
                    sig = np.reshape(sig, -1)
                    segs.append(sig)
                if segs:
                    sig_all = np.concatenate(segs)
                    return sig_all, True

        return np.array([]), False

    def align_rppg_with_video(self,
                              rppg_signal: np.ndarray,
                              video_frames: List[np.ndarray],
                              target_length: int = None) -> torch.Tensor:
        if target_length is None:
            target_length = len(video_frames)

        if target_length <= 0:
            return torch.zeros(1, 1, 0, dtype=torch.float32)

        if len(rppg_signal) == 0:
            aligned = np.zeros(target_length, dtype=np.float32)
        else:
            orig_idx = np.linspace(0, len(rppg_signal) - 1, len(rppg_signal))
            tgt_idx = np.linspace(0, len(rppg_signal) - 1, target_length)
            aligned = np.interp(tgt_idx, orig_idx, rppg_signal).astype(np.float32)

        if aligned.std() > 1e-6:
            aligned = (aligned - aligned.mean()) / aligned.std()

        return torch.from_numpy(aligned).view(1, 1, -1)  # (1,1,T)

    # ---------- 视频 / face 特征 ----------
    def extract_face_features(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        输入: list of BGR frames (H,W,3)
        输出: (1,T,2048)
        """
        feat_list = []
        for f in frames:
            try:
                img = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img)
                inp = self.transform(img_pil).unsqueeze(0).to(self.device)  # (1,3,224,224)
                with torch.no_grad():
                    feat = self.resnet(inp)  # (1,2048,1,1)
                    feat = feat.view(1, 2048)
                feat_list.append(feat)
            except Exception as e:
                print(f"[Face] Error on frame, use zero vector. Err={e}")
                feat_list.append(torch.zeros(1, 2048, device=self.device))

        if len(feat_list) == 0:
            return torch.zeros(1, 0, 2048, device=self.device)

        # (T,1,2048) -> (1,T,2048)
        feats = torch.stack(feat_list, dim=0)  # (T,1,2048)
        feats = feats.transpose(0, 1).contiguous()  # (1,T,2048)
        return feats

    # ---------- 主流程 ----------
    def process_video_with_rppg(self,
                                video_path: str,
                                session_id: str,
                                target_frames: int = 300) -> Dict[str, Any]:
        print(f"\n[Process] video={video_path}")
        print(f"[Process] session_id={session_id}")

        rppg_signal, found_rppg = self.get_rppg_signal_for_session(session_id)
        print(f"[Process] rPPG found={found_rppg}, len={len(rppg_signal)}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration = total_frames / fps if fps > 0 else 0.0

        print(f"[Video] total_frames={total_frames}, fps={fps:.2f}, duration={duration:.2f}s")

        if total_frames <= target_frames:
            indices = list(range(total_frames))
        else:
            step = max(total_frames // target_frames, 1)
            indices = list(range(0, total_frames, step))[:target_frames]

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                print(f"[Video] Warning: cannot read frame {idx}")
                continue
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            raise RuntimeError("No frames read from video.")

        print(f"[Video] sampled_frames={len(frames)}")

        # 对齐 rPPG
        rppg_seq = self.align_rppg_with_video(rppg_signal, frames, target_length=len(frames)).to(self.device)
        print(f"[rPPG] aligned shape={tuple(rppg_seq.shape)}")

        # 提取 face features
        face_feats = self.extract_face_features(frames).to(self.device)
        print(f"[Face] feature sequence shape={tuple(face_feats.shape)}")

        # multimodal forward
        with torch.no_grad():
            logits = self.multimodal_model(rppg_seq, face_feats)  # (1,num_classes)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        emotion_classes = ['LV-LA (sad)', 'LV-HA (anger)', 'HV-LA (calm)', 'HV-HA (happy)']
        pred_idx = int(np.argmax(probs))
        pred_emotion = emotion_classes[pred_idx]
        conf = float(probs[pred_idx])

        print(f"[Result] Pred Emotion: {pred_emotion}, Confidence: {conf:.4f}")

        return {
            "logits": logits.cpu().numpy(),
            "probs": probs,
            "pred_class": pred_idx,
            "pred_emotion": pred_emotion,
            "confidence": conf,
            "video_info": {
                "session_id": session_id,
                "total_frames": total_frames,
                "sampled_frames": len(frames),
                "fps": fps,
                "duration": duration,
            },
            "rppg_info": {
                "found": found_rppg,
                "original_len": len(rppg_signal),
            },
        }

'''
    def forward(self, rppg_seq, face_feats, mode='face_only'):
        """
        rppg_seq:  (B,1,T_r)
        face_feats:(B,T_f,2048)
        """
        _log("INPUT/rppg_seq(B,1,T_r)", rppg_seq, self.debug)
        _log("INPUT/face_feats(B,T_f,2048)", face_feats, self.debug)
        B = rppg_seq.size(0)

        # ---- Face path ----
        face_feat1 = self.face_block1(face_feats)  # (B,64,T_f1)
        face_tokens = face_feat1.permute(0, 2, 1)  # (B,T_f1,64)
        _log("MAIN/face_tokens(B,T_f1,64)", face_tokens, self.debug)

        # ---- rPPG path ----
        r_feat1 = self.rppg_block1(rppg_seq)  # (B,32,T_r1)
        r_tokens = r_feat1.permute(0, 2, 1)  # (B,T_r1,32)
        _log("MAIN/r_tokens(B,T_r1,32)", r_tokens, self.debug)

        # ---- 时间维对齐 ----
        if face_tokens.size(1) != r_tokens.size(1):
            target_len = max(face_tokens.size(1), r_tokens.size(1))
            # 对 time 维线性插值
            face_tokens = F.interpolate(
                face_tokens.permute(0, 2, 1), size=target_len, mode="linear", align_corners=False
            ).permute(0, 2, 1)
            r_tokens = F.interpolate(
                r_tokens.permute(0, 2, 1), size=target_len, mode="linear", align_corners=False
            ).permute(0, 2, 1)

        _log("MAIN/face_tokens_aligned(B,T,64)", face_tokens, self.debug)
        _log("MAIN/r_tokens_aligned(B,T,32)", r_tokens, self.debug)

        # ---- TCMA ----
        fused_r_tokens = self.tcma_r2f(face_tokens, r_tokens)  # (B,T,64)
        fused_f_tokens = self.tcma_f2r(face_tokens, r_tokens)  # (B,T,64)
        _log("MAIN/fused_r_tokens(B,T,64)", fused_r_tokens, self.debug)
        _log("MAIN/fused_f_tokens(B,T,64)", fused_f_tokens, self.debug)

        # # ---- rPPG 分支后半段 ----
        # rppg_block2_in = fused_r_tokens.permute(0, 2, 1)  # (B,64,T)
        # r_feat2 = self.rppg_block2(rppg_block2_in)  # (B,128,T2)
        # r_vec = F.adaptive_avg_pool1d(r_feat2, 1).view(B, -1)  # (B,128)
        # _log("MAIN/r_vec(B,128)", r_vec, self.debug)

        # # ---- Face 分支后半段 ----
        # face_block2_in = fused_f_tokens.permute(0, 2, 1)  # (B,64,T)
        # f_vec = self.face_block2(face_block2_in)  # (B,128)
        # _log("MAIN/f_vec(B,128)", f_vec, self.debug)

        # ==========================================
        #  使用新模块的位置
        # ==========================================
        
        # 1. rPPG 分支后半段
        # 先变回 (B, C, T) 格式，因为 RefineBlock 和 Conv1d 都喜欢 (B, C, T)
        rppg_block2_in = fused_r_tokens.permute(0, 2, 1)  # (B,64,T)
        # ---> 加入精炼 <---
        if mode == 'fusion':
            rppg_block2_in = self.refine_r(rppg_block2_in)    # (B,64,T)  <--- 新增调用
        _log("MAIN/rppg_refined", rppg_block2_in, self.debug)

        r_feat2 = self.rppg_block2(rppg_block2_in)        # (B,128,T2)
        r_vec = F.adaptive_avg_pool1d(r_feat2, 1).view(B, -1)

        # 2. Face 分支后半段
        face_block2_in = fused_f_tokens.permute(0, 2, 1)  # (B,64,T)
        # ---> 加入精炼 <---
        if mode == 'fusion':
            face_block2_in = self.refine_f(face_block2_in)    # (B,64,T)  <--- 新增调用
        _log("MAIN/face_refined", face_block2_in, self.debug)

        f_vec = self.face_block2(face_block2_in)          # (B,128)
        _log("MAIN/f_vec(B,128)", f_vec, self.debug)

        # 消融实验
        if mode == 'fusion' or mode == 'TCMA':
            # 你的 TCCA 完整逻辑
            fused = torch.cat([r_vec, f_vec], dim=-1) # (B, 256)
            logits = self.fc_head(fused)
            
        elif mode == 'face_only':
            # print("face_only......")
            # 只用 Face，这就需要你临时改一下 fc_head 或者在这里临时定义一个
            # 为了省事，我们可以把 rPPG 部分全置为 0，假装它不存在
            zero_r = torch.zeros_like(r_vec).to(r_vec.device)
            fused = torch.cat([zero_r, f_vec], dim=-1)
            logits = self.fc_head(fused)
            
        elif mode == 'rppg_only':
            # print("rppg_only......")
            # 只用 rPPG，把 Face 置为 0
            zero_f = torch.zeros_like(f_vec).to(f_vec.device)
            fused = torch.cat([r_vec, zero_f], dim=-1)
            logits = self.fc_head(fused)

        elif mode == 'concat_only':
            # 1. 不做 TCMA，也不做 Refine
            # 2. 直接把 face_tokens (B, T, 64) -> Pooling -> (B, 128)
            #    需要你调用一下 self.face_block2
            
            # 注意：这里需要跳过 TCMA 层
            # Face路: face_block1 -> face_block2 -> pooling
            # rPPG路: rppg_block1 -> rppg_block2 -> pooling
            
            # ... 这里你需要写一点代码跳过中间层 ...
            # 比如直接把 face_tokens 传给 face_block2 (维度要对上)
            pass

        # # ---- 融合 + 分类 ----
        # fused = torch.cat([r_vec, f_vec], dim=-1)  # (B,256)
        # logits = self.fc_head(fused)  # (B,num_classes)
        # _log("OUTPUT/logits(B,num_classes)", logits, self.debug)

        return logits
'''

def extract_numeric_session_id(key: str) -> str:
    """
    从 key 中提取数字，例如:
    "10" -> "10"
    "Session10" -> "10"
    "s10" -> "10"
    "session_10" -> "10"
    """
    import re
    nums = re.findall(r'\d+', key)
    if len(nums) == 0:
        return None
    return nums[-1]  # 最后一段数字通常就是 ID


def find_avi_in_session(session_root: str) -> str:
    """
    在 /dataset/MAHNOB-HCI/Sessions/{id}/ 下寻找 avi 文件
    """
    import glob
    avi_files = glob.glob(os.path.join(session_root, "*.avi"))
    if len(avi_files) == 0:
        return None
    return avi_files[0]  # 默认取第 1 个文件


def main():
    face_model_path = "./weight/face_feature_net.pth"
    rppg_pickle_path = "./accurate_pickle/accurate.pickle"
    multimodal_model_path = None

    # MAHNOB sessions 根目录
    sessions_base = "/dataset/MAHNOB-HCI/Sessions"

    extractor = VideoMultimodalFeatureExtractor(
        face_model_path=face_model_path,
        rppg_pickle_path=rppg_pickle_path,
        multimodal_model_path=multimodal_model_path,
        device="cuda"
    )

    # =============================
    #  Step 1: 取 pickle 中有哪些 session key
    # =============================
    pred_dict = extractor.rppg_data.get("predictions", {})
    session_keys = list(pred_dict.keys())

    if len(session_keys) == 0:
        print("[Error] No sessions found in pickle['predictions']")
        return

    print(f"[INFO] Found {len(session_keys)} session keys in pickle")
    print("Sample keys:", session_keys[:10])

    # =============================
    #  Step 2: 提取 session ID (数字)
    # =============================
    session_ids = []
    for k in session_keys:
        sid = extract_numeric_session_id(k)
        if sid is not None:
            session_ids.append(sid)

    session_ids = sorted(list(set(session_ids)), key=lambda x: int(x))
    print(f"[INFO] Parsed {len(session_ids)} valid numeric session IDs:")
    print(session_ids)

    # =============================
    #  Step 3: 遍历所有 ID, 自动找到 avi 文件并处理
    # =============================

    for sid in session_ids:
        print("\n" + "=" * 60)
        print(f"[PROCESS] Session ID = {sid}")

        session_root = os.path.join(sessions_base, sid)
        if not os.path.isdir(session_root):
            print(f"[WARN] Directory not found: {session_root}")
            continue

        avi_path = find_avi_in_session(session_root)
        if avi_path is None:
            print(f"[WARN] No avi found in {session_root}")
            continue

        print(f"[INFO] Using video: {avi_path}")

        # 调用你的推理流程
        try:
            result = extractor.process_video_with_rppg(avi_path, sid)
            print(f"[RESULT] {sid}: {result['pred_emotion']}  (conf={result['confidence']:.3f})")
        except Exception as e:
            print(f"[ERROR] Failed to process session {sid}: {e}")
            continue


if __name__ == "__main__":
    main()
