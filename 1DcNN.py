import math, torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- rPPG blocks (unchanged) ----------
class RPPGBlock1(nn.Module):
    def __init__(self, in_channels=1, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=20, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(16)
        self.drop1 = nn.Dropout(dropout)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=10, stride=1, padding=0)
        self.bn2 = nn.BatchNorm1d(32)
        self.drop2 = nn.Dropout(dropout)
        self.pool2 = nn.MaxPool1d(2)

    def forward(self, x):
        x = self.conv1(x);
        x = self.bn1(x);
        x = F.relu(x);
        x = self.drop1(x);
        x = self.pool1(x)
        x = self.conv2(x);
        x = self.bn2(x);
        x = F.relu(x);
        x = self.drop2(x);
        x = self.pool2(x)
        return x  # (B,32,T1)


class RPPGBlock2(nn.Module):
    def __init__(self, in_channels=32, dropout=0.3):
        super().__init__()
        # 第一层卷积 + BN + ReLU + Dropout + MaxPool
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, stride=1, padding=1)  # padding=1 保持长度
        self.bn1 = nn.BatchNorm1d(64)
        self.drop1 = nn.Dropout(dropout)
        self.pool1 = nn.MaxPool1d(2)

        # 第二层卷积 + BN + ReLU + Dropout + MaxPool
        self.conv2 = nn.Conv1d(64, 128, kernel_size=10, stride=1, padding=0)  # 不填充
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(dropout)
        self.pool2 = nn.MaxPool1d(2)

    def forward(self, x):
        # 第一层
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.drop1(x)
        x = self.pool1(x)

        # 第二层
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.drop2(x)
        x = self.pool2(x)

        return x  # (B, 128, T2)


# ---------- Face temporal 1D-CNN blocks ----------
class FaceBlock1(nn.Module):
    def __init__(self, in_channels=2048, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(256, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, C) -> (B, C, T)
        if x.dim() == 3:
            x = x.permute(0, 2, 1).contiguous()
        x = self.conv1(x);
        x = self.bn1(x);
        x = F.relu(x);
        x = self.drop1(x)
        x = self.conv2(x);
        x = self.bn2(x);
        x = F.relu(x);
        x = self.drop2(x)
        return x  # (B,64,T)


class FaceBlock2(nn.Module):
    def __init__(self, in_channels=64, dropout=0.3, out_dim=1, use_classification=False, hidden=64):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(dropout)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.use_classification = use_classification
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        x = self.conv1(x);
        x = self.bn1(x);
        x = F.relu(x);
        x = self.drop1(x)
        x = self.global_pool(x)
        out = self.fc(x)
        return out, x  # prediction and pooled feature


# ---------- TCMA ----------
class TCMA_R2F(nn.Module):
    def __init__(self, dim_face_token, dim_rppg_token, d_k=128, d_v=128, n_heads=4, post_layers=1):
        super().__init__()
        assert d_k % n_heads == 0 and d_v % n_heads == 0
        self.n_heads = n_heads
        self.head_k = d_k // n_heads
        self.head_v = d_v // n_heads

        self.Wq_r = nn.Linear(dim_rppg_token, d_k)
        self.Wk_f = nn.Linear(dim_face_token, d_k)
        self.Wv_f = nn.Linear(dim_face_token, d_v)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_v, nhead=n_heads, dim_feedforward=4 * d_v,
                                                   dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=post_layers)
        self.rppg_to_dv = nn.Linear(dim_rppg_token, d_v)
        self.fusion_proj = nn.Linear(d_v + d_v, d_v)

    def forward(self, Xf_tokens, Xr_tokens):
        B, Nf, Cf = Xf_tokens.shape
        B2, Nr, Cr = Xr_tokens.shape
        assert B == B2

        K = self.Wk_f(Xf_tokens)
        V = self.Wv_f(Xf_tokens)
        Q = self.Wq_r(Xr_tokens)

        Qh = Q.view(B, Nr, self.n_heads, self.head_k).permute(0, 2, 1, 3)
        Kh = K.view(B, Nf, self.n_heads, self.head_k).permute(0, 2, 3, 1)
        Vh = V.view(B, Nf, self.n_heads, self.head_v).permute(0, 2, 1, 3)

        scores = torch.matmul(Qh, Kh) / math.sqrt(self.head_k)
        attn = torch.softmax(scores, dim=-1)
        att = torch.matmul(attn, Vh)
        att = att.permute(0, 2, 1, 3).contiguous().view(B, Nr, -1)

        att_out = self.transformer(att)
        rppg_proj = self.rppg_to_dv(Xr_tokens)
        fused = torch.cat([att_out, rppg_proj], dim=-1)
        fused = self.fusion_proj(fused)
        return fused


class TCMA_F2R(nn.Module):
    """
    Face → rPPG cross-modal attention
    Input:
        Xf_tokens: (B, Nf, Cf)
        Xr_tokens: (B, Nr, Cr)
    Output:
        fused: (B, Nf, d_v)
    """

    def __init__(self, dim_face_token, dim_rppg_token, d_k=128, d_v=128, n_heads=4, post_layers=1):
        super().__init__()
        assert d_k % n_heads == 0 and d_v % n_heads == 0
        self.n_heads = n_heads
        self.head_k = d_k // n_heads
        self.head_v = d_v // n_heads

        # 线性映射
        self.Wq_f = nn.Linear(dim_face_token, d_k)  # Face → Query
        self.Wk_r = nn.Linear(dim_rppg_token, d_k)  # rPPG → Key
        self.Wv_r = nn.Linear(dim_rppg_token, d_v)  # rPPG → Value

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_v, nhead=n_heads,
                                                   dim_feedforward=4 * d_v, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=post_layers)

        # 投影融合
        self.face_to_dv = nn.Linear(dim_face_token, d_v)
        self.fusion_proj = nn.Linear(d_v + d_v, d_v)

    def forward(self, Xf_tokens, Xr_tokens):
        B, Nf, Cf = Xf_tokens.shape
        B2, Nr, Cr = Xr_tokens.shape
        assert B == B2

        # Linear projection
        Q = self.Wq_f(Xf_tokens)  # (B, Nf, d_k)
        K = self.Wk_r(Xr_tokens)  # (B, Nr, d_k)
        V = self.Wv_r(Xr_tokens)  # (B, Nr, d_v)

        # 多头注意力
        Qh = Q.view(B, Nf, self.n_heads, self.head_k).permute(0, 2, 1, 3)  # (B, n_heads, Nf, head_k)
        Kh = K.view(B, Nr, self.n_heads, self.head_k).permute(0, 2, 3, 1)  # (B, n_heads, head_k, Nr)
        Vh = V.view(B, Nr, self.n_heads, self.head_v).permute(0, 2, 1, 3)  # (B, n_heads, Nr, head_v)

        scores = torch.matmul(Qh, Kh) / math.sqrt(self.head_k)
        attn = torch.softmax(scores, dim=-1)
        att = torch.matmul(attn, Vh)  # (B, n_heads, Nf, head_v)

        # 合并 heads
        att = att.permute(0, 2, 1, 3).contiguous().view(B, Nf, -1)  # (B, Nf, d_v)

        # Transformer Encoder
        att_out = self.transformer(att)

        # 融合原始 face 投影
        face_proj = self.face_to_dv(Xf_tokens)
        fused = torch.cat([att_out, face_proj], dim=-1)
        fused = self.fusion_proj(fused)  # (B, Nf, d_v)

        return fused


# ---------- Full multimodal network with feature input ----------
class MultiModalTCMA_v4_Classification(nn.Module):
    def __init__(self, rppg_in_ch=1, rppg_token_dim=64, tcma_dk=128, tcma_dv=128, tcma_heads=4,
                 dropout=0.3, num_classes=4):
        super().__init__()
        # ---------- Face blocks ----------
        self.face_block1 = FaceBlock1(in_channels=2048, dropout=dropout)
        self.face_block2 = FaceBlock2(in_channels=64, dropout=dropout, out_dim=64)  # 输出池化特征64

        # ---------- rPPG blocks ----------
        self.r_block1 = RPPGBlock1(in_channels=rppg_in_ch, dropout=dropout)
        self.r_block2 = RPPGBlock2(in_channels=32, dropout=dropout, out_dim=128)  # 输出池化特征128
        self.rppg_token_proj = nn.Linear(32, rppg_token_dim)

        # ---------- TCMA 双向 ----------
        self.tcma_r2f = TCMA_R2F(dim_face_token=64, dim_rppg_token=rppg_token_dim,
                                 d_k=tcma_dk, d_v=tcma_dv, n_heads=tcma_heads, post_layers=1)  # rPPG→Face
        self.tcma_f2r = TCMA_F2R(dim_face_token=64, dim_rppg_token=rppg_token_dim,
                                 d_k=tcma_dk, d_v=tcma_dv, n_heads=tcma_heads, post_layers=1)  # Face→rPPG

        # ---------- final head ----------
        # face + rPPG 融合特征拼接后FC
        self.fc_head = nn.Sequential(
            nn.Linear(64 + 128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, rppg_seq, face_feats):
        """
        rppg_seq: (B,1,T)
        face_feats: (B,T,2048)
        """
        B = rppg_seq.size(0)

        # ---------- face path ----------
        face_block1_out = self.face_block1(face_feats)  # (B,64,T)
        face_tokens = face_block1_out.permute(0, 2, 1).contiguous()  # (B,T,64)

        # ---------- rPPG path ----------
        r_feat = self.r_block1(rppg_seq)  # (B,32,T1)
        r_tokens = r_feat.permute(0, 2, 1).contiguous()  # (B,T1,32)
        r_tokens = self.rppg_token_proj(r_tokens)  # (B,T1,rppg_token_dim)

        # ---------- temporal alignment ----------
        if r_tokens.size(1) != face_tokens.size(1):
            target_len = max(r_tokens.size(1), face_tokens.size(1))
            r_tokens = F.interpolate(r_tokens.permute(0, 2, 1), size=target_len, mode='linear',
                                     align_corners=False).permute(0, 2, 1)
            face_tokens = F.interpolate(face_tokens.permute(0, 2, 1), size=target_len, mode='linear',
                                        align_corners=False).permute(0, 2, 1)

        # ---------- TCMA 双向 ----------
        fused_r2f = self.tcma_r2f(face_tokens, r_tokens)  # (B,T,d_v)
        fused_f2r = self.tcma_f2r(face_tokens, r_tokens)  # (B,T,d_v)

        # ---------- RPPG post-fusion ----------
        r_tokens_post = fused_f2r.permute(0, 2, 1).contiguous()  # (B,d_v,T)
        r_feat2 = self.r_block2(r_tokens_post)  # (B,128,T2)
        r_pooled = F.adaptive_avg_pool1d(r_feat2, 1).view(B, -1)  # (B,128)

        # ---------- Face post-fusion ----------
        face_feat2 = self.face_block2(face_block1_out)[1]  # 取池化特征 (B,64,1)
        face_pooled = face_feat2.view(B, -1)  # (B,64)

        # ---------- 融合 + 分类 ----------
        fused_feat = torch.cat([r_pooled, face_pooled], dim=-1)  # (B,128+64)
        out = self.fc_head(fused_feat)  # (B,4)

        return out
