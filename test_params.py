# pip install thop
from thop import profile
# from deepseek_1DcNN_gpt_no_refine import MultiModalTCMAClassification
from deepseek_1DcNN_gpt import MultiModalTCMAClassification
import torch
import torch
from thop import profile

# 1. 准备设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 准备 Dummy Input (并移动到 device)
# 注意：根据你的模型定义，face_feats 输入形状应该是 (B, T, 2048) 或 (B, 2048, T)
# 你的FaceBlock1里有 permute 处理，所以这里假设输入是 (1, 300, 2048)
dummy_rppg = torch.randn(1, 1, 300).to(device)
dummy_face = torch.randn(1, 300, 2048).to(device)

# 3. 实例化模型
model = MultiModalTCMAClassification(num_classes=4, dropout=0.3)

# 4. 【关键步骤】把模型移动到 device
model = model.to(device)  # <--- 加上这一行！！！

# 5. 运行 thop
# thop 可能会打印很多日志，这行代码能计算 FLOPs 和 Params
flops, params = profile(model, inputs=(dummy_rppg, dummy_face))

print(f"FLOPs: {flops / 1e9:.3f} G")  # 转换为 G (10^9)
print(f"Params: {params / 1e6:.3f} M") # 转换为 M (10^6)