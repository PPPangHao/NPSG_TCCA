# -*- coding: gbk -*-
import numpy as np
import matplotlib.pyplot as plt

# ======================
# 数据（来自你的表格）
# ======================
classes = [
    "Class 0\n(Low/Resting)",
    "Class 1\n(High/Active)",
    "Macro Avg"
]

metrics = ["Precision", "Recall", "F1-score"]

values = np.array([
    [0.46, 0.25, 0.32],
    [0.67, 0.84, 0.74],
    [0.56, 0.54, 0.53]
])

# ======================
# 绘制热力图
# ======================
fig, ax = plt.subplots(figsize=(8, 4))

im = ax.imshow(values)

# 坐标轴标签
ax.set_xticks(np.arange(len(metrics)))
ax.set_yticks(np.arange(len(classes)))
ax.set_xticklabels(metrics)
ax.set_yticklabels(classes)

# 数值标注
for i in range(values.shape[0]):
    for j in range(values.shape[1]):
        ax.text(
            j, i,
            f"{values[i, j]:.2f}",
            ha="center",
            va="center",
            color="black",
            fontsize=10,
            fontweight="bold" if i == 1 else "normal"  # 高亮 High/Active
        )

# 标题
ax.set_title(
    "Stage-1 Classification Performance Heatmap\n"
    "High/Active class shows dominant Recall (0.84)",
    pad=12
)

# 颜色条
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Score Value")

# 布局与保存
plt.tight_layout()
plt.savefig("stage1_metrics_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()
