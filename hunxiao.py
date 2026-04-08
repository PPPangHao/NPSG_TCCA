import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import numpy as np

# 1. 加载结果
with open("./output/final_four_class_predictions.pkl", "rb") as f:
    results = pickle.load(f)

y_true = []
y_pred = []

for sid, data in results.items():
    y_true.append(data['gt_four'])
    y_pred.append(data['pred_four'])

# 2. 检查一下到底少了哪个类 (调试用)
unique_true = sorted(list(set(y_true)))
unique_pred = sorted(list(set(y_pred)))
print(f"真实标签包含的类别: {unique_true}")
print(f"预测标签包含的类别: {unique_pred}")

# 3. 定义标签
# 确保顺序对应：0, 1, 2, 3
class_names = ['LA-LV', 'LA-HV', 'HA-LV', 'HA-HV']
all_labels = [0, 1, 2, 3] # <--- 关键修正：显式定义所有可能的标签ID

# 4. 打印分类报告 (加入 labels 参数)
print("\nClassification Report:")
# zero_division=0 防止除以0报错，labels参数强制评估所有4个类
print(classification_report(y_true, y_pred, target_names=class_names, labels=all_labels, zero_division=0))

# 5. 绘制混淆矩阵 (加入 labels 参数)
cm = confusion_matrix(y_true, y_pred, labels=all_labels)

# 归一化矩阵 (显示百分比)
# 加上 1e-10 防止某一行全为0导致除以0
cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (4-Class Emotion Recognition)')
plt.show()

# 这一步生成图片后，如果发现某一列全是0，说明那个类别样本太少或模型没学到，
# 这正是您需要在论文中讨论的“数据不平衡”问题。