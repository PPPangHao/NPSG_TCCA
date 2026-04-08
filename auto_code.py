import pickle
import numpy as np
from sklearn.metrics import accuracy_score

# --- 1. 配置 ---
RESULTS_PICKLE_PATH = './output/final_four_class_predictions.pkl'
SEARCH_RANGE = np.linspace(0.0, 2.0, 201) # 从 0.0 到 2.0 搜索 201 个点

# --- 2. 加载数据 ---
try:
    with open(RESULTS_PICKLE_PATH, 'rb') as f:
        results = pickle.load(f)
except FileNotFoundError:
    print(f"错误：未找到文件 {RESULTS_PICKLE_PATH}。请确保文件路径正确。")
    exit()

# --- 3. 提取特征和真值 ---
# stm_intensity: 对应于 I_A(t) 的平均累积强度（STM 模块的输出）
# gt_arousal_binary: Arousal 的二分类真值 (0: LA, 1: HA)
intensities = []
gt_arousal = []

for sid, data in results.items():
    # 确保数据完整，跳过可能缺失真值的样本
    if data['gt_binary'] and data['stm_intensity'] is not None:
        intensities.append(data['stm_intensity'])
        # gt_binary 是 (arousal, valence) 的元组，我们取第一个元素
        gt_arousal.append(data['gt_binary'][0]) 

intensities = np.array(intensities)
gt_arousal = np.array(gt_arousal)

if len(intensities) == 0:
    print("错误：未从结果文件中提取到有效数据。")
    exit()

print(f"成功提取 {len(intensities)} 个样本的 STM 强度和 Arousal 真值。")
print(f"初始 Arousal 准确率 (假设阈值 0.5): {accuracy_score(gt_arousal, (intensities > 0.5).astype(int)):.4f}")

# --- 4. 搜索最佳阈值 ---
best_threshold = 0.0
max_accuracy = 0.0
accuracies = {}

for threshold in SEARCH_RANGE:
    # STM 决策规则：强度 > 阈值 -> High Arousal (1)
    predicted_arousal = (intensities > threshold).astype(int)
    
    # 计算当前阈值下的准确率
    acc = accuracy_score(gt_arousal, predicted_arousal)
    accuracies[threshold] = acc
    
    if acc > max_accuracy:
        max_accuracy = acc
        best_threshold = threshold

# --- 5. 输出结果 ---
print("\n--- Arousal STM 最佳阈值搜索结果 ---")
print(f"原始阈值 (0.5) 准确率: {accuracies.get(0.5, 0.0):.4f}")
print(f"最佳 $\\theta_{{POOL\\_A}}^*$ 阈值: {best_threshold:.4f}")
print(f"最高 Arousal 准确率: {max_accuracy:.4f}")
print("------------------------------------------")

# --- 6. (可选) 可视化以进行人工确认 ---
# 这一步需要 matplotlib，如果您需要可视化，请安装该库。
# 它可以显示 STM 强度分布和最佳分割点。

#
