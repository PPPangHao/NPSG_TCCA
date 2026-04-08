# 统计分析session中的标签分布
import os
import glob
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# ================= 配置区域 =================
CONFIG = {
    # 数据集 Session 根目录
    "SESSIONS_ROOT": "/dataset/MAHNOB-HCI/Sessions",

    # 阈值设置 (请根据你的具体定义修改)
    "RESTING_THRESH": 3,      # Arousal <= 3 为静息
    "VALENCE_SPLIT": 4.5,     # Valence < 6.5 为 Low, >= 6.5 为 High
    "AROUSAL_HIGH_THRESH": 6, # Arousal >= 6 为 High, (3, 6) 为 Mid

    # 图片保存路径
    "SAVE_PATH": "./class_distribution.png"
}
# ===========================================

def parse_session_xml(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        # 获取情感标签，如果缺失默认为 -1
        valence = int(root.attrib.get('feltVlnc', -1))
        arousal = int(root.attrib.get('feltArsl', -1))
        return valence, arousal
    except Exception as e:
        return None, None

def determine_class(v, a):
    """
    根据五分类规则返回类别名称
    """
    # 1. 过滤无效数据
    if v == -1 or a == -1:
        return None

    # 2. 第一层：静息判定
    if a <= CONFIG["RESTING_THRESH"]:
        return "Resting"

    # 3. 第二层：激活状态细分
    # 判定 Valence
    v_label = "HV" if v >= CONFIG["VALENCE_SPLIT"] else "LV"

    # 判定 Arousal 强度 (Mid vs High)
    a_label = "HA" if a >= CONFIG["AROUSAL_HIGH_THRESH"] else "MA"

    return f"{v_label}-{a_label}"

def main():
    # 1. 扫描文件
    xml_files = glob.glob(os.path.join(CONFIG["SESSIONS_ROOT"], "*", "session.xml"))
    print(f"Found {len(xml_files)} session XML files.")

    data = []

    # 2. 统计标签
    for xml_file in xml_files:
        v, a = parse_session_xml(xml_file)
        cls_name = determine_class(v, a)
        if cls_name:
            data.append(cls_name)

    # 3. 转换为 DataFrame
    df = pd.DataFrame(data, columns=["Class"])

    # 定义类别顺序 (让图表更整齐)
    order = ["Resting", "LV-MA", "HV-MA", "LV-HA", "HV-HA"]

    # 统计数量和比例
    counts = df["Class"].value_counts().reindex(order).fillna(0)
    total = sum(counts)

    print("\n=== Class Distribution Stats ===")
    print(counts)
    print(f"Total Samples: {total}")

    # 4. 绘图 (学术风格)
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.figure(figsize=(10, 6))

    # 使用一般学术论文喜欢的配色 (比如蓝色系或用于强调差异的颜色)
    # 这里的 palette 可以把 Resting 设为灰色，其他设为彩色，突出“激活”
    colors = {"Resting": "#95a5a6", "LV-MA": "#3498db", "HV-MA": "#2ecc71",
              "LV-HA": "#e74c3c", "HV-HA": "#9b59b6"}

    ax = sns.countplot(x="Class", data=df, order=order, palette=colors, edgecolor="black")

    plt.title("Sample Distribution across 5 Emotion Classes", fontsize=16, pad=20)
    plt.xlabel("Emotion Class", fontsize=14)
    plt.ylabel("Number of Sessions", fontsize=14)

    # 在柱子上添加具体的数值和百分比
    for p in ax.patches:
        height = int(p.get_height())
        if height > 0:
            percentage = '{:.1f}%'.format(100 * height / total)
            ax.annotate(f'{height}\n({percentage})',
                        (p.get_x() + p.get_width() / 2., height),
                        ha = 'center', va = 'center',
                        xytext = (0, 15),
                        textcoords = 'offset points',
                        fontsize=11, color='black', fontweight='bold')

    # 设置 y 轴上限，留出空间给文字
    plt.ylim(0, max(counts) * 1.15)

    plt.tight_layout()
    plt.savefig(CONFIG["SAVE_PATH"], dpi=300)
    print(f"\nFigure saved to {CONFIG['SAVE_PATH']}")
    plt.show()

if __name__ == "__main__":
    main()
