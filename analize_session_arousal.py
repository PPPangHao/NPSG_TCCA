import os
import xml.etree.ElementTree as ET
import glob
import matplotlib.pyplot as plt

def get_arousal_from_xml(xml_path):
    """
    从 session.xml 文件中解析 feltArsl 标签值
    :param xml_path: session.xml 文件路径
    :return: Arousal (int), 如果没有找到标签，返回 -1
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        arousal = int(root.attrib.get("feltArsl", -1))  # 默认为-1，表示没有找到
        return arousal
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return -1

def analyze_arousal_distribution(sessions_root):
    """
    遍历所有 session.xml 文件，统计每个 session 中的 Arousal 分布
    :param sessions_root: Sessions 数据目录路径
    :return: Arousal 分布统计
    """
    arousal_values = []
    
    # 查找所有 session.xml 文件
    session_dirs = glob.glob(os.path.join(sessions_root, "*"))
    
    for session_dir in session_dirs:
        if os.path.isdir(session_dir):
            xml_path = os.path.join(session_dir, "session.xml")
            if os.path.exists(xml_path):
                arousal = get_arousal_from_xml(xml_path)
                if arousal != -1:  # 只有有效的 Arousal 值才统计
                    arousal_values.append(arousal)

    return arousal_values

def plot_arousal_distribution(arousal_values):
    """
    绘制 Arousal 分布图
    :param arousal_values: Arousal 数据列表
    """
    if not arousal_values:
        print("No Arousal data available.")
        return

    plt.figure(figsize=(8, 6))
    plt.hist(arousal_values, bins=9, range=(1, 9), edgecolor='black', alpha=0.7)
    plt.xlabel("Arousal")
    plt.ylabel("Frequency")
    plt.title("Arousal Distribution")
    plt.xticks(range(1, 10))  # Arousal score range from 1 to 9
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # 设置你的 session 数据目录路径
    SESSIONS_ROOT = "/dataset/MAHNOB-HCI/Sessions"

    # 分析 Arousal 分布
    arousal_values = analyze_arousal_distribution(SESSIONS_ROOT)

    # 统计结果
    if arousal_values:
        print(f"Arousal values: {arousal_values}")
        print(f"Total sessions: {len(arousal_values)}")
        print(f"Unique Arousal values: {sorted(set(arousal_values))}")
    else:
        print("No Arousal data found.")
    
    # 绘制分布图
    plot_arousal_distribution(arousal_values)

