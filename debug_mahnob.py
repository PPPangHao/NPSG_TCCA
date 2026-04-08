import sys
import os
sys.path.append('.')

from config import get_config
from dataset.data_loader.MAHNOBHCIrPPGLoader import MAHNOBHCIrPPGLoader
import argparse

def debug_mahnob_loader():
    """独立调试 MAHNOB-HCI DataLoader"""
    
    # 创建配置
    config = get_config(argparse.Namespace(
        config_file="configs/train_configs/MAHNOB_HCI_TSCAN_BASIC.yaml"
    ))
    
    print("=== MAHNOB-HCI DataLoader Debug ===")
    
    # 创建 DataLoader
    dataloader = MAHNOBHCIrPPGLoader(
        name="debug",
        data_path=config.TRAIN.DATA.DATA_PATH,
        config_data=config.TRAIN.DATA
    )
    
    # 1. 查看基本信息
    print(f"\n1. Basic Information:")
    print(f"   Dataset size: {len(dataloader)}")
    print(f"   Input files: {len(dataloader.inputs)}")
    print(f"   Label files: {len(dataloader.labels)}")
    
    # 2. 查看前几个样本的详细信息
    print(f"\n2. Sample Details:")
    dataloader.debug_labels(num_samples=5)
    
    # 3. 分析标签分布
    print(f"\n3. Label Distribution:")
    dataloader.analyze_label_distribution()
    
    # 4. 验证数据一致性
    print(f"\n4. Data Consistency Check:")
    validate_data_consistency(dataloader)

def validate_data_consistency(dataloader):
    """验证数据一致性"""
    print("Checking data consistency...")
    
    issues_found = 0
    for i in range(min(20, len(dataloader))):
        try:
            data, label, filename, chunk_id = dataloader[i]
            
            # 检查数据是否包含 NaN 或 Inf
            if np.isnan(data).any() or np.isinf(data).any():
                print(f"  ❌ Sample {i}: Data contains NaN or Inf")
                issues_found += 1
            
            if np.isnan(label).any() or np.isinf(label).any():
                print(f"  ❌ Sample {i}: Label contains NaN or Inf")
                issues_found += 1
            
            # 检查数据范围是否合理
            if data.max() > 1000 or data.min() < -1000:
                print(f"  ⚠️  Sample {i}: Data range suspicious [{data.min():.2f}, {data.max():.2f}]")
            
            # 检查标签范围
            if label.max() > 10 or label.min() < -10:
                print(f"  ⚠️  Sample {i}: Label range suspicious [{label.min():.2f}, {label.max():.2f}]")
                
        except Exception as e:
            print(f"  ❌ Sample {i}: Error loading - {e}")
            issues_found += 1
    
    if issues_found == 0:
        print("  ✅ All checks passed!")
    else:
        print(f"  ⚠️  Found {issues_found} issues")

if __name__ == "__main__":
    debug_mahnob_loader()
