import os
import cv2
import glob
from tqdm import tqdm
import argparse

def check_avi_files(dataset_path):
    """
    扫描指定路径下所有 AVI 文件的有效性
    
    Args:
        dataset_path (str): 数据集根路径
    """
    print(f"开始扫描路径: {dataset_path}")
    
    # 查找所有 AVI 文件
    avi_pattern = os.path.join(dataset_path, "**", "*.avi")
    avi_files = glob.glob(avi_pattern, recursive=True)
    
    print(f"找到 {len(avi_files)} 个 AVI 文件")
    
    invalid_files = []
    valid_files = []
    file_details = []
    
    # 检查每个 AVI 文件
    for avi_file in tqdm(avi_files, desc="检查 AVI 文件"):
        try:
            # 尝试打开视频文件
            cap = cv2.VideoCapture(avi_file)
            
            if not cap.isOpened():
                invalid_files.append(avi_file)
                file_details.append({
                    'file': avi_file,
                    'status': '无法打开',
                    'frames': 0,
                    'width': 0,
                    'height': 0,
                    'fps': 0
                })
                continue
            
            # 获取视频信息
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # 尝试读取第一帧
            ret, frame = cap.read()
            
            if not ret or frame_count == 0:
                invalid_files.append(avi_file)
                file_details.append({
                    'file': avi_file,
                    'status': '无法读取帧',
                    'frames': frame_count,
                    'width': width,
                    'height': height,
                    'fps': fps
                })
            else:
                valid_files.append(avi_file)
                file_details.append({
                    'file': avi_file,
                    'status': '有效',
                    'frames': frame_count,
                    'width': width,
                    'height': height,
                    'fps': fps
                })
            
            cap.release()
            
        except Exception as e:
            invalid_files.append(avi_file)
            file_details.append({
                'file': avi_file,
                'status': f'异常: {str(e)}',
                'frames': 0,
                'width': 0,
                'height': 0,
                'fps': 0
            })
    
    return valid_files, invalid_files, file_details

def print_results(valid_files, invalid_files, file_details):
    """打印扫描结果"""
    print("\n" + "="*80)
    print("扫描结果汇总")
    print("="*80)
    print(f"总文件数: {len(valid_files) + len(invalid_files)}")
    print(f"有效文件: {len(valid_files)}")
    print(f"无效文件: {len(invalid_files)}")
    print(f"有效率: {len(valid_files)/(len(valid_files)+len(invalid_files))*100:.2f}%")
    
    if invalid_files:
        print("\n" + "="*80)
        print("无效文件列表:")
        print("="*80)
        for detail in file_details:
            if detail['status'] != '有效':
                print(f"❌ {detail['file']}")
                print(f"   状态: {detail['status']}")
                print(f"   帧数: {detail['frames']}, 分辨率: {detail['width']}x{detail['height']}, FPS: {detail['fps']:.2f}")
                print()
    
    # 打印有效文件的统计信息
    if valid_files:
        print("\n" + "="*80)
        print("有效文件统计:")
        print("="*80)
        
        valid_details = [d for d in file_details if d['status'] == '有效']
        
        # 帧数统计
        frames = [d['frames'] for d in valid_details]
        print(f"帧数范围: {min(frames)} - {max(frames)}")
        print(f"平均帧数: {sum(frames)/len(frames):.1f}")
        
        # 分辨率统计
        resolutions = set([(d['width'], d['height']) for d in valid_details])
        print(f"分辨率类型: {len(resolutions)} 种")
        for res in sorted(resolutions):
            count = len([d for d in valid_details if d['width'] == res[0] and d['height'] == res[1]])
            print(f"  {res[0]}x{res[1]}: {count} 个文件")
        
        # FPS 统计
        fps_list = [d['fps'] for d in valid_details]
        print(f"FPS 范围: {min(fps_list):.2f} - {max(fps_list):.2f}")
        print(f"平均 FPS: {sum(fps_list)/len(fps_list):.2f}")

def save_report(valid_files, invalid_files, file_details, output_file="avi_scan_report.txt"):
    """保存详细报告到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("AVI 文件扫描报告\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"扫描时间: {os.path.basename(__file__)}\n")
        f.write(f"总文件数: {len(valid_files) + len(invalid_files)}\n")
        f.write(f"有效文件: {len(valid_files)}\n")
        f.write(f"无效文件: {len(invalid_files)}\n")
        f.write(f"有效率: {len(valid_files)/(len(valid_files)+len(invalid_files))*100:.2f}%\n\n")
        
        if invalid_files:
            f.write("无效文件详情:\n")
            f.write("-"*50 + "\n")
            for detail in file_details:
                if detail['status'] != '有效':
                    f.write(f"文件: {detail['file']}\n")
                    f.write(f"状态: {detail['status']}\n")
                    f.write(f"帧数: {detail['frames']}, 分辨率: {detail['width']}x{detail['height']}, FPS: {detail['fps']:.2f}\n\n")
        
        f.write("所有文件详情:\n")
        f.write("-"*50 + "\n")
        for detail in file_details:
            f.write(f"文件: {detail['file']}\n")
            f.write(f"状态: {detail['status']}\n")
            f.write(f"帧数: {detail['frames']}, 分辨率: {detail['width']}x{detail['height']}, FPS: {detail['fps']:.2f}\n\n")
    
    print(f"\n详细报告已保存到: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='扫描 AVI 文件有效性')
    parser.add_argument('--dataset_path', type=str, default='/dataset/MAC',
                       help='数据集根路径 (默认: /dataset/MAC)')
    parser.add_argument('--save_report', action='store_true',
                       help='保存详细报告到文件')
    parser.add_argument('--output_file', type=str, default='avi_scan_report.txt',
                       help='报告输出文件名')
    
    args = parser.parse_args()
    
    # 检查路径是否存在
    if not os.path.exists(args.dataset_path):
        print(f"错误: 路径 {args.dataset_path} 不存在!")
        return
    
    # 扫描 AVI 文件
    valid_files, invalid_files, file_details = check_avi_files(args.dataset_path)
    
    # 打印结果
    print_results(valid_files, invalid_files, file_details)
    
    # 保存报告
    if args.save_report:
        save_report(valid_files, invalid_files, file_details, args.output_file)

# 简化版本 - 直接运行
def quick_scan():
    """快速扫描版本"""
    dataset_path = "/dataset/MAC"
    
    if not os.path.exists(dataset_path):
        print(f"路径 {dataset_path} 不存在!")
        return
    
    print(f"快速扫描: {dataset_path}")
    
    # 查找所有 AVI 文件
    avi_pattern = os.path.join(dataset_path, "**", "*.avi")
    avi_files = glob.glob(avi_pattern, recursive=True)
    
    print(f"找到 {len(avi_files)} 个 AVI 文件")
    
    invalid_files = []
    
    for avi_file in tqdm(avi_files, desc="快速检查"):
        try:
            cap = cv2.VideoCapture(avi_file)
            if not cap.isOpened():
                invalid_files.append(avi_file)
                continue
            
            # 只检查是否能打开和读取第一帧
            ret, frame = cap.read()
            if not ret:
                invalid_files.append(avi_file)
            
            cap.release()
            
        except Exception as e:
            invalid_files.append(avi_file)
    
    # 打印无效文件
    if invalid_files:
        print(f"\n❌ 发现 {len(invalid_files)} 个无效文件:")
        for file in invalid_files:
            print(f"  {file}")
    else:
        print("\n✅ 所有 AVI 文件都是有效的!")

if __name__ == "__main__":
    # 使用完整版本
    main()
    
    # 或者使用快速版本
    # quick_scan()
