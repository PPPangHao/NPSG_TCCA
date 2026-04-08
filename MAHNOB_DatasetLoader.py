import os
import torch
import numpy as np
import pandas as pd
import cv2
import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional
import pyedflib  # 用于读取BDF文件

class MAHNOBEmotionDataset(Dataset):
    """
    MAHNOB-HCI情绪识别数据集加载器
    支持valence/arousal二分类和四分类
    """
    
    def __init__(self, 
                 dataset_path: str,
                 session_ids: List[str],  # 改为字符串列表
                 transform=None,
                 target_type: str = 'valence_arousal',
                 classification_type: str = 'binary',
                 modality: str = 'both',
                 window_length: int = 300,
                 overlap: float = 0.5,
                 sessions_info_file: str = 'sessions_info.json'):
        """
        初始化数据集
        
        Args:
            dataset_path: MAHNOB-HCI数据集路径
            session_ids: 使用的会话ID列表 (字符串格式，如 ['2', '4', '5'])
            transform: 图像变换
            target_type: 目标类型
            classification_type: 分类类型
            modality: 使用的模态
            window_length: 时间窗口长度
            overlap: 窗口重叠率
            sessions_info_file: 会话信息JSON文件
        """
        self.dataset_path = dataset_path
        self.session_ids = session_ids
        self.transform = transform
        self.target_type = target_type
        self.classification_type = classification_type
        self.modality = modality
        self.window_length = window_length
        self.overlap = overlap
        
        # 情感阈值
        self.valence_threshold = 5
        self.arousal_threshold = 5
        
        # 会话信息文件路径
        self.sessions_info_file = os.path.join(dataset_path, sessions_info_file)
        
        # 获取或创建会话信息
        self.sessions_info = self._get_sessions_info()
        
        # 加载所有会话数据
        self.samples = self._load_all_sessions()
        
        print(f"Loaded {len(self.samples)} samples from {len(session_ids)} sessions")
        print(f"Target: {target_type}, Classification: {classification_type}, Modality: {modality}")
    
    def _get_sessions_info(self) -> Dict[str, Dict]:
        """获取或创建会话信息"""
        if os.path.exists(self.sessions_info_file):
            print(f"Loading sessions info from {self.sessions_info_file}")
            with open(self.sessions_info_file, 'r') as f:
                return json.load(f)
        else:
            print("Scanning Sessions folder...")
            sessions_info = self._scan_sessions_folder()
            # 保存到JSON文件
            with open(self.sessions_info_file, 'w') as f:
                json.dump(sessions_info, f, indent=2)
            print(f"Sessions info saved to {self.sessions_info_file}")
            return sessions_info
    
    def _scan_sessions_folder(self) -> Dict[str, Dict]:
        """扫描Sessions文件夹，获取所有可用的会话"""
        sessions_path = os.path.join(self.dataset_path, 'Sessions')
        sessions_info = {}
        
        if not os.path.exists(sessions_path):
            raise ValueError(f"Sessions folder not found: {sessions_path}")
        
        # 获取Sessions文件夹下的所有子文件夹
        session_folders = [f for f in os.listdir(sessions_path) 
                          if os.path.isdir(os.path.join(sessions_path, f))]
        
        print(f"Found {len(session_folders)} session folders: {session_folders}")
        
        for session_folder in session_folders:
            session_path = os.path.join(sessions_path, session_folder)
            session_info = self._parse_session_info(session_path, session_folder)
            if session_info:
                sessions_info[session_folder] = session_info
        
        print(f"Successfully parsed {len(sessions_info)} sessions")
        return sessions_info
    
    def _parse_session_info(self, session_path: str, session_folder: str) -> Optional[Dict]:
        """解析单个会话的信息"""
        xml_path = os.path.join(session_path, 'session.xml')
        
        if not os.path.exists(xml_path):
            print(f"Warning: session.xml not found in {session_path}")
            return None
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            session_attrs = root.attrib
            
            # 获取情感标签
            emotion_labels = {
                'valence': int(session_attrs.get('feltVlnc', 0)),
                'arousal': int(session_attrs.get('feltArsl', 0)),
                'emotion_type': int(session_attrs.get('feltEmo', 0)),
                'control': int(session_attrs.get('feltCtrl', 0)),
                'predictability': int(session_attrs.get('feltPred', 0))
            }
            
            # 检查情感标签是否有效
            if emotion_labels['valence'] == 0 or emotion_labels['arousal'] == 0:
                print(f"Warning: Invalid emotion labels in session {session_folder}")
                return None
            
            # 获取视频文件
            video_files = []
            for track in root.findall('track'):
                if track.get('type') == 'Video':
                    video_file = track.get('filename')
                    if video_file and os.path.exists(os.path.join(session_path, video_file)):
                        video_files.append(video_file)
            
            # 获取生理数据文件
            physio_file = None
            for track in root.findall('track'):
                if track.get('type') == 'Physiological':
                    physio_file = track.get('filename')
                    break
            
            session_info = {
                'session_id': session_folder,
                'xml_path': xml_path,
                'session_path': session_path,
                'video_files': video_files,
                'physio_file': physio_file,
                'emotion_labels': emotion_labels,
                'duration': float(session_attrs.get('cutLenSec', 0)),
                'video_rate': float(session_attrs.get('vidRate', 0)),
                'has_valid_data': len(video_files) > 0
            }
            
            print(f"Session {session_folder}: valence={emotion_labels['valence']}, "
                  f"arousal={emotion_labels['arousal']}, videos={len(video_files)}")
            
            return session_info
            
        except Exception as e:
            print(f"Error parsing session {session_folder}: {e}")
            return None
    
    def _load_all_sessions(self) -> List[Dict]:
        """加载所有会话的数据样本"""
        samples = []
        
        for session_id in self.session_ids:
            if session_id not in self.sessions_info:
                print(f"Warning: Session {session_id} not found in sessions info, skipping...")
                continue
            
            session_info = self.sessions_info[session_id]
            
            if not session_info['has_valid_data']:
                print(f"Warning: Session {session_id} has no valid data, skipping...")
                continue
            
            # 生成时间窗口样本
            emotion_labels = session_info['emotion_labels']
            valence = emotion_labels['valence']
            arousal = emotion_labels['arousal']
            
            session_samples = self._generate_window_samples(session_info, valence, arousal)
            samples.extend(session_samples)
        
        return samples
    
    def _generate_window_samples(self, session_info: Dict, valence: int, arousal: int) -> List[Dict]:
        """为会话生成时间窗口样本"""
        samples = []
        
        # 计算总帧数
        video_rate = session_info['video_rate']
        total_frames = int(session_info['duration'] * video_rate)
        
        # 如果总帧数小于窗口长度，使用所有帧
        if total_frames < self.window_length:
            window_length = total_frames
            step = 1
        else:
            window_length = self.window_length
            step = int(window_length * (1 - self.overlap))
            if step == 0:
                step = 1
        
        # 生成时间窗口
        for start_frame in range(0, total_frames - window_length + 1, step):
            end_frame = start_frame + window_length
            
            sample = {
                'session_info': session_info,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'valence': valence,
                'arousal': arousal,
                'window_length': window_length
            }
            samples.append(sample)
        
        print(f"Generated {len(samples)} samples for session {session_info['session_id']}")
        return samples
    
    def _extract_rppg_signals(self, video_frames: List[np.ndarray]) -> np.ndarray:
        """从视频帧中提取rPPG信号"""
        rppg_signals = []
        
        for frame in video_frames:
            try:
                # 转换为RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 简单的面部区域提取 (实际应用中应该使用面部检测)
                h, w = frame_rgb.shape[:2]
                face_region = frame_rgb[h//4:3*h//4, w//4:3*w//4]
                
                if len(face_region) > 0:
                    # 提取绿色通道的平均值作为rPPG信号
                    g_channel = face_region[:, :, 1].mean()
                    rppg_signals.append(g_channel)
                else:
                    rppg_signals.append(0)
            except Exception as e:
                rppg_signals.append(0)
        
        return np.array(rppg_signals)
    
    def _load_video_frames(self, session_info: Dict, start_frame: int, end_frame: int) -> List[np.ndarray]:
        """加载指定范围的视频帧"""
        session_path = session_info['session_path']
        video_files = session_info['video_files']
        
        for video_file in video_files:
            video_path = os.path.join(session_path, video_file)
            
            if not os.path.exists(video_path):
                continue
            
            try:
                frames = []
                cap = cv2.VideoCapture(video_path)
                
                # 设置起始帧
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                
                for i in range(self.window_length):
                    ret, frame = cap.read()
                    if ret:
                        frames.append(frame)
                    else:
                        break
                
                cap.release()
                
                if len(frames) == self.window_length:
                    return frames
                    
            except Exception as e:
                print(f"Error loading video {video_path}: {e}")
                continue
        
        return []  # 没有找到可用的视频
    
    def _get_emotion_label(self, valence: int, arousal: int) -> torch.Tensor:
        """根据valence和arousal生成情感标签"""
        
        if self.classification_type == 'binary':
            if self.target_type == 'valence':
                label = 1 if valence >= self.valence_threshold else 0
                return torch.tensor(label, dtype=torch.long)
            
            elif self.target_type == 'arousal':
                label = 1 if arousal >= self.arousal_threshold else 0
                return torch.tensor(label, dtype=torch.long)
            
            else:  # valence_arousal
                valence_label = 1 if valence >= self.valence_threshold else 0
                arousal_label = 1 if arousal >= self.arousal_threshold else 0
                return torch.tensor([valence_label, arousal_label], dtype=torch.long)
        
        else:  # four_class
            valence_label = 1 if valence >= self.valence_threshold else 0
            arousal_label = 1 if arousal >= self.arousal_threshold else 0
            
            if valence_label == 0 and arousal_label == 0:  # LV-LA
                class_id = 0  # sad
            elif valence_label == 0 and arousal_label == 1:  # LV-HA
                class_id = 1  # anger
            elif valence_label == 1 and arousal_label == 0:  # HV-LA
                class_id = 2  # calm
            else:  # HV-HA
                class_id = 3  # happy
            
            return torch.tensor(class_id, dtype=torch.long)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        session_info = sample_info['session_info']
        
        # 加载视频数据
        video_frames = self._load_video_frames(
            session_info, sample_info['start_frame'], sample_info['end_frame']
        )
        
        if not video_frames:
            return self._get_dummy_sample()
        
        # 提取特征
        features = {}
        
        if self.modality in ['video', 'both']:
            # 面部表情特征 - 使用中间帧
            middle_frame = video_frames[len(video_frames) // 2]
            try:
                middle_frame_pil = Image.fromarray(cv2.cvtColor(middle_frame, cv2.COLOR_BGR2RGB))
                if self.transform:
                    facial_features = self.transform(middle_frame_pil)
                else:
                    facial_features = torch.tensor(np.array(middle_frame_pil), dtype=torch.float32).permute(2, 0, 1)
                features['facial'] = facial_features
            except Exception as e:
                features['facial'] = torch.zeros((3, 224, 224), dtype=torch.float32)
        
        if self.modality in ['rppg', 'both']:
            # 提取rPPG信号
            try:
                rppg_signal = self._extract_rppg_signals(video_frames)
                features['rppg'] = torch.tensor(rppg_signal, dtype=torch.float32)
            except Exception as e:
                features['rppg'] = torch.zeros(self.window_length, dtype=torch.float32)
        
        # 获取情感标签
        label = self._get_emotion_label(
            sample_info['valence'], sample_info['arousal']
        )
        
        return features, label
    
    def _get_dummy_sample(self):
        """返回虚拟样本用于处理缺失数据"""
        features = {}
        
        if self.modality in ['video', 'both']:
            features['facial'] = torch.zeros((3, 224, 224), dtype=torch.float32)
        
        if self.modality in ['rppg', 'both']:
            features['rppg'] = torch.zeros(self.window_length, dtype=torch.float32)
        
        # 使用中性标签
        label = self._get_emotion_label(5, 5)
        
        return features, label

# 数据变换定义
def get_transforms():
    """获取数据变换"""
    from torchvision import transforms
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def get_available_sessions(dataset_path: str, sessions_info_file: str = 'sessions_info.json') -> List[str]:
    """获取所有可用的会话ID"""
    sessions_info_path = os.path.join(dataset_path, sessions_info_file)
    
    if os.path.exists(sessions_info_path):
        with open(sessions_info_path, 'r') as f:
            sessions_info = json.load(f)
        available_sessions = [session_id for session_id, info in sessions_info.items() 
                             if info.get('has_valid_data', False)]
        return available_sessions
    else:
        # 如果JSON文件不存在，创建数据集实例来生成它
        dataset = MAHNOBEmotionDataset(
            dataset_path=dataset_path,
            session_ids=[],  # 空列表，只生成JSON文件
            transform=None
        )
        return list(dataset.sessions_info.keys())

def create_mahnob_dataloaders(dataset_path, batch_size=8, modality='both', 
                             target_type='valence_arousal', test_ratio=0.2, val_ratio=0.2):
    """创建MAHNOB数据加载器"""
    
    # 获取所有可用的会话
    available_sessions = get_available_sessions(dataset_path)
    print(f"Available sessions: {available_sessions}")
    
    if not available_sessions:
        raise ValueError("No available sessions found!")
    
    # 随机分割会话
    np.random.seed(42)
    np.random.shuffle(available_sessions)
    
    n_total = len(available_sessions)
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_test - n_val
    
    train_sessions = available_sessions[:n_train]
    val_sessions = available_sessions[n_train:n_train + n_val]
    test_sessions = available_sessions[n_train + n_val:]
    
    print(f"Data split: Train={len(train_sessions)}, Val={len(val_sessions)}, Test={len(test_sessions)}")
    print(f"Train sessions: {train_sessions}")
    print(f"Val sessions: {val_sessions}")
    print(f"Test sessions: {test_sessions}")
    
    train_transform, val_transform = get_transforms()
    
    # 创建数据集
    train_dataset = MAHNOBEmotionDataset(
        dataset_path=dataset_path,
        session_ids=train_sessions,
        transform=train_transform,
        target_type=target_type,
        modality=modality
    )
    
    val_dataset = MAHNOBEmotionDataset(
        dataset_path=dataset_path,
        session_ids=val_sessions,
        transform=val_transform,
        target_type=target_type,
        modality=modality
    )
    
    test_dataset = MAHNOBEmotionDataset(
        dataset_path=dataset_path,
        session_ids=test_sessions,
        transform=val_transform,
        target_type=target_type,
        modality=modality
    )
    
    # 检查数据集是否为空
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty! Check your data paths and session availability.")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0  # 先设为0避免多进程问题
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    return train_loader, val_loader, test_loader

# 使用示例
if __name__ == "__main__":
    dataset_path = r"F:\LNNU\Code\Review\rPPG-Toolbox\MAHNOB-HCI"  # 使用原始字符串
    
    try:
        # 首先扫描并生成会话信息
        print("Scanning MAHNOB-HCI dataset...")
        available_sessions = get_available_sessions(dataset_path)
        print(f"Found {len(available_sessions)} available sessions: {available_sessions}")
        
        # 创建数据加载器
        train_loader, val_loader, test_loader = create_mahnob_dataloaders(
            dataset_path=dataset_path,
            batch_size=4,  # 先用小batch size测试
            modality='both',
            target_type='valence_arousal'
        )
        
        # 测试数据加载
        print("\nTesting data loading...")
        for batch_idx, (features, labels) in enumerate(train_loader):
            print(f"Batch {batch_idx}:")
            if 'facial' in features:
                print(f"  Facial features shape: {features['facial'].shape}")
            if 'rppg' in features:
                print(f"  rPPG features shape: {features['rppg'].shape}")
            print(f"  Labels shape: {labels.shape}")
            
            if batch_idx >= 2:  # 只查看前3个batch
                break
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()