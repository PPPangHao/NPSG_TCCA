import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
from PIL import Image
import os
from tqdm import tqdm
import numpy as np


# ------------------- 配置 -------------------
class Config:
    data_dir = './FERPlus_dataset'
    pretrained_path = './weight/res50_ir_0.887.pth'
    num_classes = 8  # FER+ 的8个表情类别（排除unknown和NF）
    batch_size = 230
    num_epochs = 500  # 设置为较大的值，由早停机制控制实际训练轮数
    lr = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = './checkpoints'
    # 新增早停相关参数
    early_stop_patience = 50  # 早停耐心值
    min_delta = 0.001  # 效果提升的最小阈值
    early_stop_mode = 'acc'  # 'acc' 或 'loss'，选择监控指标


# ------------------- 自定义数据集 -------------------
class FERPlusDataset(Dataset):
    def __init__(self, image_dir, label_dir, split='FER2013Train', transform=None, mode='single_label'):
        self.image_dir = os.path.join(image_dir, split)
        self.label_file = os.path.join(label_dir, split, 'label.csv')
        self.transform = transform
        self.mode = mode

        # 首先检查CSV文件的实际格式
        print(f"Loading label file: {self.label_file}")

        # 尝试不同的分隔符
        for sep in ['\t', ',', ' ']:
            try:
                self.labels_df = pd.read_csv(self.label_file, sep=sep, header=None)
                if self.labels_df.shape[1] > 1:
                    print(f"Successfully loaded with separator: '{sep}', columns: {self.labels_df.shape[1]}")
                    break
            except:
                continue
        else:
            # 如果上面的方法都失败了，尝试读取原始数据并手动分割
            with open(self.label_file, 'r') as f:
                lines = f.readlines()
            data = []
            for line in lines:
                parts = line.strip().split()
                data.append(parts)
            self.labels_df = pd.DataFrame(data)
            print(f"Manually parsed, columns: {self.labels_df.shape[1]}")

        print(f"CSV shape: {self.labels_df.shape}")
        print(f"First few rows:\n{self.labels_df.head()}")

        # 根据实际列数动态设置列名
        num_columns = self.labels_df.shape[1]
        if num_columns >= 12:
            # 标准FER+格式
            column_names = [
                'image_name', 'usage', 'neutral', 'happiness', 'surprise', 'sadness',
                'anger', 'disgust', 'fear', 'contempt', 'unknown', 'NF'
            ]
            self.labels_df.columns = column_names[:num_columns]
        elif num_columns >= 2:
            # 简化格式
            self.labels_df.columns = ['image_name', 'main_emotion'] + [f'extra_{i}' for i in range(2, num_columns)]
        else:
            raise ValueError(f"CSV file has only {num_columns} columns, expected at least 2")

        # 确保这些列表被正确初始化
        self.image_paths = []
        self.label_list = []

        print(f"Processing {split} dataset...")
        valid_count = 0

        for idx, row in self.labels_df.iterrows():
            img_name = row['image_name']
            # 确保图片名有正确的扩展名
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_name += '.png'

            img_path = os.path.join(self.image_dir, img_name)

            if os.path.exists(img_path):
                self.image_paths.append(img_path)
                valid_count += 1

                try:
                    if 'neutral' in row and num_columns >= 10:  # 标准FER+格式
                        emotion_votes = [
                            int(row['neutral']), int(row['happiness']), int(row['surprise']),
                            int(row['sadness']), int(row['anger']), int(row['disgust']),
                            int(row['fear']), int(row['contempt'])
                        ]

                        total_votes = sum(emotion_votes)
                        if total_votes > 0:
                            emotion_probs = [vote / total_votes for vote in emotion_votes]
                        else:
                            emotion_probs = [0.125] * 8

                        if self.mode == 'single_label':
                            label = np.argmax(emotion_probs)
                            self.label_list.append(label)
                        else:
                            self.label_list.append(emotion_probs)
                    else:  # 简化格式
                        main_emotion = int(row.iloc[1])  # 使用第二列作为标签
                        self.label_list.append(main_emotion)

                except (ValueError, KeyError) as e:
                    print(f"Error processing row {idx}: {e}")
                    # 使用默认值
                    if self.mode == 'single_label':
                        self.label_list.append(0)
                    else:
                        self.label_list.append([1.0] + [0.0] * 7)
            else:
                if idx < 3:  # 只打印前3个缺失文件的警告
                    print(f"Warning: Image {img_path} not found")

        print(f"Successfully loaded {valid_count} samples from {split}")

        if valid_count == 0:
            raise ValueError(f"No valid images found in {split}")

    def __len__(self):
        """返回数据集大小"""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """根据索引获取数据样本"""
        img_path = self.image_paths[idx]

        # 加载图片
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 返回一个黑色图像作为备用
            image = Image.new('RGB', (48, 48), color='black')

        if self.transform:
            image = self.transform(image)

        if self.mode == 'single_label':
            label = torch.tensor(self.label_list[idx], dtype=torch.long)
        else:
            label = torch.tensor(self.label_list[idx], dtype=torch.float)

        return image, label


# ------------------- 支持多标签的损失函数 -------------------
class MultiLabelLoss(nn.Module):
    def __init__(self):
        super(MultiLabelLoss, self).__init__()

    def forward(self, outputs, targets):
        # 使用KL散度损失来处理概率分布
        log_probs = nn.functional.log_softmax(outputs, dim=1)
        loss = nn.functional.kl_div(log_probs, targets, reduction='batchmean')
        return loss


# ------------------- 早停类 -------------------
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, mode='acc'):
        """
        Args:
            patience: 在多少个epoch没有改善后停止训练
            min_delta: 最小改善阈值
            mode: 'acc' 或 'loss'，选择监控指标
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'loss':
            # 对于loss，我们希望它越小越好
            self.best_score = float('inf')
            self.min_delta = -min_delta  # 对于loss，delta应该是负的

    def __call__(self, current_score):
        if self.mode == 'acc':
            # 对于准确率，越高越好
            if self.best_score is None:
                self.best_score = current_score
                return False
                
            if current_score - self.best_score > self.min_delta:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
                    
        elif self.mode == 'loss':
            # 对于损失，越低越好
            if self.best_score is None:
                self.best_score = current_score
                return False
                
            if self.best_score - current_score > self.min_delta:
                self.best_score = current_score
                self.counter = 0
            else:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
        
        return self.early_stop


# ------------------- 数据增强 -------------------
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform


# ------------------- 创建DataLoader -------------------
def create_dataloaders(config, mode='single_label'):
    image_root = os.path.join(config.data_dir, 'images')
    label_root = os.path.join(config.data_dir, 'labels')

    train_transform, val_transform = get_transforms()

    # 创建数据集
    train_dataset = FERPlusDataset(
        image_dir=image_root,
        label_dir=label_root,
        split='FER2013Train',
        transform=train_transform,
        mode=mode
    )

    val_dataset = FERPlusDataset(
        image_dir=image_root,
        label_dir=label_root,
        split='FER2013Valid',
        transform=val_transform,
        mode=mode
    )

    test_dataset = FERPlusDataset(
        image_dir=image_root,
        label_dir=label_root,
        split='FER2013Test',
        transform=val_transform,
        mode=mode
    )

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


# ------------------- 加载预训练模型 -------------------
def load_pretrained_model(config):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, config.num_classes)

    if os.path.exists(config.pretrained_path):
        print(f"Loading pretrained weights from {config.pretrained_path}")
        state_dict = torch.load(config.pretrained_path, map_location=config.device)

        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict, strict=False)
        print("Pretrained weights loaded successfully!")
    else:
        print("Using ImageNet pretrained weights...")
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, config.num_classes)

    return model.to(config.device)


# ------------------- 训练函数（支持多标签） -------------------
def train_one_epoch(model, loader, criterion, optimizer, device, mode='single_label'):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        if mode == 'single_label':
            loss = criterion(outputs, labels)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
        else:
            loss = criterion(outputs, labels)
            # 对于多标签，我们可以计算top-1准确率
            _, preds = outputs.max(1)
            _, true_labels = labels.max(1)
            correct += (preds == true_labels).sum().item()

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total += labels.size(0)

        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{correct / total:.4f}' if total > 0 else '0.0000'
        })

    return running_loss / total if total > 0 else 0, correct / total if total > 0 else 0


def validate(model, loader, criterion, device, mode='single_label'):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            if mode == 'single_label':
                loss = criterion(outputs, labels)
                _, preds = outputs.max(1)
                correct += (preds == labels).sum().item()
            else:
                loss = criterion(outputs, labels)
                _, preds = outputs.max(1)
                _, true_labels = labels.max(1)
                correct += (preds == true_labels).sum().item()

            running_loss += loss.item() * images.size(0)
            total += labels.size(0)

            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{correct / total:.4f}' if total > 0 else '0.0000'
            })

    return running_loss / total if total > 0 else 0, correct / total if total > 0 else 0


# ------------------- 主训练循环 -------------------
def main(mode='single_label'):
    config = Config()
    os.makedirs(config.save_dir, exist_ok=True)

    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config, mode=mode)

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    print("Loading model...")
    model = load_pretrained_model(config)

    # 选择损失函数
    if mode == 'single_label':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = MultiLabelLoss()

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # 初始化早停器
    early_stopping = EarlyStopping(
        patience=config.early_stop_patience,
        min_delta=config.min_delta,
        mode=config.early_stop_mode
    )

    best_acc = 0.0
    best_loss = float('inf')
    print("Starting training...")

    for epoch in range(config.num_epochs):
        print(f"\nEpoch [{epoch + 1}/{config.num_epochs}]")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, config.device, mode)
        val_loss, val_acc = validate(model, val_loader, criterion, config.device, mode)

        scheduler.step()

        print(f"Epoch {epoch + 1}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(config.save_dir, 'best_model.pth'))
            print(f"New best model saved with accuracy: {val_acc:.4f}")

        torch.save(model.state_dict(), os.path.join(config.save_dir, 'last_model.pth'))

        # 早停检查
        if config.early_stop_mode == 'acc':
            should_stop = early_stopping(val_acc)
        else:  # loss mode
            should_stop = early_stopping(val_loss)
            
        if should_stop:
            print(f"Early stopping triggered after {epoch + 1} epochs!")
            print(f"Best validation accuracy: {best_acc:.4f}, Best validation loss: {best_loss:.4f}")
            break

    # 如果没有触发早停，训练完成
    if not early_stopping.early_stop:
        print("Training completed!")
        print(f"Best validation accuracy: {best_acc:.4f}, Best validation loss: {best_loss:.4f}")


if __name__ == "__main__":
    # 可以选择单标签或多标签模式
    main(mode='single_label')  # 或者 'multi_label'