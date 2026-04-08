import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import time

# ------------------- 自定义模型类 -------------------
class ResNetFER(nn.Module):
    def __init__(self, num_classes=8, pretrained_path=None, device='cuda'):
        super(ResNetFER, self).__init__()
        self.device = device

        # 创建ResNet50骨干网络
        self.backbone = models.resnet50(pretrained=False)

        # 如果提供了预训练权重路径
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading pretrained weights from {pretrained_path}")
            self.load_pretrained_weights(pretrained_path)
        else:
            print("Using randomly initialized weights")

        # 修改最后一层
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)

        # 初始化最后一层
        nn.init.xavier_uniform_(self.backbone.fc.weight)
        nn.init.constant_(self.backbone.fc.bias, 0)

    def load_pretrained_weights(self, weight_path):
        """加载预训练权重"""
        try:
            checkpoint = torch.load(weight_path, map_location='cpu')

            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    k = k[7:]
                if not k.startswith('backbone.'):
                    new_key = 'backbone.' + k
                else:
                    new_key = k
                new_state_dict[new_key] = v

            model_dict = self.backbone.state_dict()
            pretrained_dict = {k: v for k, v in new_state_dict.items()
                               if k in model_dict and v.shape == model_dict[k].shape}

            print(f"Successfully loaded {len(pretrained_dict)}/{len(model_dict)} parameters")
            model_dict.update(pretrained_dict)
            self.backbone.load_state_dict(model_dict, strict=False)

        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            print("Using randomly initialized weights instead")

    def forward(self, x):
        return self.backbone(x)

    def init(self):
        self.to(self.device)
        print(f"Model initialized on {self.device}")

# ------------------- 修复的数据集类 -------------------
class FERPlusDataset(Dataset):
    def __init__(self, image_dir, label_dir, split='FER2013Train', transform=None, mode='multi_label'):
        self.image_dir = os.path.join(image_dir, split)
        self.label_file = os.path.join(label_dir, split, 'label.csv')
        self.transform = transform
        self.mode = mode

        # 加载标签文件
        print(f"Loading label file: {self.label_file}")

        for sep in ['\t', ',', ' ']:
            try:
                self.labels_df = pd.read_csv(self.label_file, sep=sep, header=None)
                if self.labels_df.shape[1] > 1:
                    print(f"Successfully loaded with separator: '{sep}', columns: {self.labels_df.shape[1]}")
                    break
            except:
                continue
        else:
            with open(self.label_file, 'r') as f:
                lines = f.readlines()
            data = []
            for line in lines:
                parts = line.strip().split()
                data.append(parts)
            self.labels_df = pd.DataFrame(data)
            print(f"Manually parsed, columns: {self.labels_df.shape[1]}")

        print(f"CSV shape: {self.labels_df.shape}")

        # 动态设置列名
        num_columns = self.labels_df.shape[1]
        if num_columns >= 12:
            column_names = [
                'image_name', 'usage', 'neutral', 'happiness', 'surprise', 'sadness',
                'anger', 'disgust', 'fear', 'contempt', 'unknown', 'NF'
            ]
            self.labels_df.columns = column_names[:num_columns]
        else:
            raise ValueError(f"CSV file has only {num_columns} columns, expected at least 12.")

        # 初始化图像路径和标签
        self.image_paths = []
        self.label_list = []

        print(f"Processing {split} dataset...")
        valid_count = 0

        for idx, row in self.labels_df.iterrows():
            img_name = row['image_name']
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_name += '.png'

            img_path = os.path.join(self.image_dir, img_name)

            if os.path.exists(img_path):
                self.image_paths.append(img_path)
                valid_count += 1

                try:
                    if 'neutral' in row and num_columns >= 10:
                        # 获取8个情绪的投票数
                        emotion_votes = [
                            int(row['neutral']), int(row['happiness']), int(row['surprise']),
                            int(row['sadness']), int(row['anger']), int(row['disgust']),
                            int(row['fear']), int(row['contempt'])
                        ]

                        # 计算总票数（排除unknown和NF）
                        total_votes = sum(emotion_votes)
                        
                        if total_votes > 0:
                            # 转换为概率分布
                            emotion_probs = [vote / total_votes for vote in emotion_votes]
                        else:
                            # 如果没有投票，使用均匀分布
                            emotion_probs = [1.0 / len(emotion_votes)] * len(emotion_votes)
                        
                        label = torch.tensor(emotion_probs, dtype=torch.float32)
                        self.label_list.append(label)
                    else:
                        # 单标签情况
                        main_emotion = int(row.iloc[1])
                        one_hot = torch.zeros(8, dtype=torch.float32)
                        one_hot[main_emotion] = 1.0
                        self.label_list.append(one_hot)

                except (ValueError, KeyError, IndexError) as e:
                    print(f"Error processing row {idx}: {e}")
                    # 使用中性标签作为默认值
                    default_label = torch.zeros(8, dtype=torch.float32)
                    default_label[0] = 1.0  # neutral
                    self.label_list.append(default_label)

        print(f"Successfully loaded {valid_count} samples from {split}")

        if valid_count == 0:
            raise ValueError(f"No valid images found in {split}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (48, 48), color='black')

        if self.transform:
            image = self.transform(image)

        label = self.label_list[idx]
        return image, label

# ------------------- 数据变换 -------------------
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

# ------------------- 创建 DataLoader -------------------
def create_dataloaders(config, mode='multi_label'):
    image_root = os.path.join(config.data_dir, 'images')
    label_root = os.path.join(config.data_dir, 'labels')

    train_transform, val_transform = get_transforms()

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

# ------------------- 修复的可视化类 -------------------
class TrainingVisualizer:
    def __init__(self):
        plt.ion()  # 开启交互模式
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 5))
        self.fig.suptitle('Training Progress', fontsize=16)
        
        # 初始化数据存储
        self.train_losses = []
        self.val_f1_scores = []
        self.epochs = []
        
        # 设置图表
        self.setup_plots()
        plt.tight_layout()
        plt.show()
    
    def setup_plots(self):
        # 训练损失图
        self.ax1.clear()
        self.ax1.set_title('Training Loss')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # 验证F1分数图
        self.ax2.clear()
        self.ax2.set_title('Validation F1 Score')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('F1 Score')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.ax2.set_ylim(0, 1)
    
    def update_plots(self, epoch, train_loss, val_f1):
        # 确保数据是标量值
        train_loss = float(train_loss) if torch.is_tensor(train_loss) else float(train_loss)
        val_f1 = float(val_f1) if torch.is_tensor(val_f1) else float(val_f1)
        
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_f1_scores.append(val_f1)
        
        # 清除并重新绘制
        self.setup_plots()
        
        # 绘制训练损失
        if len(self.epochs) > 0:
            self.ax1.plot(self.epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
            self.ax1.legend()
            
            # 自动调整y轴范围，但确保包含0
            y_min = min(self.train_losses)
            y_max = max(self.train_losses)
            margin = (y_max - y_min) * 0.1
            self.ax1.set_ylim(y_min - margin, y_max + margin)
        
        # 绘制验证F1分数
        if len(self.epochs) > 0:
            self.ax2.plot(self.epochs, self.val_f1_scores, 'r-', label='Val F1 Score', linewidth=2)
            self.ax2.legend()
            self.ax2.set_ylim(0, max(1.0, max(self.val_f1_scores) * 1.1))
        
        # 刷新图表
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # 添加短暂暂停以确保图表更新
        plt.pause(0.1)
    
    def save_plot(self, filename='training_progress.png'):
        """保存训练过程图表"""
        try:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Training progress saved as {filename}")
        except Exception as e:
            print(f"Error saving plot: {e}")
    
    def close(self):
        """关闭图表"""
        plt.ioff()
        plt.close()

# ------------------- 修复的训练器类 -------------------
class FERTrainer:
    def __init__(self, model, device, save_path, learning_rate=1e-4, enable_visualization=True):
        self.model = model
        self.device = device
        self.save_path = save_path
        self.learning_rate = learning_rate
        self.enable_visualization = enable_visualization

        # 使用更适合多标签分类的损失函数
        self.criterion = nn.BCEWithLogitsLoss()
        
        # 使用Adam优化器，更稳定
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=3, verbose=True)

        # 初始化可视化（放在最后，确保其他初始化完成）
        if self.enable_visualization:
            try:
                self.visualizer = TrainingVisualizer()
            except Exception as e:
                print(f"Warning: Failed to initialize visualizer: {e}")
                self.visualizer = None
                self.enable_visualization = False
        else:
            self.visualizer = None

    def fit(self, train_loader, val_loader, num_epochs=10):
        best_f1 = 0.0
        start_time = time.time()
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Enable visualization: {self.enable_visualization}")
        print("-" * 80)

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # 训练阶段
            self.model.train()
            running_loss = 0.0
            batch_count = 0

            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                
                # 确保输出和标签形状匹配
                if outputs.shape != labels.shape:
                    print(f"Shape mismatch: outputs {outputs.shape}, labels {labels.shape}")
                    continue
                
                loss = self.criterion(outputs, labels)
                loss.backward()
                
                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()

                running_loss += loss.item()
                batch_count += 1
                
                # 每20个batch打印一次进度
                if (batch_idx + 1) % 20 == 0:
                    avg_loss_so_far = running_loss / batch_count
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx+1}/{len(train_loader)} | "
                          f"Loss: {avg_loss_so_far:.4f} | LR: {current_lr:.2e}")

            train_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0.0

            # 验证阶段
            val_f1 = self.validate(val_loader)
            epoch_time = time.time() - epoch_start_time

            # 更新学习率
            self.scheduler.step(val_f1)

            print(f"Epoch {epoch + 1}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val F1: {val_f1:.4f} | "
                  f"Time: {epoch_time:.2f}s")
            print("-" * 80)

            # 更新可视化（添加错误处理）
            if self.enable_visualization and self.visualizer is not None:
                try:
                    self.visualizer.update_plots(epoch + 1, train_loss, val_f1)
                except Exception as e:
                    print(f"Warning: Failed to update visualization: {e}")
                    self.enable_visualization = False
                    self.visualizer = None

            # 保存最佳模型
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_f1': best_f1,
                    'train_loss': train_loss
                }, os.path.join(self.save_path, 'best_model.pth'))
                print(f"*** New best model saved with F1: {val_f1:.4f} ***")

            # 每5个epoch保存一次检查点
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(self.save_path, f'checkpoint_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_f1': best_f1,
                    'train_loss': train_loss
                }, checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")

        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f}s")
        print(f"Best validation F1 score: {best_f1:.4f}")

        # 保存最终图表
        if self.enable_visualization and self.visualizer is not None:
            try:
                self.visualizer.save_plot(os.path.join(self.save_path, 'training_progress.png'))
                self.visualizer.close()
            except Exception as e:
                print(f"Warning: Failed to save final plot: {e}")

    def validate(self, val_loader):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)

                preds = torch.sigmoid(outputs).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())

        if len(all_preds) == 0:
            return 0.0

        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        # 使用0.5作为阈值
        all_preds_bin = (all_preds > 0.5).astype(int)
        all_labels_bin = (all_labels > 0.5).astype(int)

        # 计算macro F1分数
        try:
            f1 = f1_score(all_labels_bin, all_preds_bin, average='macro', zero_division=0)
        except Exception as e:
            print(f"Error calculating F1 score: {e}")
            f1 = 0.0
            
        return f1

# ------------------- 主程序 -------------------
class Config:
    data_dir = './FERPlus_dataset'
    batch_size = 200
    learning_rate = 1e-4
    num_epochs = 100

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    config = Config()

    # 创建保存目录
    os.makedirs('./checkpoints', exist_ok=True)

    # 初始化模型
    model = ResNetFER(
        num_classes=8,
        pretrained_path='./weight/res50_ir_0.887.pth' if os.path.exists('./weight/res50_ir_0.887.pth') else None,
        device=device
    )
    model.init()

    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders(config, mode='multi_label')

    # 创建训练器并开始训练
    trainer = FERTrainer(
        model=model,
        device=device,
        save_path='./checkpoints',
        learning_rate=config.learning_rate,
        enable_visualization=True
    )

    try:
        trainer.fit(train_loader, val_loader, num_epochs=config.num_epochs)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
