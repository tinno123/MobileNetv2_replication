import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm  # 需要安装: pip install timm
from PIL import Image
import os
from pathlib import Path
from sklearn.model_selection import train_test_split


class ImageClassificationDataset(Dataset):
    """图像分类数据集"""

    def __init__(self, image_dir, label_dir, transform=None, mode='train'):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        self.mode = mode

        self.samples = []
        self._load_samples()

    def _load_samples(self):
        """加载样本"""
        for label_file in self.label_dir.glob('*.txt'):
            # 查找对应的图片
            image_path = self.image_dir / (label_file.stem + '.jpg')
            if not image_path.exists():
                image_path = self.image_dir / (label_file.stem + '.png')
            if not image_path.exists():
                continue

            # 读取标签（假设txt里是类别ID）
            with open(label_file, 'r') as f:
                content = f.read().strip()
                if content:
                    class_id = int(content.split()[0])  # 只取第一个数字
                    self.samples.append({
                        'image_path': str(image_path),
                        'label': class_id
                    })

        print(f"Loaded {len(self.samples)} samples for {self.mode}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')
        label = sample['label']

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(dataloader), 100. * correct / total


def main():
    # ========== 配置参数 ==========
    image_dir = r"E:\hands\yolo\images\train"  # 图片目录
    label_dir = r"E:\hands\yolo\labels\train"  # txt标签目录
    num_classes = 5  # 手型类别数
    batch_size = 32
    epochs = 50
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ========== 数据预处理 ==========
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ========== 加载数据集 ==========
    full_dataset = ImageClassificationDataset(image_dir, label_dir, transform=None)

    # 划分训练集和验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    # 应用不同的transform
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # ========== 创建模型 ==========
    # 使用timm加载预训练模型
    model = timm.create_model('mobilenetv2_100', pretrained=True, num_classes=num_classes)
    model = model.to(device)

    # 可选：冻结主干
    # for param in model.parameters():
    #     param.requires_grad = False
    # for param in model.classifier.parameters():
    #     param.requires_grad = True

    # ========== 损失函数和优化器 ==========
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ========== 训练 ==========
    best_acc = 0
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print("-" * 50)

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved best model with acc: {best_acc:.2f}%")

    print(f"Training finished! Best accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()