import os
import re
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

# ================ 1. 数据预处理 ================

def clean_hex_line(line):
    """清理16进制字符串，过滤非16进制字符并转为小写"""
    return re.sub(r'[^0-9a-fA-F]', '', line.strip().lower())

def hex_file_to_bytes(file_path, max_length=4096):
    """将16进制文本文件转换为字节数组"""
    bytes_list = []
    with open(file_path, 'r') as f:
        for line in f:
            clean_line = clean_hex_line(line)
            if not clean_line:  # 跳过空行
                continue
            # 每行应包含32个字符(16字节)，不足则补零
            if len(clean_line) < 32:
                clean_line = clean_line.ljust(32, '0')
            # 每2个字符转换为1个字节
            bytes_list.extend([int(clean_line[i:i+2], 16) for i in range(0, 32, 2)])
    
    # 截断或填充至固定长度
    if len(bytes_list) > max_length:
        return np.array(bytes_list[:max_length], dtype=np.uint8)
    else:
        return np.pad(bytes_list, (0, max_length - len(bytes_list)), mode='constant')

# ================ 2. 自定义数据集 ================

class HexDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        sample = self.X[idx]
        
        # 归一化处理
        sample = sample / 255.0
        
        # 添加通道维度 (4096,) -> (1, 4096)
        sample = np.expand_dims(sample, axis=0)
        
        if self.transform:
            sample = self.transform(sample)
            
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

# ================ 3. 构建一维CNN模型 ================

class MalwareCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=2):
        super(MalwareCNN, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )
        
        # 计算全连接层输入维度
        fc_input_size = self._calculate_fc_input_size(input_channels, 4096)
        
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(fc_input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def _calculate_fc_input_size(self, in_channels, seq_length):
        x = torch.rand(1, in_channels, seq_length)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x.view(1, -1).size(1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        return x

# ================ 4. 训练和评估函数 ================

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=50, writer=None):
    best_val_auc = 0.0
    best_model = None
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100.0 * correct / total
        train_loss /= len(train_loader)
        
        # 验证阶段
        val_loss, val_auc = evaluate_model(model, val_loader, criterion, device)
        
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 更新学习率
        old_lr = current_lr
        scheduler.step(val_auc)
        new_lr = optimizer.param_groups[0]['lr']
        
        # 打印学习率变化
        if new_lr < old_lr:
            print(f"Epoch {epoch+1}: 学习率从 {old_lr:.6f} 调整为 {new_lr:.6f}")
        
        # 保存最佳模型
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model = model.state_dict().copy()
            torch.save(best_model, 'best_pytorch_model.pth')
        
        print(f'Epoch {epoch+1}/{epochs} - '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, LR: {new_lr:.6f}')
        
        # 写入TensorBoard
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('AUC/val', val_auc, epoch)
            writer.add_scalar('LearningRate', new_lr, epoch)
    
    return best_model

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc='Evaluating'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # 获取预测概率
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
    
    # 计算AUC
    auc = roc_auc_score(all_labels, all_probs)
    avg_loss = total_loss / len(data_loader)
    
    return avg_loss, auc

def predict(model, data_loader, device):
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
    
    return np.array(all_probs)

# ================ 5. 主函数 ================

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 创建TensorBoard写入器
    writer = SummaryWriter('runs/malware_cnn')
    
    # 加载数据
    print("Loading dataset...")
    benign_dir = r"F:\page\benpage"
    malware_dir = r"F:\page\malpage"
    
    # 加载数据集
    X, y = [], []
    
    # 加载良性样本
    print("加载良性样本...")
    for file in tqdm(os.listdir(benign_dir)):
        if file.endswith('.txt'):
            file_path = os.path.join(benign_dir, file)
            X.append(hex_file_to_bytes(file_path))
            y.append(0)  # 良性标签为0
    
    # 加载恶意样本
    print("加载恶意样本...")
    for file in tqdm(os.listdir(malware_dir)):
        if file.endswith('.txt'):
            file_path = os.path.join(malware_dir, file)
            X.append(hex_file_to_bytes(file_path))
            y.append(1)  # 恶意标签为1
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    
    # 数据划分
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"训练集形状: {X_train.shape}, 标签分布: {np.bincount(y_train)}")
    print(f"验证集形状: {X_val.shape}, 标签分布: {np.bincount(y_val)}")
    print(f"测试集形状: {X_test.shape}, 标签分布: {np.bincount(y_test)}")
    
    # 创建数据加载器
    train_dataset = HexDataset(X_train, y_train)
    val_dataset = HexDataset(X_val, y_val)
    test_dataset = HexDataset(X_test, y_test)
    
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 初始化模型
    model = MalwareCNN(input_channels=1, num_classes=2).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # 创建学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5
    )
    
    # 训练模型
    print("开始训练模型...")
    best_model = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=50, writer=writer
    )
    
    # 关闭TensorBoard写入器
    writer.close()
    
    # 加载最佳模型并在测试集上评估
    model.load_state_dict(best_model)
    test_loss, test_auc = evaluate_model(model, test_loader, criterion, device)
    print(f'测试集 AUC: {test_auc:.4f}')
    
    # 获取预测结果
    y_pred_prob = predict(model, test_loader, device)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # 计算评估指标
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # 打印评估结果
    print("\n测试集结果:")
    print(f"准确率: {(tp + tn) / (tp + tn + fp + fn):.4f}")
    print(f"精确率: {tp / (tp + fp):.4f}")
    print(f"召回率: {tp / (tp + fn):.4f}")
    print(f"F1分数: {2 * (tp / (tp + fp)) * (tp / (tp + fn)) / ((tp / (tp + fp)) + (tp / (tp + fn))):.4f}")
    print("\n混淆矩阵:")
    print(cm)
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['良性', '恶意']))

if __name__ == "__main__":
    main()