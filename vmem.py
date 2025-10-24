import os
import re
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

# ================ 1. 内存文件处理 ================

def split_vmem_to_hex(input_file_path, output_dir="vmem_output", skip_all_zero=True):
    """将vmem内存文件按4K块分割，并将每个块保存为16进制文本文件"""
    # 获取文件名（不包含扩展名和路径）
    file_name = os.path.splitext(os.path.basename(input_file_path))[0]
    
    try:
        # 以二进制模式打开vmem文件
        with open(input_file_path, 'rb') as f:
            block_size = 4096  # 4K块大小
            block_number = 1  # 块号从1开始
            written_blocks = 0
            
            # 确保输出目录存在
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 循环读取文件直到结束
            while True:
                # 读取4K数据
                data = f.read(block_size)
                
                # 如果没有数据，说明已到达文件末尾
                if not data:
                    break
                
                # 检查是否全为0
                if skip_all_zero and all(byte == 0 for byte in data):
                    block_number += 1
                    continue
                
                # 转换为16进制字符串
                hex_data = data.hex()
                
                # 按每16字节(32个字符)分割，并添加换行
                formatted_hex = ''
                for i in range(0, len(hex_data), 32):
                    formatted_hex += hex_data[i:i+32] + '\n'
                
                # 生成输出文件名
                output_file = os.path.join(output_dir, f"{file_name}-{block_number}.txt")
                
                # 写入16进制数据到文本文件
                with open(output_file, 'w') as out_f:
                    out_f.write(formatted_hex)
                
                block_number += 1
                written_blocks += 1
        
        return written_blocks, block_number - 1  # 返回实际处理的块数
        
    except Exception as e:
        print(f"处理文件 {input_file_path} 时出错: {e}")
        return 0, 0

# ================ 2. 数据预处理 ================

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

# ================ 3. 自定义数据集 ================

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

# ================ 4. 构建一维CNN模型 ================

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

# ================ 5. 训练和评估函数 ================

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
    """优化后的预测函数，适应无标签数据"""
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        for batch in data_loader:  # 处理无标签的批次数据
            inputs = batch[0].to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
    
    return np.array(all_probs)

# ================ 6. 检测vmem文件的函数（优化内存版本） ================

def detect_vmem_file(vmem_path, model, device, threshold=0.5, batch_size=128):
    """
    检测vmem文件中的恶意代码片段比例，优化了内存使用
    
    参数:
    vmem_path (str): vmem文件路径
    model: 训练好的模型
    device: 计算设备
    threshold: 恶意判定阈值
    batch_size: 批处理大小，可根据内存调整
    """
    # 临时输出目录
    temp_output_dir = "temp_vmem_blocks"
    
    # 将vmem文件分割为4K块的hex文件
    print(f"正在处理vmem文件: {vmem_path}")
    written_blocks, total_blocks = split_vmem_to_hex(vmem_path, temp_output_dir)
    
    if written_blocks == 0:
        print("未生成任何有效块文件")
        return 0.0, 0, 0
    
    print(f"共生成 {written_blocks} 个有效块文件 (总块数: {total_blocks})")
    
    # 获取所有块文件路径（不立即加载全部数据）
    block_files = [
        os.path.join(temp_output_dir, file) 
        for file in os.listdir(temp_output_dir) 
        if file.endswith('.txt')
    ]
    total_count = len(block_files)
    malicious_count = 0
    
    # 分批处理文件，减少内存占用
    for i in tqdm(range(0, total_count, batch_size), desc="检测恶意代码片段"):
        # 加载当前批次的文件
        batch_files = block_files[i:i+batch_size]
        X_batch = []
        
        for file_path in batch_files:
            X_batch.append(hex_file_to_bytes(file_path))
        
        # 转换为模型输入格式
        X_batch = np.array(X_batch, dtype=np.float32)
        X_batch = X_batch / 255.0  # 归一化
        X_batch = np.expand_dims(X_batch, axis=1)  # 添加通道维度
        
        # 创建临时数据集和加载器
        temp_dataset = TensorDataset(torch.tensor(X_batch, dtype=torch.float32))
        temp_loader = DataLoader(temp_dataset, batch_size=batch_size)
        
        # 预测当前批次
        batch_probs = predict(model, temp_loader, device)
        malicious_count += np.sum(batch_probs > threshold)
        
        # 手动释放内存
        del X_batch, temp_dataset, temp_loader, batch_probs
        if device.type == 'cuda':
            torch.cuda.empty_cache()  # 清理GPU内存
    
    # 计算恶意片段比例
    malicious_ratio = malicious_count / total_count if total_count > 0 else 0.0
    
    # 清理临时文件
    for file in os.listdir(temp_output_dir):
        os.remove(os.path.join(temp_output_dir, file))
    os.rmdir(temp_output_dir)
    
    return malicious_ratio, total_count, malicious_count

# ================ 7. 主函数 ================

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 尝试加载已训练的模型，如果不存在则训练新模型
    model_path = 'best_pytorch_model.pth'
    model = MalwareCNN(input_channels=1, num_classes=2).to(device)
    
    if os.path.exists(model_path):
        print("加载已训练的模型...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("未找到已训练的模型，开始训练新模型...")
        # 创建TensorBoard写入器
        writer = SummaryWriter('runs/malware_cnn')
        
        # 加载训练数据（良性和恶意样本）
        print("加载训练数据集...")
        benign_dir = r"F:\page\benpage"  # 良性样本目录
        malware_dir = r"F:\page\malpage"  # 恶意样本目录
        
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
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
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
            model, train_loader, val_loader, criterion, optimizer, scheduler, 
            device, epochs=50, writer=writer
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
    
    # 检测vmem文件
    vmem_file_path = input("请输入要检测的vmem文件路径: ").strip()
    if not os.path.exists(vmem_file_path) or not vmem_file_path.endswith('.vmem'):
        print("无效的vmem文件路径")
        return
    
    # 设置检测阈值
    try:
        threshold_input = input("请输入检测阈值(0-1之间，默认0.5): ").strip()
        threshold = float(threshold_input) if threshold_input else 0.5
        if not (0 <= threshold <= 1):
            raise ValueError
    except ValueError:
        print("无效的阈值，使用默认值0.5")
        threshold = 0.5
    
    # 设置批次大小（可根据内存情况调整）
    try:
        batch_input = input("请输入批处理大小(默认128，内存不足可减小): ").strip()
        batch_size = int(batch_input) if batch_input else 128
        if batch_size <= 0:
            raise ValueError
    except ValueError:
        print("无效的批次大小，使用默认值128")
        batch_size = 128
    
    malicious_ratio, total_count, malicious_count = detect_vmem_file(
        vmem_file_path, model, device, threshold, batch_size
    )
    
    # 输出检测结果
    print("\n===== 检测结果 =====")
    print(f"总代码片段数: {total_count}")
    print(f"检测到的恶意代码片段数: {malicious_count}")
    print(f"恶意代码片段比例: {malicious_ratio:.2%}")

if __name__ == "__main__":
    main()
