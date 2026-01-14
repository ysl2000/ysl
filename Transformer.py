import os
import math
import numpy as np
import pandas as pd
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
from torch.utils.tensorboard import SummaryWriter
import pickle

# 设置环境变量以优化CUDA性能和调试
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 同步执行CUDA，便于调试
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'  # 限制内存分配块大小
os.environ['TORCH_USE_CUDA_DSA'] = '1'    # 启用设备端断言

# 设置随机种子确保可复现性
def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# 1. 增强数据预处理 - 考虑频率和上下文信息
def read_opcode_files(data_dir, label):
    data = []
    labels = []
    all_opcodes = set()
    opcode_freq = Counter()
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"目录不存在: {data_dir}")
    
    file_count = 0
    for file_name in os.listdir(data_dir):
        if not file_name.endswith('.txt'):
            continue
            
        file_path = os.path.join(data_dir, file_name)
        try:
            with open(file_path, 'r') as f:
                opcodes = []
                for line in f:
                    opcode = line.strip()
                    if opcode:  # 过滤空行
                        opcodes.append(opcode)
                        all_opcodes.add(opcode)
                        opcode_freq[opcode] += 1
                if opcodes:  # 确保文件不为空
                    data.append(opcodes)
                    labels.append(label)
                    file_count += 1
                    if file_count % 100 == 0:
                        print(f"已读取 {file_count} 个文件")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    return data, labels, all_opcodes, opcode_freq

# 操作码嵌入 - 考虑语义关系
def create_opcode_embedding(all_opcodes, embed_dim=32):
    # 使用Word2Vec学习操作码嵌入
    from gensim.models import Word2Vec
    
    dummy_sequences = [[opcode] for opcode in all_opcodes]
    
    # 训练Word2Vec模型
    print("Training Word2Vec model...")
    model = Word2Vec(sentences=dummy_sequences, vector_size=embed_dim, window=5, min_count=1, workers=4)
    
    # 创建嵌入矩阵
    embedding_matrix = np.zeros((len(all_opcodes), embed_dim))
    opcode_to_index = {opcode: i for i, opcode in enumerate(all_opcodes)}
    
    print("Creating embedding matrix...")
    for opcode, i in opcode_to_index.items():
        if opcode in model.wv:
            embedding_matrix[i] = model.wv[opcode]
        else:
            embedding_matrix[i] = np.random.randn(embed_dim) * 0.01
    
    return embedding_matrix, opcode_to_index

# 3. 简化版Transformer架构模型
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=2*d_model,  # 减小前馈网络维度
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.d_model = d_model
        
    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, 
                 nhead=2, num_transformer_layers=1, dropout=0.3):
        super().__init__()
        
        self.transformer = TransformerEncoder(
            input_dim=vocab_size,
            d_model=embed_dim,
            nhead=nhead,
            num_layers=num_transformer_layers,
            dropout=dropout
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, num_classes)
        )
        
        self.dropout = dropout
        
    def forward(self, x):
        transformer_output = self.transformer(x)
        # 取序列的第一个位置的输出作为整个序列的表示
        pooled_output = transformer_output[:, 0, :]
        logits = self.classifier(pooled_output)
        return logits

# 4. 优化的数据集类
class OpcodeDataset(Dataset):
    def __init__(self, data, labels, opcode_to_index, config):
        self.data = []
        self.labels = labels
        self.config = config
        
        max_len = config.get('max_seq_len', 300)
        
        print(f"Creating dataset with {len(data)} samples...")
        for i, opcodes in enumerate(data):
            # 打印特定样本的调试信息
            if i == 10:  # 检查第10个样本（可调整）
                print(f"Sample {i} original length: {len(opcodes)}")
            
            # 截断长序列以避免内存溢出和计算超时
            opcodes = opcodes[:max_len]
            
            indices = [opcode_to_index.get(op, 0) for op in opcodes]
            
            if len(indices) < max_len:
                indices = indices + [0] * (max_len - len(indices))
            
            self.data.append(torch.tensor(indices, dtype=torch.long))
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

# 数据加载器的collate函数
def collate_fn(batch):
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    return torch.stack(inputs), torch.tensor(targets)

# 5. 训练和评估函数
def train(model, train_loader, criterion, optimizer, device, use_amp=True, accumulation_steps=2):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    optimizer.zero_grad()
    
    for i, (inputs, targets) in enumerate(train_loader):
        # 打印批次进度，监控训练速度
        if i % 10 == 0:
            print(f"Batch {i}/{len(train_loader)}")
        
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 检查输入数据是否包含NaN或inf
        if torch.isnan(inputs).any() or torch.isinf(inputs).any():
            print(f"Warning: Input data contains NaN or inf at batch {i}")
            # 可以选择跳过这个批次，或者进行更详细的调试
            continue
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(inputs)
            
            # 检查模型输出是否包含NaN或inf
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print(f"Warning: Model outputs contain NaN or inf at batch {i}")
                # 可以选择跳过这个批次，或者进行更详细的调试
                continue
            
            loss = criterion(outputs, targets)
            
            # 检查损失值是否正常
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Loss is NaN or inf at batch {i}")
                # 可以选择跳过这个批次，或者进行更详细的调试
                continue
            
            loss = loss / accumulation_steps
            
        scaler.scale(loss).backward()
        
        # 检查梯度是否包含NaN或inf
        has_nan_or_inf = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"Warning: Gradient for {name} contains NaN or inf at batch {i}")
                    has_nan_or_inf = True
        
        if has_nan_or_inf:
            # 梯度异常，跳过本次更新
            optimizer.zero_grad(set_to_none=True)
            print(f"Skipping optimizer step due to invalid gradients at batch {i}")
            continue
        
        if (i + 1) % accumulation_steps == 0:
            # 在调用scaler.step之前，检查是否有梯度更新
            grads_exist = False
            for param in model.parameters():
                if param.grad is not None:
                    grads_exist = True
                    break
            
            if not grads_exist:
                print(f"Warning: No gradients exist for optimizer step at batch {i}")
                optimizer.zero_grad(set_to_none=True)
                continue
            
            try:
                # 尝试执行优化器步骤
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                torch.cuda.empty_cache()  # 释放缓存内存
            except AssertionError as e:
                print(f"Error: {e} at batch {i}")
                print("Skipping optimizer step and resetting gradients")
                optimizer.zero_grad(set_to_none=True)
                # 降低学习率作为恢复措施
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
                    print(f"Reduced learning rate to {param_group['lr']}")
                continue
        
        total_loss += loss.item() * accumulation_steps
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    # 处理剩余的梯度累积步骤
    if i % accumulation_steps != 0:
        # 同样进行梯度检查
        grads_exist = False
        for param in model.parameters():
            if param.grad is not None:
                grads_exist = True
                break
        
        if grads_exist:
            try:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
            except AssertionError as e:
                print(f"Error: {e} during final optimizer step")
                optimizer.zero_grad(set_to_none=True)
                # 降低学习率作为恢复措施
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
                    print(f"Reduced learning rate to {param_group['lr']}")
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy

def evaluate(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 使用半精度进行推理以节省内存
            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(inputs)
            
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            torch.cuda.empty_cache()  # 释放推理过程中的缓存
    
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    f1 = f1_score(all_targets, all_preds, average='weighted')
    
    return accuracy, precision, recall, f1

def main():
    # 直接在代码中设置参数，无需命令行
    config = {
        'benign_dir': r'F:\OPGCN\benop',
        'malicious_dir': r'F:\OPGCN\malop',
        'save_path': 'transformer_model.pth',
        'save_embedding': False,
        'batch_size': 16,             # 进一步减小批处理大小
        'epochs': 10,
        'lr': 0.001,
        'lrf': 0.1,
        'weight_decay': 5e-4,
        'embed_dim': 32,            # 减小嵌入维度
        'hidden_dim': 32,           # 减小隐藏层维度
        'nhead': 2,                 # 减少注意力头数
        'num_transformer_layers': 1,
        'dropout': 0.3,             # 略微降低dropout
        'test_size': 0.2,
        'seed': 42,
        'use_amp': True,            # 启用混合精度训练
        'accumulation_steps': 2,    # 梯度累积，模拟更大batch
        'num_workers': 0,           # 禁用多线程数据加载
        'use_tensorboard': False,
        'max_seq_len': 10000,         # 截断长序列
    }
    
    # 设置随机种子
    if config['seed']:
        seed_torch(config['seed'])
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
    
    # 检查数据集路径是否存在
    for path in [config['benign_dir'], config['malicious_dir']]:
        if not os.path.exists(path):
            print(f"错误: 指定的路径不存在 - {path}")
            print(f"当前工作目录: {os.getcwd()}")
            print("可用目录:")
            try:
                for item in os.listdir(os.path.dirname(path)):
                    print(f"  - {item}")
            except Exception as e:
                print(f"无法列出目录内容: {e}")
            raise FileNotFoundError(f"目录不存在: {path}")
    
    # 读取数据
    print("Reading benign data...")
    benign_data, benign_labels, all_opcodes, opcode_freq = read_opcode_files(
        config['benign_dir'], 0
    )
    
    print("Reading malicious data...")
    malicious_data, malicious_labels, _, _ = read_opcode_files(
        config['malicious_dir'], 1
    )
    
    # 合并数据
    all_data = benign_data + malicious_data
    all_labels = benign_labels + malicious_labels
    
    # 创建操作码嵌入
    print("Creating opcode embeddings...")
    embedding_matrix, opcode_to_index = create_opcode_embedding(all_opcodes, config['embed_dim'])
    
    # 保存嵌入矩阵和映射字典供以后使用
    if config['save_embedding']:
        with open('opcode_embedding.pkl', 'wb') as f:
            pickle.dump({
                'embedding_matrix': embedding_matrix,
                'opcode_to_index': opcode_to_index
            }, f)
    
    # 划分训练集和测试集
    print("Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        all_data, all_labels, test_size=config['test_size'], random_state=config['seed'], stratify=all_labels
    )
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # 创建数据集和数据加载器
    print("Starting to create train dataset...")
    start_time = time.time()
    train_dataset = OpcodeDataset(X_train, y_train, opcode_to_index, config)
    print(f"Train dataset created in {time.time() - start_time:.2f} seconds")
    
    print("Starting to create test dataset...")
    start_time = time.time()
    test_dataset = OpcodeDataset(X_test, y_test, opcode_to_index, config)
    print(f"Test dataset created in {time.time() - start_time:.2f} seconds")
    
    print("Creating dataloaders...")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )
    print("Dataloaders created successfully")
    
    # 初始化模型
    print("Initializing model...")
    model = TransformerModel(
        vocab_size=len(opcode_to_index),
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        num_classes=2,
        nhead=config['nhead'],
        num_transformer_layers=config['num_transformer_layers'],
        dropout=config['dropout']
    ).to(device)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(pg, lr=config['lr'], weight_decay=config['weight_decay'])
    
    # 学习率调度
    lf = lambda x: ((1 + math.cos(x * math.pi / config['epochs'])) / 2) * (1 - config['lrf']) + config['lrf']
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    
    # 训练循环
    best_acc = 0.0
    print("Starting training...")
    for epoch in range(config['epochs']):
        start_time = time.time()
        
        # 训练
        train_loss, train_acc = train(
            model, train_loader, criterion, optimizer, device, 
            config['use_amp'], config['accumulation_steps']
        )
        
        # 评估
        test_acc, test_prec, test_rec, test_f1 = evaluate(model, test_loader, device)
        
        # 学习率更新
        scheduler.step()
        
        # 记录最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), config['save_path'])
            print(f"Model saved with accuracy: {test_acc:.4f}")
        
        # 打印训练进度
        end_time = time.time()
        epoch_time = end_time - start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'Epoch [{epoch+1}/{config["epochs"]}] | '
              f'Time: {epoch_time:.2f}s | '
              f'LR: {current_lr:.6f} | '
              f'Train Loss: {train_loss:.4f} | '
              f'Train Acc: {train_acc:.2f}% | '
              f'Test Acc: {test_acc:.2f}% | '
              f'Test Prec: {test_prec:.4f} | '
              f'Test Rec: {test_rec:.4f} | '
              f'Test F1: {test_f1:.4f}')
        
        # 打印GPU内存使用情况
        if device.type == 'cuda':
            print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            print(f"GPU memory cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    
    # 加载最佳模型并进行最终评估
    model.load_state_dict(torch.load(config['save_path']))
    final_acc, final_prec, final_rec, final_f1 = evaluate(model, test_loader, device)
    
    print(f'\nFinal Test Results:')
    print(f'Accuracy: {final_acc:.4f}')
    print(f'Precision: {final_prec:.4f}')
    print(f'Recall: {final_rec:.4f}')
    print(f'F1-score: {final_f1:.4f}')

# 直接执行主函数
if __name__ == '__main__':
    main()