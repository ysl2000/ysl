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
import torch_geometric
from torch_geometric.nn import GCNConv, global_max_pool
import scipy.sparse as sp
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

# 2. 改进的图构建策略 - 考虑语义关系（优化版本）
def build_semantic_graph(opcode_indices, opcode_to_index, embedding_matrix, window_size=2, threshold=0.6):
    num_nodes = len(opcode_indices)
    if num_nodes == 0:
        return sp.csr_matrix((0, 0))
    
    # 对于长序列，进一步减少计算量
    if num_nodes > 500:
        window_size = 1
    
    # 使用lil_matrix进行高效构建
    adj_matrix = sp.lil_matrix((num_nodes, num_nodes))
    
    # 1. 添加顺序关系
    for i in range(num_nodes - 1):
        adj_matrix[i, i+1] = 1
        adj_matrix[i+1, i] = 1
    
    # 2. 添加语义关系
    for i in range(num_nodes):
        # 进度打印
        if i % 100 == 0 and num_nodes > 100:
            print(f"Building graph: {i}/{num_nodes}")
            
        current_embed = embedding_matrix[opcode_indices[i]]
        start = max(0, i - window_size)
        end = min(num_nodes, i + window_size + 1)
        
        for j in range(start, end):
            if i == j:
                continue
                
            other_embed = embedding_matrix[opcode_indices[j]]
            # 使用点积代替余弦相似度，减少计算量
            sim = np.dot(current_embed, other_embed)
            
            if sim > threshold:
                adj_matrix[i, j] = sim
    
    # 转换为csr_matrix用于后续计算
    adj_matrix = adj_matrix.tocsr()
    
    # 确保邻接矩阵不包含越界索引
    if adj_matrix.nnz > 0:
        adj_matrix = normalize_adj(adj_matrix)
    
    return adj_matrix

def normalize_adj(adj):
    adj = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocsr()

# 3. 简化版混合架构模型 - Transformer + GNN（修改版）
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

class MixedModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, 
                 nhead=2, num_transformer_layers=1, num_gcn_layers=1, dropout=0.3):
        super().__init__()
        
        self.transformer = TransformerEncoder(
            input_dim=vocab_size,
            d_model=embed_dim,
            nhead=nhead,
            num_layers=num_transformer_layers,
            dropout=dropout
        )
        
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNConv(embed_dim, hidden_dim))
        
        for _ in range(num_gcn_layers - 1):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        self.node_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.Tanh(),
            nn.Linear(hidden_dim//2, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, num_classes)
        )
        
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        transformer_output = self.transformer(x)
        batch_size, seq_len, embed_dim = transformer_output.shape
        
        # 创建掩码，标记哪些是真实节点（非填充）
        mask = (x != 0).any(dim=-1).unsqueeze(-1).float()  # [batch_size, seq_len, 1]
        
        batch_edge_indices = []
        batch_node_features = []
        batch_indices = []
        batch_offsets = []
        
        # 计算每个样本的实际节点数
        actual_nodes_per_sample = mask.sum(dim=1).squeeze().long()
        
        # 构建批次图
        offset = 0
        for i in range(batch_size):
            node_features = transformer_output[i] * mask[i]  # 应用掩码
            actual_nodes = actual_nodes_per_sample[i].item()
            
            # 打印调试信息
            if i == 0 and self.training:  # 只在训练时打印第一个样本的信息
                print(f"Sample {i} in batch:")
                print(f"  Input shape: {x[i].shape}, non-zero nodes: {actual_nodes}")
                print(f"  Original edge_index shape: {edge_index.shape}")
                if edge_index.numel() > 0:
                    print(f"  Edge index min: {edge_index.min().item()}, max: {edge_index.max().item()}")
            
            # 过滤掉指向填充节点的边
            if edge_index.numel() > 0:
                valid_edges_mask = (edge_index[0] < actual_nodes) & (edge_index[1] < actual_nodes)
                valid_edge_index = edge_index[:, valid_edges_mask]
                
                if valid_edge_index.numel() == 0:
                    # 如果没有有效边，创建自环
                    valid_edge_index = torch.arange(actual_nodes, device=edge_index.device).repeat(2, 1)
                    print(f"Warning: Sample {i} has no valid edges, added self-loops")
            else:
                # 空边索引，创建自环
                valid_edge_index = torch.arange(actual_nodes, device=edge_index.device).repeat(2, 1)
                print(f"Warning: Sample {i} has empty edge index, added self-loops")
            
            edge_index_i = valid_edge_index.clone()
            edge_index_i += offset  # 使用实际偏移量
            
            batch_edge_indices.append(edge_index_i)
            batch_node_features.append(node_features[:actual_nodes])  # 只保留真实节点
            batch_indices.extend([i] * actual_nodes)
            batch_offsets.append(offset)
            
            offset += actual_nodes
        
        x = torch.cat(batch_node_features, dim=0)
        edge_index = torch.cat(batch_edge_indices, dim=1)
        batch = torch.tensor(batch_indices, device=x.device)
        
        # 验证合并后的边索引
        if edge_index.numel() > 0:
            max_idx = edge_index.max().item()
            if max_idx >= offset:
                print(f"ERROR: Model edge index has invalid value: {max_idx} (total nodes: {offset})")
                print(f"Batch offsets: {batch_offsets}")
                print(f"Actual nodes per sample: {actual_nodes_per_sample.cpu().numpy()}")
                # 添加安全措施，防止越界
                mask = (edge_index[0] < offset) & (edge_index[1] < offset)
                edge_index = edge_index[:, mask]
                print(f"Fixed edge index shape: {edge_index.shape}")
        
        # GCN层处理
        for i, conv in enumerate(self.gcn_layers):
            x = conv(x, edge_index)
            if i != len(self.gcn_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 注意力机制
        attn_weights = self.node_attn(x)
        attn_weights = F.softmax(attn_weights, dim=0)
        x = x * attn_weights
        
        x = global_max_pool(x, batch)
        logits = self.classifier(x)
        
        return logits

# 4. 优化的数据集类（添加图验证）
class OpcodeDataset(Dataset):
    def __init__(self, data, labels, opcode_to_index, embedding_matrix, config):
        self.data = []
        self.labels = labels
        self.config = config
        
        max_len = config.get('max_seq_len', 300)
        window_size = config.get('window_size', 2)
        threshold = config.get('threshold', 0.6)
        
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
            
            # 计算实际非零节点数
            actual_nodes = (torch.tensor(indices) != 0).sum().item()
            
            adj_matrix = build_semantic_graph(
                indices, 
                opcode_to_index, 
                embedding_matrix,
                window_size=window_size,
                threshold=threshold
            )
            
            edge_index = torch_geometric.utils.from_scipy_sparse_matrix(adj_matrix)[0]
            
            # 验证和修复edge_index
            if edge_index.numel() > 0:
                max_idx = edge_index.max().item()
                
                if max_idx >= actual_nodes:
                    # 过滤无效索引
                    mask = (edge_index[0] < actual_nodes) & (edge_index[1] < actual_nodes)
                    edge_index = edge_index[:, mask]
                    
                    # 如果过滤后为空，添加自环
                    if edge_index.numel() == 0:
                        edge_index = torch.arange(actual_nodes).repeat(2, 1)
                        print(f"Warning: sample {i} generated invalid graph, added self-loops (actual nodes: {actual_nodes})")
            else:
                # 空图处理：添加自环
                if actual_nodes == 0:
                    actual_nodes = len(indices)  # 如果全部为零，使用最大长度
                edge_index = torch.arange(actual_nodes).repeat(2, 1)
                print(f"Warning: sample {i} generated empty graph, added self-loops (actual nodes: {actual_nodes})")
            
            # 打印特定样本的调试信息
            if i == 10:
                print(f"Sample {i} nodes: {len(indices)}, actual nodes: {actual_nodes}")
                print(f"Sample {i} edge_index shape: {edge_index.shape}")
                if edge_index.numel() > 0:
                    print(f"Sample {i} edge_index max: {edge_index.max().item()}")
            
            self.data.append((torch.tensor(indices, dtype=torch.long), edge_index))
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x, edge_index = self.data[idx]
        y = self.labels[idx]
        return x, edge_index, y

# 数据加载器的collate函数（修改版）
def collate_fn(batch):
    inputs = [item[0] for item in batch]
    edge_indices = [item[1] for item in batch]
    targets = [item[2] for item in batch]
    
    # 计算每个样本的实际节点数（忽略填充的0）
    batch_num_nodes = []
    for x in inputs:
        num_nodes = (x != 0).sum().item()
        batch_num_nodes.append(num_nodes)
    
    # 打印批次信息
    print(f"Batch size: {len(batch)}")
    print(f"Actual nodes per sample: {batch_num_nodes}")
    
    # 简单合并边索引，不添加偏移量（模型中处理）
    if edge_indices and edge_indices[0].numel() > 0:
        merged_edge_index = torch.cat(edge_indices, dim=1)
    else:
        merged_edge_index = torch.empty((2, 0), dtype=torch.long)
    
    # 验证合并后的边索引
    if merged_edge_index.numel() > 0:
        print(f"Merged edge index shape: {merged_edge_index.shape}")
        print(f"Merged edge index min: {merged_edge_index.min().item()}, max: {merged_edge_index.max().item()}")
    
    return torch.stack(inputs), merged_edge_index, torch.tensor(targets)

# 5. 训练和评估函数（修改版：添加梯度缩放器错误处理）
def train(model, train_loader, criterion, optimizer, device, use_amp=True, accumulation_steps=2):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    optimizer.zero_grad()
    
    for i, (inputs, edge_index, targets) in enumerate(train_loader):
        # 打印批次进度，监控训练速度
        if i % 10 == 0:
            print(f"Batch {i}/{len(train_loader)}")
        
        inputs, edge_index, targets = inputs.to(device), edge_index.to(device), targets.to(device)
        
        # 检查输入数据是否包含NaN或inf
        if torch.isnan(inputs).any() or torch.isinf(inputs).any():
            print(f"Warning: Input data contains NaN or inf at batch {i}")
            # 可以选择跳过这个批次，或者进行更详细的调试
            continue
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(inputs, edge_index)
            
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
        for inputs, edge_index, targets in test_loader:
            inputs, edge_index, targets = inputs.to(device), edge_index.to(device), targets.to(device)
            
            # 使用半精度进行推理以节省内存
            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(inputs, edge_index)
            
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
        'benign_dir': r'F:\TFGCN\family\benignB',
        'malicious_dir': r'F:\TFGCN\family\Botnet',
        'save_path': 'mixed_model.pth',
        'save_embedding': False,
        'batch_size': 16,             # 进一步减小批处理大小
        'epochs': 10,
        'lr': 0.01,
        'lrf': 0.1,
        'weight_decay': 5e-4,
        'embed_dim': 32,            # 减小嵌入维度
        'hidden_dim': 32,           # 减小隐藏层维度
        'nhead': 2,                 # 减少注意力头数
        'num_transformer_layers': 1,
        'num_gcn_layers': 3,        # GCN层数
        'dropout': 0.1,             # 略微降低dropout
        'test_size': 0.2,
        'seed': 42,
        'use_amp': True,            # 启用混合精度训练
        'accumulation_steps': 2,    # 梯度累积，模拟更大batch
        'num_workers': 0,           # 禁用多线程数据加载
        'use_tensorboard': True,   # 启用TensorBoard
        'window_size': 5,           # 减小图构建窗口大小
        'threshold': 0.6,           # 调整相似度阈值
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
    train_dataset = OpcodeDataset(X_train, y_train, opcode_to_index, embedding_matrix, config)
    print(f"Train dataset created in {time.time() - start_time:.2f} seconds")
    
    print("Starting to create test dataset...")
    start_time = time.time()
    test_dataset = OpcodeDataset(X_test, y_test, opcode_to_index, embedding_matrix, config)
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
    model = MixedModel(
        vocab_size=len(opcode_to_index),
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        num_classes=2,
        nhead=config['nhead'],
        num_transformer_layers=config['num_transformer_layers'],
        num_gcn_layers=config['num_gcn_layers'],
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