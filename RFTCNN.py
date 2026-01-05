import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import matplotlib.pyplot as plt
from collections import Counter

# 尝试导入seaborn，若失败则用matplotlib原生绘制
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("提示：未安装seaborn，将使用matplotlib原生绘制混淆矩阵")

# ===================== 全局配置项 =====================
# 数据路径
ENHANCED_DATA_PATH = "enhanced_api_sequences.csv"

# 设备配置（优先使用GPU）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"✅ 检测到GPU：{torch.cuda.get_device_name(0)}，将使用GPU训练")
else:
    print("⚠️ 未检测到GPU，将使用CPU训练")

# TextCNN配置
MAX_SEQ_LENGTH = 10  # API序列最大长度
EMBEDDING_DIM = 10   # 嵌入维度（与Word2Vec一致）
FILTERS = 32         # 卷积核数量
KERNEL_SIZES = [2, 3]  # 卷积核大小（二元组、三元组）
DROPOUT_RATE = 0.5   # Dropout概率

# 训练配置
BATCH_SIZE = 8 if torch.cuda.is_available() else 4  # GPU用更大的批量
EPOCHS = 10
LEARNING_RATE = 1e-3

# 随机种子（保证结果可复现）
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_STATE)
    torch.cuda.manual_seed_all(RANDOM_STATE)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ===================== 1. 数据加载与预处理模块 =====================
class APIDataset(Dataset):
    """PyTorch数据集类：返回CPU张量，避免提前移至GPU"""
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # 仅转为CPU Tensor，不指定设备
        seq = torch.tensor(self.sequences[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return seq, label

def collate_fn(batch):
    """自定义批处理函数：批量将数据移至指定设备"""
    sequences, labels = zip(*batch)
    sequences = torch.stack(sequences).to(DEVICE)
    labels = torch.stack(labels).to(DEVICE)
    return sequences, labels

def load_and_preprocess_data():
    """加载并预处理数据，过滤异常值"""
    # 1. 加载数据
    df = pd.read_csv(ENHANCED_DATA_PATH, encoding="utf-8-sig")
    print(f"\n原始数据集大小：{df.shape}")

    # 2. 数据清洗：过滤空值/NaN
    df = df[
        df["api_sequence"].notna() &
        (df["api_sequence"].astype(str).str.strip() != "")
    ].reset_index(drop=True)
    print(f"清洗后数据集大小：{df.shape}")

    # 3. 提取随机森林特征
    rf_feature_cols = [
        col for col in df.columns
        if col.startswith(("seq_length", "api_count_", "w2v_"))
    ]
    X_rf = df[rf_feature_cols].values
    y = df["label"].values

    # 4. 提取TextCNN特征
    api_sequences = df["api_sequence"].str.split(",").tolist()
    # 过滤非列表序列
    api_sequences = [seq for seq in api_sequences if isinstance(seq, list) and len(seq) > 0]
    y = y[:len(api_sequences)]
    print(f"有效API序列数量：{len(api_sequences)}")

    # 构建词汇表
    all_apis = [api for seq in api_sequences for api in seq if api.strip() != ""]
    api2idx = {api: idx+1 for idx, api in enumerate(Counter(all_apis).keys())}
    vocab_size = len(api2idx)
    print(f"词汇表大小（唯一API数量）：{vocab_size}")

    # 序列转索引并统一长度
    indexed_sequences = []
    for seq in api_sequences:
        seq = [api.strip() for api in seq if api.strip() != ""]
        indexed = [api2idx.get(api, 0) for api in seq]
        if len(indexed) < MAX_SEQ_LENGTH:
            indexed += [0] * (MAX_SEQ_LENGTH - len(indexed))
        else:
            indexed = indexed[:MAX_SEQ_LENGTH]
        indexed_sequences.append(indexed)
    X_cnn = np.array(indexed_sequences)

    # 5. 划分训练集/测试集
    X_rf_train, X_rf_test, y_train, y_test = train_test_split(
        X_rf[:len(X_cnn)], y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_cnn_train, X_cnn_test, _, _ = train_test_split(
        X_cnn, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # 6. 构建DataLoader（关键：禁用pin_memory）
    train_dataset = APIDataset(X_cnn_train, y_train)
    test_dataset = APIDataset(X_cnn_test, y_test)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=False  # 彻底禁用pin_memory，解决错误
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=False
    )

    return (
        X_rf_train, X_rf_test, y_train, y_test,
        train_loader, test_loader, vocab_size, df
    )

# ===================== 2. 随机森林模型模块 =====================
def train_random_forest(X_train, X_test, y_train, y_test):
    """训练随机森林模型"""
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }
    cm = confusion_matrix(y_test, y_pred)

    return rf_model, metrics, cm

# ===================== 3. PyTorch版TextCNN模型模块 =====================
class TextCNN(nn.Module):
    """TextCNN模型类"""
    def __init__(self, vocab_size, embedding_dim, num_filters, kernel_sizes, dropout_rate):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size + 1,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=k
            ) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), 1)

    def forward(self, x):
        x_emb = self.embedding(x)
        x_emb = x_emb.permute(0, 2, 1)
        
        conv_outputs = [torch.relu(conv(x_emb)) for conv in self.convs]
        pool_outputs = [torch.max_pool1d(out, kernel_size=out.size(2)).squeeze(2) for out in conv_outputs]
        
        concat = torch.cat(pool_outputs, dim=1)
        dropout_out = self.dropout(concat)
        prob = torch.sigmoid(self.fc(dropout_out))
        
        return prob

def train_textcnn(train_loader, test_loader, vocab_size):
    """训练TextCNN模型"""
    model = TextCNN(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        num_filters=FILTERS,
        kernel_sizes=KERNEL_SIZES,
        dropout_rate=DROPOUT_RATE
    ).to(DEVICE)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    # 训练
    print("\n=== TextCNN训练过程 ===")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for batch_seq, batch_label in train_loader:
            optimizer.zero_grad()
            output = model(batch_seq).squeeze()
            loss = criterion(output, batch_label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_seq.size(0)
        
        scheduler.step()
        avg_train_loss = train_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    # 测试
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch_seq, batch_label in test_loader:
            output = model(batch_seq).squeeze()
            pred = (output > 0.5).int().cpu().numpy()
            true = batch_label.cpu().numpy()
            y_true.extend(true)
            y_pred.extend(pred)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }
    cm = confusion_matrix(y_true, y_pred)

    return model, metrics, cm

# ===================== 4. 模型融合模块 =====================
def ensemble_predict(rf_model, cnn_model, X_rf_test, test_loader):
    """融合随机森林和TextCNN的预测结果"""
    # 随机森林概率
    rf_proba = rf_model.predict_proba(X_rf_test)[:, 1]

    # TextCNN概率
    cnn_model.eval()
    cnn_proba = []
    with torch.no_grad():
        for batch_seq, _ in test_loader:
            output = cnn_model(batch_seq).squeeze().cpu().numpy()
            cnn_proba.extend(output)
    cnn_proba = np.array(cnn_proba)

    # 加权融合
    ensemble_proba = 0.5 * rf_proba + 0.5 * cnn_proba
    ensemble_pred = (ensemble_proba > 0.5).astype(int)

    return ensemble_pred

# ===================== 5. 评估与可视化模块 =====================
def plot_confusion_matrix(cm, title, ax):
    """绘制混淆矩阵"""
    if SEABORN_AVAILABLE:
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Benign", "Malicious"], yticklabels=["Benign", "Malicious"]
        )
    else:
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Benign", "Malicious"])
        ax.set_yticklabels(["Benign", "Malicious"])
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Real label")

def evaluate_and_visualize(rf_metrics, cnn_metrics, ensemble_metrics, rf_cm, cnn_cm, ensemble_cm):
    """可视化评估结果"""
    # 指标对比图
    plt.figure(figsize=(12, 6))
    metrics_df = pd.DataFrame({
        "Random Forest": rf_metrics,
        "TextCNN": cnn_metrics,
        "RFTCNN": ensemble_metrics
    }).T[["accuracy", "precision", "recall", "f1"]]
    metrics_df.plot(kind="bar", ylim=(0, 1.05), ax=plt.gca())
    plt.title("Model Performance Comparison")
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    # 混淆矩阵对比图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plot_confusion_matrix(rf_cm, "Random Forest-Confusion Matrix", axes[0])
    plot_confusion_matrix(cnn_cm, "TextCNN-Confusion Matrix", axes[1])
    plot_confusion_matrix(ensemble_cm, "RFTCNN-Confusion Matrix", axes[2])
    plt.tight_layout()
    plt.show()

# ===================== 6. 主函数 =====================
def main():
    """主流程调度"""
    # 1. 数据加载与预处理
    (
        X_rf_train, X_rf_test, y_train, y_test,
        train_loader, test_loader, vocab_size, df
    ) = load_and_preprocess_data()
    print(f"\n数据预处理完成：训练集{len(y_train)}条，测试集{len(y_test)}条")

    # 2. 训练随机森林
    print("\n=== 训练随机森林 ===")
    rf_model, rf_metrics, rf_cm = train_random_forest(X_rf_train, X_rf_test, y_train, y_test)
    print(f"随机森林评估结果：{rf_metrics}")

    # 3. 训练TextCNN
    print("\n=== 训练TextCNN ===")
    cnn_model, cnn_metrics, cnn_cm = train_textcnn(train_loader, test_loader, vocab_size)
    print(f"TextCNN评估结果：{cnn_metrics}")

    # 4. 模型融合
    print("\n=== 模型融合 ===")
    ensemble_pred = ensemble_predict(rf_model, cnn_model, X_rf_test, test_loader)
    ensemble_metrics = {
        "accuracy": accuracy_score(y_test, ensemble_pred),
        "precision": precision_score(y_test, ensemble_pred),
        "recall": recall_score(y_test, ensemble_pred),
        "f1": f1_score(y_test, ensemble_pred)
    }
    ensemble_cm = confusion_matrix(y_test, ensemble_pred)
    print(f"模型融合评估结果：{ensemble_metrics}")

    # 5. 结果可视化
    evaluate_and_visualize(rf_metrics, cnn_metrics, ensemble_metrics, rf_cm, cnn_cm, ensemble_cm)

if __name__ == "__main__":
    main()