import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_feature_vectors(benign_folder_path, malicious_folder_path):
    """
    从指定的良性和恶意文件夹中加载特征向量及其对应的标签
    :param benign_folder_path: 存储良性特征向量txt文件的文件夹路径
    :param malicious_folder_path: 存储恶意特征向量txt文件的文件夹路径
    :return: 特征向量数组和标签数组
    """
    feature_vectors = []
    labels = []
    max_length = 0

    # 加载良性特征向量
    for filename in os.listdir(benign_folder_path):
        file_path = os.path.join(benign_folder_path, filename)
        if os.path.isfile(file_path) and filename.endswith('.txt'):
            vector = []
            with open(file_path, 'r') as file:
                for line in file:
                    # 分割每行的操作码和TF-IDF值
                    parts = line.strip().split(':')
                    if len(parts) == 2:
                        try:
                            # 提取TF-IDF值并转换为浮点数
                            value = float(parts[1].strip())
                            vector.append(value)
                        except ValueError:
                            print(f"无法将 '{parts[1].strip()}' 转换为浮点数。")
            feature_vectors.append(vector)
            if len(vector) > max_length:
                max_length = len(vector)
            labels.append(0)

    # 加载恶意特征向量
    for filename in os.listdir(malicious_folder_path):
        file_path = os.path.join(malicious_folder_path, filename)
        if os.path.isfile(file_path) and filename.endswith('.txt'):
            vector = []
            with open(file_path, 'r') as file:
                for line in file:
                    # 分割每行的操作码和TF-IDF值
                    parts = line.strip().split(':')
                    if len(parts) == 2:
                        try:
                            # 提取TF-IDF值并转换为浮点数
                            value = float(parts[1].strip())
                            vector.append(value)
                        except ValueError:
                            print(f"无法将 '{parts[1].strip()}' 转换为浮点数。")
            feature_vectors.append(vector)
            if len(vector) > max_length:
                max_length = len(vector)
            labels.append(1)

    # 填充缺失值
    for i in range(len(feature_vectors)):
        vector = feature_vectors[i]
        padding = [0] * (max_length - len(vector))
        feature_vectors[i] = vector + padding

    return np.array(feature_vectors), np.array(labels)


def train_and_evaluate_model(X, y):
    """
    训练并评估双层随机森林模型
    :param X: 特征向量数组
    :param y: 标签数组
    """
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 第一层随机森林训练
    first_layer_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    first_layer_rf.fit(X_train, y_train)

    # 获取第一层特征重要性，选择重要特征
    feature_importances = first_layer_rf.feature_importances_
    important_feature_indices = np.argsort(feature_importances)[::-1][:int(len(feature_importances) * 0.5)]
    X_train_selected = X_train[:, important_feature_indices]
    X_test_selected = X_test[:, important_feature_indices]

    # 第二层随机森林训练
    second_layer_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    second_layer_rf.fit(X_train_selected, y_train)

    # 模型评估
    y_pred = second_layer_rf.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"准确率: {accuracy}")
    print(f"精确率: {precision}")
    print(f"召回率: {recall}")
    print(f"F1值: {f1}")

    # 交叉验证
    cv_scores = cross_val_score(second_layer_rf, X_train_selected, y_train, cv=5)
    print(f"5折交叉验证得分: {cv_scores}")
    print(f"交叉验证平均得分: {np.mean(cv_scores)}")


if __name__ == "__main__":
    benign_folder_path = r'F:\OPGCN\bentf'  # 替换为实际存储良性特征向量txt文件的文件夹路径
    malicious_folder_path = r'F:\OPGCN\maltf'  # 替换为实际存储恶意特征向量txt文件的文件夹路径
    X, y = load_feature_vectors(benign_folder_path, malicious_folder_path)
    train_and_evaluate_model(X, y)