import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_tfidf_features_and_labels(benign_folder, malicious_folder):
    features = []
    labels = []
    all_feature_names = set()  # 用于存储所有特征名称

    # 辅助函数：处理单个特征文件，返回特征字典
    def process_feature_file(file_path):
        feature_dict = {}
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()  # 去除首尾空白字符
                if not line:  # 跳过空行
                    continue
                parts = line.split(': ')
                if len(parts) != 2:  # 跳过格式不正确的行
                    print(f"警告: 文件 {file_path} 包含格式不正确的行: '{line}'")
                    continue
                feature_name, feature_value = parts
                try:
                    feature_dict[feature_name] = float(feature_value)
                except ValueError:
                    print(f"警告: 文件 {file_path} 包含无法转换为浮点数的值: '{feature_value}'")
        return feature_dict

    # 遍历良性软件文件夹，收集所有特征名称
    for file_name in os.listdir(benign_folder):
        if file_name.endswith('.txt'):
            file_path = os.path.join(benign_folder, file_name)
            feature_dict = process_feature_file(file_path)
            all_feature_names.update(feature_dict.keys())
    
    # 遍历恶意软件文件夹，收集所有特征名称
    for file_name in os.listdir(malicious_folder):
        if file_name.endswith('.txt'):
            file_path = os.path.join(malicious_folder, file_name)
            feature_dict = process_feature_file(file_path)
            all_feature_names.update(feature_dict.keys())

    # 将所有特征名称转换为列表，并排序
    all_feature_names = sorted(list(all_feature_names))

    # 再次遍历良性软件文件夹，加载特征
    for file_name in os.listdir(benign_folder):
        if file_name.endswith('.txt'):
            file_path = os.path.join(benign_folder, file_name)
            feature_dict = process_feature_file(file_path)
            # 确保特征向量包含所有特征名称，并填充缺失的特征为0
            feature_vector = [feature_dict.get(feature_name, 0.0) for feature_name in all_feature_names]
            features.append(feature_vector)
            labels.append(0)  # 良性软件标签为0

    # 遍历恶意软件文件夹，加载特征
    for file_name in os.listdir(malicious_folder):
        if file_name.endswith('.txt'):
            file_path = os.path.join(malicious_folder, file_name)
            feature_dict = process_feature_file(file_path)
            # 确保特征向量包含所有特征名称，并填充缺失的特征为0
            feature_vector = [feature_dict.get(feature_name, 0.0) for feature_name in all_feature_names]
            features.append(feature_vector)
            labels.append(1)  # 恶意软件标签为1

    return np.array(features), np.array(labels)

# 指定良性软件和恶意软件特征文件夹的路径
benign_folder = r'F:\OPGCN\bentf'
malicious_folder = r'F:\OPGCN\maltf'

# 加载特征和标签
X, y = load_tfidf_features_and_labels(benign_folder, malicious_folder)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化决策树分类器
clf = DecisionTreeClassifier(random_state=42)

# 训练模型
clf.fit(X_train, y_train)

# 预测测试集
y_pred = clf.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 输出评估结果
print("模型评估结果:")
print(f"准确率: {accuracy:.4f}")
print(f"精确率: {precision:.4f}")
print(f"召回率: {recall:.4f}")
print(f"F1值: {f1:.4f}")