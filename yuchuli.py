import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
import csv

# ===================== 配置项 =====================
# 原始数据集路径
RAW_DATA_PATH = "api_sequences_dataset.csv"
# 清洗后的数据集路径
CLEAN_DATA_PATH = "cleaned_api_sequences.csv"
# 特征增强后的数据集路径
ENHANCED_DATA_PATH = "enhanced_api_sequences.csv"
# API行为类别映射（可扩展）
API_CATEGORY_MAP = {
    # 恶意特征API
    "VirtualAlloc": "Memory_Operation",
    "NtProtectVirtualMemory": "Memory_Operation",
    "connect": "Network_Communication",
    "send": "Network_Communication",
    "CreateRemoteThread": "Thread_Operation",
    "RegSetValueEx": "Persistence",
    "CONNECT": "Network_Communication",  # 匹配CONNECT
    "CREATEREMOTE": "Thread_Operation",  # 匹配CREATEREMOTE（你的数据中的缩写）
    "CREATEREMOTETHREAD": "Thread_Operation",  # 若有完整名称也补充
    "VIRTUALALLOC": "Memory_Operation",  # 匹配大写的VIRTUALALLOC
    "CRYPTENCRYPT": "Encryption",        # 匹配CRYPTENCRYPT
    "GETMESSAGEW": "Window_Message",     # 匹配GETMESSAGEW
    # 良性特征API
    "CreateFileW": "File_Operation",
    "ReadFile": "File_Operation",
    "WriteFile": "File_Operation",
    "GetMessageW": "Window_Message",
    "DispatchMessageW": "Window_Message",
    # 其他API
    "CryptEncrypt": "Encryption",
    "GetMessageW": "Window_Message"
}
# Word2Vec配置
W2V_VECTOR_SIZE = 10  # 词向量维度
W2V_WINDOW = 2        # 窗口大小
W2V_MIN_COUNT = 1     # 最小词频

# ===================== 1. 数据清洗 =====================
def data_cleaning(raw_path, clean_path):
    """
    数据清洗：删除空序列、去重、格式标准化
    """
    # 读取原始数据
    df = pd.read_csv(raw_path, encoding="utf-8-sig")
    print(f"原始数据行数：{len(df)}")

    # 1. 删除空序列
    df = df[df["api_sequence"].notna() & (df["api_sequence"].str.strip() != "")]
    print(f"删除空序列后行数：{len(df)}")

    # 2. 去重（按镜像文件名、PID、api_sequence、label去重）
    df = df.drop_duplicates(subset=["镜像文件名", "PID", "api_sequence", "label"])
    print(f"去重后行数：{len(df)}")

    # 3. 格式标准化：去除API序列的空格，统一为大写（可选）
    df["api_sequence"] = df["api_sequence"].str.replace(" ", "").str.upper()

    # 保存清洗后的数据
    df.to_csv(clean_path, index=False, encoding="utf-8-sig")
    print(f"数据清洗完成，保存至 {clean_path}")
    return df

# ===================== 2. 特征增强 =====================
def feature_enhancement(clean_df, enhanced_path):
    """
    特征增强：构建统计特征、n-gram特征、行为类别特征、词嵌入特征
    """
    # 提取API序列列表
    api_sequences = [seq.split(",") for seq in clean_df["api_sequence"]]

    # -------------------- 2.1 基础统计特征 --------------------
    # 序列长度特征
    clean_df["seq_length"] = clean_df["api_sequence"].apply(lambda x: len(x.split(",")))

    # API计数特征（统计每个API在序列中出现的次数）
    all_apis = list(set([api for seq in api_sequences for api in seq]))
    for api in all_apis:
        clean_df[f"api_count_{api}"] = clean_df["api_sequence"].apply(
            lambda x: x.split(",").count(api)
        )

    # -------------------- 2.2 n-gram特征 --------------------
    # 构建二元组特征
    def get_ngram(seq, n=2):
        seq_list = seq.split(",")
        return ["_".join(seq_list[i:i+n]) for i in range(len(seq_list)-n+1)]
    
    clean_df["bigram"] = clean_df["api_sequence"].apply(lambda x: " ".join(get_ngram(x, 2)))
    vectorizer = CountVectorizer()
    bigram_features = vectorizer.fit_transform(clean_df["bigram"]).toarray()
    bigram_df = pd.DataFrame(bigram_features, columns=[f"bigram_{feat}" for feat in vectorizer.get_feature_names_out()])

    # -------------------- 2.3 行为类别特征 --------------------
    # 将API序列转换为行为类别序列
    def api_to_category(seq):
        return [API_CATEGORY_MAP.get(api, "Other") for api in seq.split(",")]
    
    clean_df["category_sequence"] = clean_df["api_sequence"].apply(
        lambda x: ",".join(api_to_category(x))
    )

    # -------------------- 2.4 Word2Vec词嵌入特征 --------------------
    # 训练Word2Vec模型
    w2v_model = Word2Vec(sentences=api_sequences, vector_size=W2V_VECTOR_SIZE, window=W2V_WINDOW, min_count=W2V_MIN_COUNT, workers=4)
    
    # 计算每个序列的词向量均值（作为序列的特征）
    def seq_to_w2v(seq):
        seq_list = seq.split(",")
        vectors = [w2v_model.wv[api] for api in seq_list if api in w2v_model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(W2V_VECTOR_SIZE)
    
    w2v_features = np.array([seq_to_w2v(seq) for seq in clean_df["api_sequence"]])
    w2v_df = pd.DataFrame(w2v_features, columns=[f"w2v_{i}" for i in range(W2V_VECTOR_SIZE)])

    # -------------------- 合并所有特征 --------------------
    enhanced_df = pd.concat([clean_df, bigram_df, w2v_df], axis=1)

    # 保存特征增强后的数据
    enhanced_df.to_csv(enhanced_path, index=False, encoding="utf-8-sig")
    print(f"特征增强完成，保存至 {enhanced_path}")
    return enhanced_df

# ===================== 3. 主函数 =====================
def main():
    # 1. 数据清洗
    clean_df = data_cleaning(RAW_DATA_PATH, CLEAN_DATA_PATH)

    # 2. 特征增强
    enhanced_df = feature_enhancement(clean_df, ENHANCED_DATA_PATH)

    # 输出关键信息
    print(f"\n最终特征维度：{enhanced_df.shape[1]}")
    print(f"最终数据行数：{len(enhanced_df)}")
    print(f"特征增强后的列名：{enhanced_df.columns.tolist()[:10]}...")  # 打印前10个列名

if __name__ == "__main__":
    main()