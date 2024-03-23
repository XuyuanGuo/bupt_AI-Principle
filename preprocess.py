import os
import pickle
import random
from concurrent.futures import ProcessPoolExecutor

import cudf
import numpy as np
import pandas as pd
from jieba import posseg
from cuml.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2


def get_stop_words():
    # 读取停用词文件，初始化停用词列表
    with open('stop_words_ch-停用词表.txt', 'r', encoding='GBK') as file:
        # 逐行读取停用词并添加到列表
        stopwords_list = [line.strip() for line in file.readlines()]
    return stopwords_list


def jieba_cut(content):
    words = posseg.cut(content)
    nouns = [word for word, flag in words if flag.startswith('n')]
    return " ".join(nouns)


def multi_thread_jieba_cut(row_data):
    print("JIEBA分词……")
    # 使用python线程池实现多线程jieba分词
    with ProcessPoolExecutor() as executor:
        data = list(executor.map(jieba_cut, row_data))
    return data


def get_tfidf(data, mode, stopwords_list):
    print("计算TF-IDF……")
    data = cudf.Series(data)
    if mode == 'train':
        # 使用TF-IDF进行特征提取，此语句涵盖建词典、文档对齐和统计TF-IDF
        tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords_list)
        tfidf = tfidf_vectorizer.fit_transform(data)
        # 输出词典
        word_list = tfidf_vectorizer.get_feature_names().to_pandas()
        with open('vocabulary.txt', 'w', encoding='utf-8') as file:
            for word in word_list:
                file.write(word + '\n')
        # 保存TF-IDF向量器和选择特征索引
        with open('model/tfidf_vect.pickle', 'wb') as f:
            pickle.dump(tfidf_vectorizer, f)
    else:
        with open('model/tfidf_vect.pickle', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        tfidf = tfidf_vectorizer.transform(data)
    return tfidf


def chi2_select(tfidf, label, mode, k=300):
    print("chi2特征选择……")
    if mode == 'train':
        select_features = np.zeros(tfidf.shape[1], dtype=bool)
        # 创建一个二元标签，某类文档标签为1，其他类文档标签为0
        unique_label = np.unique(label)
        for each in unique_label:
            label_binary = label == each
            # 卡方特征选择
            selector = SelectKBest(chi2, k=k)
            selector.fit_transform(tfidf.get(), label_binary)
            # 不同类别可能选出相同特征，使用“或”运算进行合并
            select_features = np.logical_or(select_features, selector.get_support())

        np.save('data/select_features.npy', select_features)
    else:
        select_features = np.load('data/select_features.npy')

    # 将原始TF-IDF矩阵映射到所选取的特征中
    tfidf_selected = tfidf[:, select_features]
    return tfidf_selected