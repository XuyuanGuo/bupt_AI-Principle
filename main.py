import os
import pickle
import time

import matplotlib as mpl
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy.sparse import load_npz
from scipy.sparse import save_npz

from model import train, test
from preprocess import get_stop_words, multi_thread_jieba_cut, get_tfidf, chi2_select
from dataloader import get_data, get_label_dict


os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def data_transform():
    stopwords = get_stop_words()

    # 获取训练数据
    data, label = get_data('../train', randomize=1)
    label = np.array(label)
    data = multi_thread_jieba_cut(data)
    data = get_tfidf(data, 'train', stopwords)
    data = chi2_select(data ,label, 'train', k=1000)
    save_npz('data/train_tfidf.npz', data)
    np.save('data/train_label.npy', label)
    # 获取测试数据
    data, label = get_data('../test')
    label = np.array(label)
    data = multi_thread_jieba_cut(data)
    data = get_tfidf(data, 'test', stopwords)
    data = chi2_select(data ,label, 'test')
    save_npz('data/test_tfidf.npz', data)
    np.save('data/test_label.npy', label)


if __name__ == '__main__':
    start = time.time()

    data_transform()


    x_train = load_npz('data/train_tfidf.npz').toarray()
    y_train = np.load('data/train_label.npy')

    print(x_train.shape, y_train.shape)

    # 模型训练
    train_pre, train_rec, train_f1 = train(x_train, y_train)
    # 训练集指标
    print(f'训练集，总体精确率：{train_pre}，总体召回率：{train_rec}，总体F1-score：{train_f1}')


    x_test = load_npz('data/test_tfidf.npz').toarray()
    y_test = np.load('data/test_label.npy')
    # 模型测试
    performance, test_pre, test_rec, test_f1, cm = test(x_test, y_test)
    print("测试集，")
    label_dict = get_label_dict('../test')
    tick_label = []
    # 打印每个类别的名称和对应的性能指标
    for category, pre, rec, f1 in performance:
        tick_label.append(label_dict[int(category)])
        print(f"{label_dict[int(category)]}: 精确率 = {pre:.2f}, 召回率 = {rec:.2f}, F1分数 = {f1:.2f}")
    print(f'总体精确率：{test_pre}，总体召回率：{test_rec}，总体F1-score：{test_f1}')
    # 打印测试集混淆矩阵
    custom_font = FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
    plt.rcParams['font.sans-serif'] = [custom_font.get_name()]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=tick_label, yticklabels=tick_label)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('cunfusion_matrix.png')

    end = time.time()
    print('本次运行时间（秒）：', end - start)
