import pickle

from sklearn.metrics import recall_score, f1_score, precision_score, confusion_matrix
from cuml.svm import LinearSVC


def train(x, y):
    # 创建 SVM 模型
    model = LinearSVC(penalty='l1', loss='hinge', class_weight='balanced', C=4)
    # 训练 SVM 模型
    model.fit(x, y)
    # 使用模型进行预测
    y_pred = model.predict(x)
    # 计算总体精确率
    precision = precision_score(y, y_pred, average='micro')
    # 计算总体召回率
    recall = recall_score(y, y_pred, average='micro')
    # 计算总体F1测度
    f1 = f1_score(y, y_pred, average='micro')
    # 保存SVM模型
    with open('model/model.pickle', 'wb') as fw:
        pickle.dump(model, fw)
    return precision, recall, f1


def test(x, y):
    # 加载 SVM 模型 
    with open('model/model.pickle', 'rb') as f:
        model = pickle.load(f)
    # 使用模型进行预测
    y_pred = model.predict(x)
    categories = model.classes_
    # 计算每类的精确率
    precision = precision_score(y, y_pred, average=None)
    # 计算每类的召回率
    recall = recall_score(y, y_pred, average=None)
    # 计算每类的F1测度
    f1 = f1_score(y, y_pred, average=None)
    # 计算总体精确率
    precision_overall = precision_score(y, y_pred, average='micro')
    # 计算总体召回率
    recall_overall = recall_score(y, y_pred, average='micro')
    # 计算总体F1测度
    f1_overall = f1_score(y, y_pred, average='micro')
    # 计算混淆矩阵
    cm = confusion_matrix(y, y_pred)
    return zip(categories, precision, recall, f1), precision_overall, recall_overall, f1_overall, cm