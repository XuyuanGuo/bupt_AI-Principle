import os
import dask.bag as db
from sklearn.utils import shuffle


def get_label_dict(folder_path):
    label_dict = dict()
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    for i, label in enumerate(subfolders):
        # 将字符串标签编码为数字，同时保留标签字典
        label_dict[i] = label
    return label_dict


def get_data(folder_path, randomize=0):
    # 初始化空列表存储数据和标签
    data = []
    label = []
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    num_subfolders = len(subfolders)
    # 遍历总文件夹下的子文件夹
    for i, folder_name in enumerate(subfolders):
        label_path = os.path.join(folder_path, folder_name)

        # 遍历子文件夹下的数据文件
        for file_name in os.listdir(label_path):
            file_path = os.path.join(label_path, file_name)
            # 读取数据文件内容
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            # 添加数据
            data.append(content)
            label.append(i)
            
            
    # 如果randomize为1，则打乱data和labels，但保持它们之间的对应关系
    if randomize == 1:  
        data, label = shuffle(data, label, random_state=42)
    return data, label