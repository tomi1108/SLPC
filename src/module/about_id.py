import torch
import numpy as np

def add_ids(data, label):
    #データとラベルのサイズが異なる場合はエラーを返す
    if len(data) != len(label):
        raise RuntimeError(">> Error: data and label are not same length.")
    
    print("---Adding IDs to data and label---")
    #データサイズ分のランダムなIDを生成
    ids = np.arange(len(data))
    np.random.shuffle(ids)

    new_data = [[] for _ in range(len(data))]
    new_label = [[] for _ in range(len(data))]
    
    for i in range(len(data)):
        new_data[i].append(data[i])
        new_data[i].append(ids[i])
        new_label[i].append(label[i])
        new_label[i].append(ids[i])

    if len(new_data) != len(new_label):
        raise RuntimeError(">> Error: new_data and new_label are not same length.")

    return data[0], label[0]