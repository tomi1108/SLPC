import torch
import numpy as np

"""
add_ids(data, label)
入力: 入力データ、その正解ラベル
出力: IDを付与して、ID順にソートした入力データ、IDを付与してシャッフルした正解ラベル
"""
def add_ids(data, label):
    #データとラベルのサイズが異なる場合はエラーを返す
    if len(data) != len(label):
        raise RuntimeError(">> Error: data and label are not same length.")
    
    print("---Adding IDs to data and label---")
    #データサイズ分のランダムなIDを生成
    ids = np.arange(len(data))
    np.random.shuffle(ids)

    #IDを付与するためのリストを作成
    new_data = [[] for _ in range(len(data))]
    new_label = [[] for _ in range(len(data))]
    
    #データとラベルにIDを付与
    for i in range(len(data)):
        new_data[i].append(data[i])
        new_data[i].append(ids[i])
        new_label[i].append(label[i])
        new_label[i].append(ids[i])
    
    #データはID順にソート、ラベルはシャッフルする
    new_data.sort(key=lambda x:x[1])
    np.random.shuffle(new_label)

    #処理後のデータとラベルのサイズが異なる場合はエラーを返す
    if len(new_data) != len(new_label):
        raise RuntimeError(">> Error: new_data and new_label are not same length.")
    else:
        print(">> Added IDs to data and label.\n")
        return new_data, new_label