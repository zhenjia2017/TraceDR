import numpy as np
import torch
from vocab import build_vocab
import pickle
import json


def load_json(file_path):
    with open(file_path, 'rb') as f:
        return json.load(f)


def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":

    # 指定.npy文件的路径
    true_file_path = 'results/true.npy'
    true_data = np.load(true_file_path)

    pred_file_path = 'results/pred.npy'
    pred_data = np.load(pred_file_path)

    # print(true_data[10])
    # print(pred_data[10])

    with open('results/input_vocab.pkl', 'rb') as f:
        in_vocab = pickle.load(f)

    with open('results/output_vocab.pkl', 'rb') as f:
        out_vocab = pickle.load(f)

    with open('DrugRec0716/test.pkl', 'rb') as f:
        test_data = pickle.load(f)

    transformer_result = []
    i = 0
    # 输出pred_data和true_data中每个元素对应的键
    for idx, data in test_data.items():
        pred_keys = [next((k for k, v in out_vocab.items() if v == val), None) for val in pred_data[i]]
        true_keys = [next((k for k, v in out_vocab.items() if v == val), None) for val in true_data[i]]
        pred_keys = [item for item in pred_keys if item not in ['<eos>', '<pad>']]
        true_keys = [item for item in true_keys if item not in ['<eos>', '<pad>']]
        transformer_result.append({
            "id": idx,
            #"info": data["question"],
            "recommended": pred_keys,
            "answer": true_keys
        })
        #print(test_data[i])
        i += 1
    save_json(transformer_result, 'results/ddxt_result.json')

 
