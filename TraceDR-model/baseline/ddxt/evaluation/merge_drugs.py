import pickle
import json
from collections import defaultdict


# 合并函数
def merge_drugs(drug_data):
    merged_data = defaultdict(dict)
    for drug in drug_data.values():
        name = drug['name']
        if name not in merged_data:
            merged_data[name] = drug
        else:
            # 保持id和CMAN不变
            merged_data[name]['相互作用'] = list(set(merged_data[name].get('相互作用', []) + drug.get('相互作用', [])))
            merged_data[name]['用药方法'] = list(set(merged_data[name].get('用药方法', []) + drug.get('用药方法', [])))
            merged_data[name]['不良反应'] = list(set(merged_data[name].get('不良反应', []) + drug.get('不良反应', [])))
            merged_data[name]['规格'] = list(set(merged_data[name].get('规格', []) + drug.get('规格', [])))
            merged_data[name]['治疗'] = list(set(merged_data[name].get('治疗', []) + drug.get('治疗', [])))
            merged_data[name]['成分'] = list(set(merged_data[name].get('成分', []) + drug.get('成分', [])))
            merged_data[name]['禁用'] = list(set(merged_data[name].get('禁用', []) + drug.get('禁用', [])))
    return dict(merged_data)


def save_pkl(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    drug_file_path = '../data/drugMsg_linux_dict.pkl'
    # result_file_path = '../results/qwen_test_result.json'

    # 读取药品数据
    with open(drug_file_path, 'rb') as f:
        drug_data = pickle.load(f)

    merged_drug_data = merge_drugs(drug_data)
    for key, item in merged_drug_data.items():
        print(item)
    print(len(merged_drug_data))

    # 将合并后的数据存储为pickle文件
    save_pkl('../results/merged_drug_data.pkl', merged_drug_data)
