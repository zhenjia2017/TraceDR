import json
import dill
import os
import numpy as np
from collections import defaultdict


def convert_json_to_pkl(json_path, voc_path, output_path):
    """将JSON患者记录转换为PKL格式数据集"""
    # 加载词汇表
    with open(voc_path, 'rb') as f:
        voc = dill.load(f)

    sym_voc = voc['sym_voc']['word2idx']
    med_voc = voc['med_voc']['word2idx']
    diag_voc = voc['diag_voc']['word2idx']

    # 读取JSON数据（显式指定UTF-8编码）
    with open(json_path, 'r', encoding='utf-8') as f:  # 修复点：添加encoding参数
        records = json.load(f)

    # 转换数据结构
    processed_data = []
    for record in records:
        # 转换症状（过滤不在词汇表中的症状）
        symptoms = [sym_voc[s] for s in record['symptoms'] if s in sym_voc]

        # 转换药物（过滤不在词汇表中的药物）
        medicines = [med_voc[m] for m in record['medicines'] if m in med_voc]
        diagnosis = [diag_voc[d] for d in record['diagnosis'] if d in diag_voc]
        # 结构：[症状列表, 空诊断占位符, 药物列表]
        processed_data.append([symptoms, diagnosis, medicines])

    # 保存为PKL文件
    with open(output_path, 'wb') as f:
        dill.dump(processed_data, f)
        
def get_gamenet_data(part = "eval"):
    with open(f"DrugRec0716/data_{part}.pkl","rb") as f1:
        data = pickle.load(f1)
    subject_id = 0
    patient_list = list()
    for adm in data:
        patient_list.append(
            {
                'SUBJECT_ID': subject_id,
                'HADM_ID': subject_id + 200000,
                'ICD9_CODE': adm[1],
                'NDC': adm[2],
                'PRO_CODE': [0]
            }
        )
        subject_id += 1
    gamenet_data = pd.DataFrame(patient_list)
    gamenet_data.to_pickle('durgrec_train_final.pkl')
    with pd.option_context('display.max_columns', None):
        print(gamenet_data.head())  # 完整显示前5行的所有列


if __name__ == "__main__":
    # 配置路径
    voc_path = 'voc_final.pkl'
    datasets = [
        ('4strain.json', 'data_train.pkl'),
        ('4sdev.json', 'data_eval.pkl'),
        ('4stest.json', 'data_test.pkl')
    ]



    # 转换每个数据集
    for json_file, pkl_file in datasets:
        convert_json_to_pkl(
            os.path.join('DrugRec0716', json_file),
            os.path.join('DrugRec0716', voc_path),
            os.path.join('DrugRec0716', pkl_file)
        )
        print(f"成功转换 {json_file} -> DrugRec0716/{pkl_file}")

    # 验证生成文件
    print("\n数据集构建完成！结构验证：")
    sample_data = dill.load(open('DrugRec0716/data_train.pkl', 'rb'))
    print(f"data_train.pkl 首样本结构: {sample_data[0]}")