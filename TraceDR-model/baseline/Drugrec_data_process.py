import json
import dill
import pickle
import csv
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import sparse


def process_patient_records(part="test"):
    """
    步骤1: 处理原始患者记录数据
    从pkl文件提取患者数据并转换为JSON和CSV格式
    """
    print(f"=== 步骤1: 处理{part}患者记录 ===")
    
    data_path = f"DrugRec0716/{part}.pkl"
    output_path = f"DrugRec0716/4s{part}.json"
    output_csv_path = f"DrugRec0716/4s{part}.csv"
    
    # 加载数据
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    patient_records = []

    for patient_id, patient_data in data.items():
        record = {
            "patient_id": patient_id,
            "symptoms": [],
            "medicines": [],
            "diagnosis": []
        }

        people = patient_data.get("people", {})

        # 提取症状（增强健壮性）
        record["symptoms"] = people.get("symptom", "")
        record["diagnosis"] = people.get("diagnosis", "")

        # 提取药物（处理None和空值）
        medicine_data = people.get("medicine", [])
        for drug_info in medicine_data:
            # 处理可能的非字典类型（防御性编程）
            if not isinstance(drug_info, dict):
                continue

            # 安全获取字段值
            name = str(drug_info.get("name", "")).strip()  # 强制转字符串并处理None
            cman = str(drug_info.get("CMAN", "")).strip()  # 关键修复点

            # 仅当两个字段都有有效内容时记录
            if name and cman:
                record["medicines"].append(f"{name}||{cman}")

        patient_records.append(record)

    # 保存文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(patient_records, f, ensure_ascii=False, indent=2)

    with open(output_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        # "患者ID", "症状", "药物", "诊断"
        for record in patient_records:
            symptoms = '；'.join(record["symptoms"])
            medicines = '；'.join(record["medicines"])
            diagnosis = '；'.join(record["diagnosis"])
            writer.writerow([record["patient_id"], symptoms, medicines, diagnosis])

    print(f"处理完成！共处理 {len(patient_records)} 条患者记录")
    print(f"输出文件: {output_path}, {output_csv_path}")


def build_vocabulary(json_files):
    """
    步骤2: 从多个JSON文件中构建症状和药物词汇表
    """
    print("=== 步骤2: 构建词汇表 ===")
    
    input_path = "DrugRec0716"
    sym_voc = {'word2idx': {}, 'idx2word': []}
    med_voc = {'word2idx': {}, 'idx2word': []}
    diag_voc = {'word2idx': {}, 'idx2word': []}  # 占位符，保持结构兼容性

    # 收集所有症状和药物（去重）
    all_symptoms = set()
    all_medicines = set()
    all_diagnoses = set()

    for file in json_files:
        file_path = os.path.join(input_path, file)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                records = json.load(f)
                for record in records:
                    # 处理症状（直接添加）
                    all_symptoms.update(record["symptoms"])
                    all_diagnoses.update(record["diagnosis"])
                    # 处理药物（不再分割 ||CMAN 部分）
                    for med in record["medicines"]:
                        all_medicines.add(med.strip())  # 直接添加完整字符串

    # 构建症状词汇表（索引从1开始）
    for idx, symptom in enumerate(sorted(all_symptoms), start=1):
        sym_voc['word2idx'][symptom] = idx
        sym_voc['idx2word'].append(symptom)

    # 构建药物词汇表（索引从1开始）
    for idx, med in enumerate(sorted(all_medicines), start=1):
        med_voc['word2idx'][med] = idx
        med_voc['idx2word'].append(med)

    # 构建诊断词汇表（索引从1开始）
    for idx, diagnosis in enumerate(sorted(all_diagnoses), start=1):
        diag_voc['word2idx'][diagnosis] = idx
        diag_voc['idx2word'].append(diagnosis)

    vocabulary = {
        'sym_voc': sym_voc,
        'diag_voc': diag_voc,
        'med_voc': med_voc
    }

    # 保存词汇表
    voc_output_path = 'DrugRec0716/voc_final.pkl'
    with open(voc_output_path, 'wb') as f:
        dill.dump(vocabulary, f)

    print("词汇表构建完成！")
    print(f"症状数量: {len(vocabulary['sym_voc']['idx2word'])}")
    print(f"药物数量: {len(vocabulary['med_voc']['idx2word'])}")
    print(f"诊断数量: {len(vocabulary['diag_voc']['idx2word'])}")
    print(f"词汇表保存至: {voc_output_path}")
    
    return vocabulary


def convert_json_to_pkl(json_path, voc_path, output_path):
    """
    步骤3: 将JSON患者记录转换为PKL格式数据集
    """
    print(f"=== 步骤3: 转换 {json_path} -> {output_path} ===")
    
    # 加载词汇表
    with open(voc_path, 'rb') as f:
        voc = dill.load(f)

    sym_voc = voc['sym_voc']['word2idx']
    med_voc = voc['med_voc']['word2idx']
    diag_voc = voc['diag_voc']['word2idx']

    # 读取JSON数据（显式指定UTF-8编码）
    with open(json_path, 'r', encoding='utf-8') as f:
        records = json.load(f)

    # 转换数据结构
    processed_data = []
    for record in records:
        # 转换症状（过滤不在词汇表中的症状）
        symptoms = [sym_voc[s] for s in record['symptoms'] if s in sym_voc]

        # 转换药物（过滤不在词汇表中的药物）
        medicines = [med_voc[m] for m in record['medicines'] if m in med_voc]
        diagnosis = [diag_voc[d] for d in record['diagnosis'] if d in diag_voc]
        
        # 结构：[症状列表, 诊断列表, 药物列表]
        processed_data.append([symptoms, diagnosis, medicines])

    # 保存为PKL文件
    with open(output_path, 'wb') as f:
        dill.dump(processed_data, f)
    
    print(f"成功转换，共 {len(processed_data)} 条记录")


def build_ddi_matrix(voc_path, interactions_path, output_path):
    """
    步骤4: 构建药物相互作用矩阵
    """
    print("=== 步骤4: 构建DDI矩阵 ===")
    
    # 加载药物词汇表
    voc = dill.load(open(voc_path, 'rb'))
    med_voc = voc['med_voc']
    drug2idx = med_voc['word2idx']
    n_drugs = len(med_voc['idx2word'])

    # 使用稀疏矩阵 (COO格式)
    rows, cols = [], []

    # 检查药物相互作用文件是否存在
    if not os.path.exists(interactions_path):
        print(f"警告: 药物相互作用文件 {interactions_path} 不存在，创建空的DDI矩阵")
        # 创建空的稀疏矩阵
        ddi_adj = sparse.csr_matrix((n_drugs + 1, n_drugs + 1), dtype=np.int8)
    else:
        # 加载药物相互作用
        with open(interactions_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t{')
                if len(parts) < 2:
                    continue
                drug_a = parts[0]
                drugs_b = parts[1].replace('{', '').replace('}', '').split('\t')
                for drug_b in drugs_b:
                    if drug_b not in drug2idx or drug_a not in drug2idx:
                        continue  # 跳过无效药物
                    i, j = drug2idx[drug_a], drug2idx[drug_b]
                    rows.extend([i, j])  # 对称填充
                    cols.extend([j, i])

        # 创建稀疏矩阵 (对称邻接矩阵)
        data = np.ones(len(rows), dtype=np.int8)
        ddi_adj = sparse.coo_matrix((data, (rows, cols)), shape=(n_drugs + 1, n_drugs + 1))
        ddi_adj = ddi_adj.tocsr()  # 转换为CSR格式节省内存

    # 保存为pkl
    with open(output_path, 'wb') as f:
        dill.dump(ddi_adj, f)

    print(f"DDI矩阵已保存至 {output_path}")
    print(f"矩阵维度: {ddi_adj.shape}, 非零元素数量: {ddi_adj.nnz}")

def get_drugrec_ehr():
    # 诊断-药品图
    # weighted ehr adj
    voc = dill.load(open('DrugRec0716/voc_final.pkl', 'rb'))

    # 获取词汇表大小
    n_diag = len(voc["diag_voc"]['word2idx'])
    n_med = len(voc['med_voc']['word2idx'])

    # 初始化空列表来存储非零元素的行、列索引和数据
    rows = []
    cols = []
    data = []

    drugmsg = dill.load(open("drugMsg_linux_dict.pkl", 'rb'))
    med_voc = voc['med_voc']['word2idx']
    diag_voc = voc['diag_voc']['word2idx']

    for idx, drug in drugmsg.items():
        name = drug["name"]
        cman = drug["CMAN"]
        treat_list = drug.get("治疗")
        word = f"{name}||{cman}"
        drug_id = med_voc.get(word)
        #drug_id = med_voc[word]
        if drug_id is None or treat_list is None:
            continue
        for treat in treat_list:
            diag_id = diag_voc.get(treat)  # 修正了变量名拼写错误
            if diag_id is None:
                continue
            # 确保索引在有效范围内
            if diag_id < n_diag+1 and drug_id < n_med+1:
                rows.append(diag_id)
                cols.append(drug_id)
                data.append(1)
            else:
                print(
                    f"Warning: 索引超出范围 - diag_id: {diag_id} (max {n_diag}), drug_id: {drug_id} (max {n_med})")

    # 创建稀疏矩阵
    ehr_adj = coo_matrix((data, (rows, cols)),
                         shape=(n_diag+1, n_med+1))

    dill.dump(ehr_adj, open('DrugRec0716/ehr_adj_final.pkl', 'wb'))

def main():
    """
    主函数：按顺序执行所有数据处理步骤
    """
    print("开始数据处理流程...")
    
    # 确保输出目录存在
    os.makedirs('DrugRec0716', exist_ok=True)
    
    # 定义数据集部分
    parts = ['train', 'dev', 'test']
    
    # 步骤1: 处理所有部分的患者记录
    for part in parts:
        if os.path.exists(f"DrugRec0716/{part}.pkl"):
            process_patient_records(part)
        else:
            print(f"警告: 源文件 DrugRec0716/{part}.pkl 不存在，跳过处理")
    
    # 步骤2: 构建词汇表
    json_files = ['4strain.json', '4sdev.json', '4stest.json']
    build_vocabulary(json_files)
    
    # 步骤3: 转换数据集
    voc_path = 'DrugRec0716/voc_final.pkl'
    datasets = [
        ('4strain.json', 'data_train.pkl'),
        ('4sdev.json', 'data_eval.pkl'),
        ('4stest.json', 'data_test.pkl')
    ]
    
    for json_file, pkl_file in datasets:
        json_path = os.path.join('DrugRec0716', json_file)
        output_path = os.path.join('DrugRec0716', pkl_file)
        if os.path.exists(json_path):
            convert_json_to_pkl(json_path, voc_path, output_path)
        else:
            print(f"警告: JSON文件 {json_path} 不存在，跳过转换")
    
    # 步骤4: 构建DDI矩阵
    build_ddi_matrix(
        voc_path='DrugRec0716/voc_final.pkl',
        interactions_path='drug_interactions.txt',
        output_path='DrugRec0716/ddi_A_final.pkl'
    )
    
    # 验证生成文件
    print("\n=== 数据集构建完成！结构验证 ===")
    train_data_path = 'DrugRec0716/data_train.pkl'
    if os.path.exists(train_data_path):
        sample_data = dill.load(open(train_data_path, 'rb'))
        if sample_data:
            print(f"data_train.pkl 首样本结构: {sample_data[0]}")
            print(f"训练集样本总数: {len(sample_data)}")
        else:
            print("训练集为空")
    else:
        print("训练集文件不存在")
    
    print("\n所有数据处理步骤完成！")


if __name__ == "__main__":
    main() 