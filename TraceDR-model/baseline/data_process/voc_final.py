import json
import dill
from collections import defaultdict
import os

def build_vocabulary(json_files):
    """从多个JSON文件中构建症状和药物词汇表（药物名称保留完整信息）"""
    input_path = "DrugRec0716"
    sym_voc = {'word2idx': {}, 'idx2word': []}
    med_voc = {'word2idx': {}, 'idx2word': []}
    diag_voc = {'word2idx': {}, 'idx2word': []}  # 占位符，保持结构兼容性

    # 收集所有症状和药物（去重）
    all_symptoms = set()
    all_medicines = set()
    all_diagnoses = set()

    for file in json_files:
        with open(os.path.join(input_path, file), 'r', encoding='utf-8') as f:
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

    return {
        'sym_voc': sym_voc,
        'diag_voc': diag_voc,
        'med_voc': med_voc
    }


if __name__ == "__main__":
    #os.makedirs('datasets/4S', exist_ok=True)
    json_files = ['4strain.json', '4sdev.json', '4stest.json']
    voc_final = build_vocabulary(json_files)

    with open('DrugRec0716/voc_final.pkl', 'wb') as f:
        dill.dump(voc_final, f)

    print("词汇表构建完成！")
    print(f"症状数量: {len(voc_final['sym_voc']['idx2word'])}")
    print(f"药物数量: {len(voc_final['med_voc']['idx2word'])}")
    print(f"诊断数量: {len(voc_final['diag_voc']['idx2word'])}")