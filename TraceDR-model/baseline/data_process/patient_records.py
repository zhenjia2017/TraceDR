import pickle
import json
import csv
part = "test"
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
    record["symptoms"] = people.get("symptom","")
    record["diagnosis"] = people.get("diagnosis","")
    #record["symptoms"] = [s.strip() for s in symptom_str.split('、') if s.strip()]

    # 提取药物（处理None和空值）
    medicine_data = people.get("medicine",[])
    for drug_info in medicine_data:
        # 处理可能的非字典类型（防御性编程）
        if not isinstance(drug_info, dict):
            continue

        # 安全获取字段值
        name = str(drug_info.get("name","")).strip()  # 强制转字符串并处理None
        cman = str(drug_info.get("CMAN","")).strip()  # 关键修复点

        # 仅当两个字段都有有效内容时记录
        if name and cman:
            record["medicines"].append(f"{name}||{cman}")

    patient_records.append(record)

# 保存文件（保持原有逻辑）
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(patient_records, f, ensure_ascii=False, indent=2)

with open(output_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
   # "患者ID", "症状", "药物"
    for record in patient_records:
        symptoms = '；'.join(record["symptoms"])
        medicines = '；'.join(record["medicines"])
        diagnosis = '；'.join(record["diagnosis"])
        writer.writerow([record["patient_id"], symptoms, medicines, diagnosis])

print(f"处理完成！共处理 {len(patient_records)} 条患者记录")



