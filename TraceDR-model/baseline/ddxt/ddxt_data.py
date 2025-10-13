import pickle
import json
import csv
import re

file_path = 'DrugRec0716/train.pkl'
csv_file_path = 'data/ddxt_train_data.csv'

with open(file_path, 'rb') as f:
    data = pickle.load(f)

person_data_list = []
for key, value in data.items():
    person_data = value.get('people', None)
    if isinstance(person_data, dict):
        person_info_dict = person_data
        person_info_dict['top_k_drugs'] = value['top_k_drugs']

        # age = person_info_dict["age"]
        # gender = person_info_dict["gender"]
        # diagnosis = person_info_dict["diagnosis"]
        # symptom = person_info_dict["symptom"]
        # group = person_info_dict["group"]
        # allergen = person_info_dict["allergen"]
        
        #person_info_dict['antecedents'] = value['antecedents']
        person_info_dict['antecedents'] = "无" if not person_info_dict.get('antecedents', []) else "、".join(person_info_dict['antecedents'])

        unique_medicines = {}
        # for med_id, med_info in person_info_dict["top_k_drugs"].items():
        for med_info in person_info_dict["medicine"]:
            if med_info["name"] not in unique_medicines:
                unique_medicines[med_info["name"]] = med_info
        person_info_dict["medicine_only_name"] = unique_medicines

        medicine_names = [med_info["name"] for med_info in person_info_dict["medicine_only_name"].values()]
        answer = "、".join(medicine_names)
        person_info_dict["answer"] = answer

        unique_on_medicines = {}
        for med_info in person_info_dict["on_medicine"]:
            if med_info["name"] not in unique_on_medicines:
                unique_on_medicines[med_info["name"]] = med_info
        person_info_dict["on_medicine_only_name"] = unique_on_medicines

        medicine_names = [med_info["name"] for med_info in person_info_dict["on_medicine_only_name"].values()]
        on_medicine = "无" if not person_info_dict.get("on_medicine_only_name", {}) else "、".join(medicine_names)
        person_info_dict["on_medicine"] = on_medicine


        attributes_to_replace = ["diagnosis", "symptom", "antecedent", "allergen", "on_medicine"]

        # 遍历属性列表，并对每个属性应用正则表达式替换
        for attribute in attributes_to_replace:
            if attribute in person_info_dict:
                #person_info_dict[attribute] = re.sub(r'[^\w]', '、', person_info_dict[attribute])
                person_info_dict[attribute] = '、'.join(person_info_dict[attribute])


        # 要删除的键列表
        keys_to_remove = ["medicine_num", "medicine", "medicine_only_name","on_medicine_only_name", "top_k_drugs"]
        # 循环并尝试删除每个键
        for item in keys_to_remove:
            person_info_dict.pop(item, None)  # 使用None作为默认值，防止KeyError

        person_data_list.append(person_info_dict)
        # 打印字典
        # print(person_info_dict)
print(len(person_data_list))

# Headers for the CSV file
headers = person_data_list[0].keys()
print(person_data_list[0].keys())

# Writing data to a CSV file
with open(csv_file_path, 'w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=headers)
    writer.writeheader()
    for item in person_data_list:
        writer.writerow(item)
