import json
import csv
import re


def read_age():
    return ['age_1-4', 'age_5-14', 'age_15-29', 'age_30-44', 'age_45-59', 'age_60-74', 'age_above75']


def read_sex():
    return ['男', '女']


def read_info():
    with open('data/ddxt_test_data.csv', mode='r', encoding='utf-8') as f:
        data_test = list(csv.DictReader(f))
    with open('data/ddxt_dev_data.csv', mode='r', encoding='utf-8') as f:
        data_dev = list(csv.DictReader(f))
    with open('data/ddxt_train_data.csv', mode='r', encoding='utf-8') as f:
        data_train = list(csv.DictReader(f))

    data = data_test + data_dev + data_train
    group_list = []
    symptom_list = []
    diagnose_list = []
    med_history_list = []
    allergen_list = []
    on_medicine_list = []

    for item in data:
        if item['group'] not in group_list:
            group_list.append(item['group'])

        symptoms = re.split(r'[、，]', item["symptom"])
        for symptom in symptoms:
            if symptom not in symptom_list:
                symptom_list.append(symptom)

        diagnosis = re.split(r'[、，]', item["diagnosis"])
        for diagnose in diagnosis:
            if diagnose not in diagnose_list and diagnose not in symptom_list:
                diagnose_list.append(diagnose)

        med_histories = re.split(r'[、，]', item["antecedents"])
        for med_history in med_histories:
            if med_history not in med_history_list and med_history not in symptom_list and med_history not in diagnose_list:
                med_history_list.append(med_history)

        allergens = re.split(r'[、，]', item["allergen"])
        for allergen in allergens:
            if allergen not in allergen_list:
                allergen_list.append(allergen)

        on_medicines = re.split(r'[、，]', item["on_medicine"])
        for on_medicine in on_medicines:
            if on_medicine not in on_medicine_list:
                on_medicine_list.append(on_medicine)


    return group_list, symptom_list, diagnose_list, med_history_list, allergen_list, on_medicine_list


def read_drug():
    # file_name = 'data/release_conditions.json'
    with open('data/ddxt_test_data.csv', mode='r', encoding='utf-8') as f:
        data_test = list(csv.DictReader(f))
    with open('data/ddxt_dev_data.csv', mode='r', encoding='utf-8') as f:
        data_dev = list(csv.DictReader(f))
    with open('data/ddxt_train_data.csv', mode='r', encoding='utf-8') as f:
        data_train = list(csv.DictReader(f))

    data = data_test + data_dev + data_train
    drug_list = []

    for item in data:
        drugs = re.split(r'[、，]', item["answer"])
        for drug in drugs:
            if drug not in drug_list:
                drug_list.append(drug)

    return drug_list


# def read_conditions_eng():
#     file_name = '../data/release_conditions.json'
#     condition_list = []
#
#     with open(file_name, mode='r', encoding='utf-8') as f:
#         data = json.load(f)
#
#     for k, v in data.items():
#         name = v['cond-name-eng'].split('/')
#         condition_list.append(name[0])
#
#     return condition_list


if __name__ == '__main__':
    group, evidences, pathologies, c, d, g = read_info()
    print(group)
    print(f'Total symptoms in the dataset: {len(evidences)}\n')
    print(len(pathologies))
    print(len(c))
    print(len(d))
    # print(c)

    # pathologies = read_diagnosis()
    drugs = read_drug()
    # print(drugs)
    print(f'Total drugs in the dataset: {len(drugs)}')
