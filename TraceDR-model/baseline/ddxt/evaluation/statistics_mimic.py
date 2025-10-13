import pickle
import os
import json
import csv
import random

path = "./results/"
files = ["mimic2drugrec_train.pkl", "mimic2drugrec_dev.pkl", "mimic2drugrec_test.pkl"]


def read_dataset_files(files):
    datas = []
    for file in files:
        with open(os.path.join(path, file), "rb") as f:
            data = pickle.load(f)
            datas += data
    return datas

def count_ehr(data):
    print (f"The total number of patients or ehrs: {len(data)}")

def count_conflict_data():
    with open(os.path.join(path, "train.json"), "r", encoding="utf8") as f:
        data = json.load(f)
        train = [item for item in data if item["conflict"]]

    with open(os.path.join(path, "dev.json"), "r", encoding="utf8") as f:
        data = json.load(f)
        dev = [item for item in data if item["conflict"]]

    with open(os.path.join(path, "test.json"), "r", encoding="utf8") as f:
        data = json.load(f)
        test = [item for item in data if item["conflict"]]

    print (len(train))
    print (len(dev))
    print (len(test))



def count_disease_ingredient_contraindication(data):
    diseases = set()
    ingredients = set()
    drugs = []
    drug_names = []
    shown_drugs_number = {}
    shown_diseases_number = {}
    contraindications = set()
    recommended_drugs = set()
    on_medicine_drugs = set()
    conflict_drugs = set()
    print (len(data))
    for item in data:
        diseases |= set(item["diagnosis"])
        for di in item["diagnosis"]:
            if di not in shown_diseases_number:
                shown_diseases_number[di] = set()
            shown_diseases_number[di].add(item["id"])
        if "antecedents" in item:
            diseases |= set(item["antecedents"])
            for di in item["antecedents"]:
                if di not in shown_diseases_number:
                    shown_diseases_number[di] = set()
                shown_diseases_number[di].add(item["id"])

            # if type(item["antecedents"]) == list:
            #     diseases |= set(item["antecedents"])
            # else:
            #     diseases.add(item["antecedents"])
        for med in item["medicine"]:
            drugs.append(med["ndc"])
            drug_names.append(med["name"])
            if med["ndc"] not in shown_drugs_number:
                shown_drugs_number[med["ndc"]] = set()
            shown_drugs_number[med["ndc"]].add(item['id'])
        # if not item["on_medicine"]:
        #     continue
        # for med in item["on_medicine"]:
        #     #print (med)
        #     if "#" in med["id"]:
        #         continue
        #     drugs.append(med["ndc"])
        #     on_medicine_drugs.add(med["ndc"])
        #     if med["ndc"] not in shown_drugs_number:
        #         shown_drugs_number[med["ndc"]] = 0
        #     shown_drugs_number[med["ndc"]] += 1
        #     ingredients |= set(med["ingredients"].split("; "))
        #     contraindications |= set(med["caution"].split("; "))

        # for med in item["conflict"]:
        #     drugs.append(med["id"])
        #     ingredients |= set(med["ingredients"].split("; "))
        #     contraindications |= set(med["caution"].split("; "))\
    print(f"The total number of patients: {len(data)}")
    print(f"The total number of drugs: {len(set(drugs))}")
    print(f"The total number of drug names: {len(set(drug_names))}")
    # print(f"The total number of recommended drugs: {len(recommended_drugs)}")
    # print(f"The total number of on medicine drugs: {len(on_medicine_drugs)}")
    print(f"The total number of diseases: {len(diseases)}")
    # print(f"The total number of ingredients: {len(ingredients)}")
    # print(f"The total number of contraindications: {len(contraindications)}")
    count_disease_dict = {}
    for key, value in shown_diseases_number.items():
        count_disease_dict[key] = len(value)

    count_drug_dict = {}
    for key, value in shown_drugs_number.items():
        count_drug_dict[key] = len(value)


    sorted_dict = dict(sorted(count_disease_dict.items(), key=lambda item: item[1], reverse=True))
    n = 0
    for key, value in sorted_dict.items():
        print (key)
        print (value)
        n += 1
        if n > 5:
            break
    sorted_disease_list = sorted(count_disease_dict.values(), reverse=True)
    print(sorted_disease_list[0:5])
    print(sorted(count_disease_dict.values(), reverse=True)[-1])
    print(sum(count_disease_dict.values())/len(shown_diseases_number))

    sorted_dict = dict(sorted(count_drug_dict.items(), key=lambda item: item[1], reverse=True))
    n = 0
    for key, value in sorted_dict.items():
        print(key)
        print(value)
        n += 1
        if n > 5:
            break
    print(sorted(count_drug_dict.values(), reverse=True)[0:10])
    print(sorted(count_drug_dict.values(), reverse=True)[-1])
    print(sum(count_drug_dict.values())/len(shown_drugs_number))

def count_ddi(data):
    mkg_dictionary = "drugMsg_linux_dict.pkl"
    with open(os.path.join(path, mkg_dictionary), "rb") as f:
        mkg = pickle.load(f)
    drugs = []
    drugs_id = {}
    for instance in data:
        for med in instance["medicine"]:
            if med["id"] not in drugs_id:
                drugs_id.update({med["id"]: mkg[med["id"]]})

        for med in instance["on_medicine"]:
            if med["id"] not in drugs_id:
                drugs_id.update({med["id"]: mkg[med["id"]]})

        for med in instance["conflict"]:
            if med["id"] not in drugs_id:
                drugs_id.update({med["id"]: mkg[med["id"]]})

    print (len(drugs_id.keys()))

    drug_no_interaction = []
    drug_no_ingredient = []
    drug_with_interaction = []
    drug_with_ingredient = []

    interaction_map_drug = {}
    ingredients_map_drug = {}

    for id in drugs_id:
        if "相互作用" in drugs_id[id]:
            if len(drugs_id[id]["相互作用"]) != 0:
                drug_with_interaction.append(id)
                for item in drugs_id[id]["相互作用"]:
                    if item not in interaction_map_drug:
                        interaction_map_drug[item] = set()
                    interaction_map_drug[item].add(id)

            else:
                drug_no_interaction.append(id)
        else:
            drug_no_interaction.append(id)

        if "成分" in drugs_id[id]:
            if len(drugs_id[id]["成分"]) != 0:
                drug_with_ingredient.append(id)
                for item in drugs_id[id]["成分"]:
                    if item not in ingredients_map_drug:
                        ingredients_map_drug[item] = set()
                    ingredients_map_drug[item].add(id)
            else:
                drug_no_ingredient.append(id)
        else:
            drug_no_ingredient.append(id)


    print(f"drug_with_interaction: {len(drug_with_interaction)}")
    print(f"drug_no_interaction: {len(drug_no_interaction)}")

    print(f"drug_with_ingredient: {len(drug_with_ingredient)}")
    print(f"drug_no_ingredient: {len(drug_no_ingredient)}")

    ddi_pair = []
    ddi_set = set()

    interaction_drug_in_ingredients = [item for item in interaction_map_drug.keys() if item in ingredients_map_drug]
    print (len(interaction_drug_in_ingredients))
    for item in interaction_drug_in_ingredients:
        drugs_1 = interaction_map_drug[item]
        drugs_2 = ingredients_map_drug[item]

        for med1 in drugs_1:
            for med2 in drugs_2:
                if med1 != med2:
                    pair = tuple(sorted([med1, med2]))  # 确保顺序一致
                    if pair not in ddi_set:
                        ddi_set.add(pair)
                        ddi_pair.append([med1, med2])

    print (f"ddi_pair: {len(ddi_set)}")
    print(f"ddi_pair_example: {ddi_pair[0:4]}")

def count_interaction(data):

    drugs = []
    drug_no_interaction = []
    drug_no_ingredient = []
    drug_with_interaction = []
    drug_with_ingredient = []
    ddi_pair = []
    interaction_map_drug = {}
    ingredients_map_drug = {}
    for instance in data:
        for med in instance["medicine"]:

            if "patient" not in med:
                med.update({"patient": []})
            if len(med["interaction"]) != 0:
                med["patient"].append(instance["id"])
                drug_with_interaction.append(med)
            elif len(med["interaction"]) == 0:
                med["patient"].append(instance["id"])
                drug_no_interaction.append(med)
                interactions = med["interaction"].split("; ")
                for item in interactions:
                    if item not in interaction_map_drug:
                        interaction_map_drug[item] = set()
                    interaction_map_drug[item].add(med["id"])


        for med in instance["on_medicine"]:

            if "patient" not in med:
                med.update({"patient": []})
            if len(med["interaction"]) == 0:
                drug_no_interaction.append(med)
                med["patient"].append(instance["id"])
            elif len(med["interaction"]) != 0:
                drug_with_interaction.append(med)
                med["patient"].append(instance["id"])
                interactions = med["interaction"].split("; ")
                for item in interactions:
                    if item not in interaction_map_drug:
                        interaction_map_drug[item] = set()
                    interaction_map_drug[item].add(med["id"])


        for med in instance["conflict"]:

            if "patient" not in med:
                med.update({"patient": []})
            if len(med["interaction"]) == 0:
                med["patient"].append(instance["id"])
                drug_no_interaction.append(med)
            elif len(med["interaction"]) != 0:
                med["patient"].append(instance["id"])
                drug_with_interaction.append(med)
                interactions = med["interaction"].split("; ")
                for item in interactions:
                    if item not in interaction_map_drug:
                        interaction_map_drug[item] = set()
                    interaction_map_drug[item].add(med["id"])


    with open(os.path.join(path, "drug_with_interaction.json"), "w", encoding='utf-8') as fp:
        fp.write(json.dumps(drug_with_interaction, ensure_ascii=False, indent=4))

    with open(os.path.join(path, "drug_no_interaction.json"), "w", encoding='utf-8') as fp:
        fp.write(json.dumps(drug_no_interaction, ensure_ascii=False, indent=4))

    drug_no_interactions = set([item["id"] for item in drug_no_interaction])
    drug_with_interactions = set([item["id"] for item in drug_with_interaction])

    drug_in_both_sets = drug_no_interactions.intersection(drug_with_interactions)

    print(f"drug_no_interaction: {len(drug_no_interactions)}")
    print(f"drug_with_interaction: {len(drug_with_interactions)}")
    print(f"drug_in_both_set: {len(drug_in_both_sets)}")

    output1 = [item for item in drug_no_interaction if item["id"] in drug_in_both_sets]
    output2 = [item for item in drug_with_interaction if item["id"] in drug_in_both_sets]

    sorted_output = sorted(output1+output2, key=lambda x: x["id"])  # Sort by "id"
    with open(os.path.join(path, "drug_in_two_sets_of_interaction.json"), "w", encoding='utf-8') as fp:
        fp.write(json.dumps(sorted_output, ensure_ascii=False, indent=4))


    #print (set(drug_no_interaction).intersection(set(drug_with_interaction)))
    print(len(set(drug_no_interactions).intersection(set(drug_with_interactions))))
    print(list(set(drug_no_interactions).intersection(set(drug_with_interactions)))[0:5])
    print(len(set(drug_with_interactions).intersection(set(drug_no_interactions))))
    print(list(set(drug_with_interactions).intersection(set(drug_no_interactions)))[0:5])

def count_allergen(data):
    allergen = list()
    for item in data:
        if item["allergen"] != "无":
            allergen.append(item["id"])
    print (f'The total number of patients with allergen {len(allergen)}')
def count_group_gender(data):
    group = {}
    gender = {}
    for item in data:
        if item["gender"] not in gender:
            gender[item["gender"]] = []
        gender[item["gender"]].append(item["gender"])
    for g in group:
        print (f"{g}: {len(group[g])}")
    for g in gender:
        print(f"{g}: {len(gender[g])}")

def count_age(data):
    #儿童（<12）; 青少年（12-17）; 成人(18-64); 老人(>=65)
    children = []
    adolescents = []
    adults = []
    elderlies = []

    for item in data:
        age = int(item["age"])
        if age < 12:
            children.append(item)
        elif age >= 12 and age < 18:
            adolescents.append(item)
        elif age >= 18 and age < 65:
            adults.append(item)
        else:
            elderlies.append(item)

    print (f"children: {len(children)}")
    print (f"adolescents: {len(adolescents)}")
    print (f"adults: {len(adults)}")
    print (f"elderlies: {len(elderlies)}")


data = read_dataset_files(files)
count_ehr(data)
count_age(data)
count_group_gender(data)
count_disease_ingredient_contraindication(data)
#count_allergen(data)
#count_ddi(data)

#count_conflict_data()



