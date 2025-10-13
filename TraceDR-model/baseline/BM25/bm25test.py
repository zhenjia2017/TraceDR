from rank_bm25 import BM25Okapi
import pickle
import jieba
import json
import numpy as np
from bm25_es import BM25Scoring
from tqdm import tqdm
import evaluation
def chinese_tokenizer(text):
    string = text.replace(",", " ")
    return jieba.lcut(string)

def person_to_query(person,delimiter):
    onmedicine = ""
    for idx,drug in person['on_medicine'].items():
        onmedicine += drug['name']
    query=f"{person['age']} {delimiter} {person['group']} {delimiter} {person['gender']} {delimiter} {person['diagnosis']} {delimiter} {person['symptom']} {delimiter} {person['medhistory']} {delimiter}{onmedicine} {delimiter} {person['allergen']}"
    return query


def get_evidence_text(evidence):
    drugid = evidence["drugid"]
    treat = evidence["treat"]
    caution = evidence["caution"]
    interaction = evidence["interaction"]
    ingredients = evidence["ingredients"]
    treatments_string = ', '.join([item['treat'] for item in treat])
    caution_string = ', '.join([item['crowd']+item['caution_level'] for item in caution])
    interaction_string = ', '.join([item['name'] for item in interaction])
    ingredients_string = ', '.join([item['ingredient'] for item in ingredients])
    name = evidence['name']
    query = f"药名:{name} || 治疗:{treatments_string} || 禁用:{caution_string} || 成分:{ingredients_string} || 相互作用:{interaction_string}"
    return query


def predata_noconcat():
    with open("drugMsg_linux.pkl","rb") as fp:
        drug_Data = pickle.load(fp)
    drug_list=list()
    corpus = list()
    #药品症状不拼接，建立mapping
    for drug in drug_Data:
        drug_id=drug.get('id',0)
        treat_list = drug.get('治疗', [])
        for treat in treat_list:
            after_token = chinese_tokenizer(treat)
            corpus.append(after_token)
            drug_list.append(drug_id)
    with open("top50-interaction-109/test.pkl","rb") as f1:
        people_data=pickle.load(f1)
    #p=people_data[:100]
    query_dict=dict()
    people_only = dict()
    for i, person in people_data.items():
        people_only[str(i)] = person['people']
        diagnosis = person['people'].diagnosis
        symptom = person['people'].symptom
        query_dict[str(i)] = diagnosis + " " + symptom
        # query=str(i)+":"+temp['query']
    return drug_list, corpus, query_dict,people_only

def predata():
    with open("drugMsg_linux.pkl","rb") as fp:
        drug_Data=pickle.load(fp)
    #drug_dict=dict()
    #sentences=list()
    mapping = dict()
    corpus=list()
    for drug in drug_Data:
        treat_list = drug.get('治疗', [])
        for treat in treat_list:
            after_token = chinese_tokenizer(treat)
            mapping_idx=" ".join(after_token)
            mapping_drug = mapping.get(mapping_idx)
            if mapping_drug is None:
                corpus.append(after_token)
                mapping_drug = list()
                mapping_drug.append(drug["id"])
                mapping[mapping_idx]=mapping_drug
            else:

                mapping_drug.append(drug["id"])

    with open("new_val_set.pkl","rb") as f1:
        people_data=pickle.load(f1)
    #p=people_data[:100]
    query_dict=dict()
    for i,person in people_data.items():
        diagnosis=person.diagnosis
        symptom=person.symptom
        query_dict[str(i)]=diagnosis+" "+symptom
        # query=str(i)+":"+temp['query']
    return mapping,corpus,query_dict,people_data


def bm25_select_topk_drugs(dataset='dev'):
    mapping, corpus, query_dict, patient_data=predata3()

    p_at_1_list = list()
    mrr_list = list()
    hit5_list = list()
    jaccard_list = list()
    precison_list = list()
    recall_list = list()
    f1_list = list()
    ddi_list = list()
    ddi_1_list = list()

    for idx,query in tqdm(query_dict.items()):
        gold_answers = list(patient_data[idx].medication.keys())
        bm25_select = BM25Scoring(corpus[idx], mapping[idx],is_retrived = False)
        topk_drug=bm25_select.get_top_evidences(query)
        select_drug = list()
        for drugs in topk_drug:
            select_drug.extend(drugs)
        #print(select_drug)
        # evaluate
        p_at_1 = evaluation.precision_at_1(select_drug, gold_answers)
        mrr = evaluation.mrr_score(select_drug, gold_answers)
        hit_at_5 = evaluation.hit_at_5(select_drug, gold_answers)
        metrics = evaluation.calculate_metrics(select_drug, gold_answers, idx)
        jaccard = metrics['jaccard_similarity']
        precison_k = metrics['average_precision']
        recall_k = metrics['average_recall']
        f1 = metrics['average_f1']
        ddi = metrics['DDI_rate']
        ddi_1 = metrics['DDI_rate@1']

        p_at_1_list.append(p_at_1)
        precison_list.append(precison_k)
        recall_list.append(recall_k)
        f1_list.append(f1)
        jaccard_list.append(jaccard)
        ddi_list.append(ddi)
        mrr_list.append(mrr)
        hit5_list.append(hit_at_5)
        ddi_1_list.append(ddi_1)

    jaccard = sum(jaccard_list) / len(jaccard_list)
    mrr = sum(mrr_list) / len(mrr_list)
    hit5 = sum(hit5_list) / len(hit5_list)
    p_at_1 = sum(p_at_1_list) / len(p_at_1_list)
    p_at_k = sum(precison_list) / len(precison_list)
    recall_at_k = sum(recall_list) / len(recall_list)
    f1_at_k = sum(f1_list) / len(f1_list)
    ddi = sum(ddi_list) / len(ddi_list)
    ddi_1 = sum(ddi_1_list) / len(ddi_1_list)

    print(f"jaccard:{jaccard}")
    print(f"p_at_1:{p_at_1}")
    print(f"mrr:{mrr}")
    print(f"hit5:{hit5}")
    print(f"p_at_k:{p_at_k}")
    print(f"recall_at_k:{recall_at_k}")
    print(f"f1_at_k:{f1_at_k}")
    print(f"ddi:{ddi}")
    print(f"ddi_1:{ddi_1}")

def predata3():
    with open("top50-interaction-109/test.pkl","rb") as fp:
        patient_data=pickle.load(fp)

    drug_dict=dict()
    #sentences=list()
    mapping_all = dict()
    corpus_all=dict()
    for idx,data in patient_data.items():
        topk_drug = data['top_k_drugs']
        corpus = list()
        mapping = dict()
        for drugid,drug in topk_drug.items():
            drug_text = get_evidence_text(drug)
            mapping_drug = mapping.get(drug_text)
            if mapping_drug is None:
                corpus.append(drug_text)
                mapping_drug = list()
                mapping_drug.append(drug["drugid"])
                mapping[drug_text]=mapping_drug
            else:
                mapping_drug.append(drug["drugid"])
        corpus_all[idx] = corpus
        mapping_all[idx] = mapping
    #tokenized_corpus=list(mapping.keys())

    #p=people_data[:100]
    query_dict=dict()
    people_data = dict()
    for i,patient in patient_data.items():
        people_data[i] = patient["people"]
        person = patient["people"]
        diagnosis=person_to_query(person,'||')
        query_dict[i] = diagnosis
        #symptom=person.symptom
        #query_dict[i]=diagnosis+" "+symptom
        # query=str(i)+":"+temp['query']
    return mapping_all,corpus_all,query_dict,people_data


if __name__ == '__main__':
    bm25_select_topk_drugs('test')
    #main()
    #predata_drugconcat()
    # with open("dataset_21958/val_set_21958.pkl","rb") as f1:
    #     data=pickle.load(f1)
    # drug_list=list()
    #
    # for i,person in data.items():
    #     every=[drug['drugid'] for drug in person.medication]
    #     every=list(set(every))
    #     drug_list.append(every)
    # # 计算最长列表
    # longest_list = len(max(drug_list, key=len))
    # # 计算最短列表
    # shortest_list = len(min(drug_list, key=len))
    # # 计算所有列表的平均长度
    # average_length = sum(map(len, drug_list)) / len(drug_list)
    # print("最大药品数量:", longest_list)
    # print("最小药品数量:", shortest_list)
    # print("平均药品数量:", average_length)
    # print("ok")
    #main()