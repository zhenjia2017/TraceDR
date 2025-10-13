import json
from tqdm import tqdm
import random
from rank_bm25 import BM25Okapi
import csv
import pickle
from torch.utils.data import Dataset
from ExplaiDR.library.utils import get_config, get_logger
from evaluation import candidate_in_answers

class DatasetES(Dataset):
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__, config)
        self.pos_sample_num = config["bert_max_pos_evidences_per_source"]
        self.neg_sample_num = config["bert_max_neg_evidences"]
        self.bert_sample_method = config["es_sample_method"]

    def person_to_query(self, person, delimiter):
        """
        Convert patient information into a query string.
        """
        on_medicine_str = "".join(drug['name'] for drug in person['on_medicine'].values())
        query_elements = [
            person['age'], person['group'], person['gender'],
            person['diagnosis'], person['symptom'],
            person['medhistory'], on_medicine_str, person['allergen']
        ]
        return f" {delimiter} ".join(query_elements)

    def get_evidence_text(self, evidence, delimiter):
        """Generate descriptive text based on drug evidence."""
        treatments_string = ', '.join([item['treat'] for item in evidence["treat"]])
        caution_string = ', '.join([item['crowd'] + item['caution_level'] for item in evidence["caution"]])
        interaction_string = ', '.join([item['name'] for item in evidence["interaction"]])
        ingredients_string = ', '.join([item['ingredient'] for item in evidence["ingredients"]])

        name = evidence['name']
        query = (
            f"药名:{name} {delimiter} 治疗:{treatments_string} {delimiter} 禁用:{caution_string} {delimiter} "
            f"成分:{ingredients_string} {delimiter} 相互作用:{interaction_string}"
        )
        return query


    # construct training dataset for bert model
    def process_dataset(self, dataset_path):

        def person_to_query(person,delimiter):
            onmedicine = ""
            for idx,drug in person.on_medicine.items():
                onmedicine += drug['name']
            query=f"{person.age} {delimiter} {person.group} {delimiter} {person.gender} {delimiter} {person.diagnosis} {delimiter} {person.symptom} {delimiter} {person.medhistory} {delimiter}{onmedicine} {delimiter} {person.allergen}"
            return query

        def get_evidence_text(evidence):
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

        # process data
        training_dataset = list()
        with open(dataset_path,"rb") as f1:
            temple_train = pickle.load(f1)
        positive_list = dict()
        for idx,item in temple_train.items():
            positive_evidences = list()
            negative_evidences = list()
            top_k_drugs = item['top_k_drugs']
            patient = item['people']
            gold_answers = patient.medication
            true_id_list = list(gold_answers.keys())
            all_id_list = list(top_k_drugs.keys())
            query = person_to_query(patient,"||")
            if not candidate_in_answers(all_id_list,true_id_list):
                continue
            for id,drug in top_k_drugs.items():
                evidence_text = get_evidence_text(drug)
                if id in true_id_list:
                    positive_evidences.append([idx, query, evidence_text, 1])
                else:
                    negative_evidences.append([idx, query, evidence_text, 0])
            if len(positive_evidences) == 0:
                continue
            if len(positive_evidences) >= self.pos_sample_num:
                selected_item = random.sample(positive_evidences, self.pos_sample_num)
                training_dataset += selected_item
                positive_list[idx] = selected_item
            else:
                self.pos_sample_num = len(positive_evidences)
                training_dataset += positive_evidences
                positive_list[idx] = positive_evidences
            # sample negative evidences
            sample_num = self.neg_sample_num
            if len(negative_evidences) < self.neg_sample_num:
                sample_num = len(negative_evidences)

            # random sample negative
            training_dataset += random.sample(negative_evidences, sample_num)
        return training_dataset,positive_list

    def load_data(self, train_path, dev_path):
        training_dataset = self.process_dataset(train_path)
        dev_dataset = self.process_dataset(dev_path)
        return training_dataset, dev_dataset

    def write_to_tsv(self, output_path, train_list):
        with open(output_path, "wt") as file:
            writer = csv.writer(file, delimiter="\t")
            #header = ["ques_id", "query", "evidence", "label", "source"]
            header = ["ques_id", "query", "evidence", "label"]
            writer.writerow(header)
            writer.writerows(train_list)
            
    def load_test_data(self, test_path):
        test_dataset = list()
        with open(test_path, "rb") as f1:
            temple_test = pickle.load(f1)
        positive_list = dict()
        for idx, item in temple_test.items():
            positive_evidences = list()
            #negative_evidences = list()
            top_k_drugs = item['top_k_drugs']
            patient = item['people']
            gold_answers = patient.medication
            true_id_list = list(gold_answers.keys())
            all_id_list = list(top_k_drugs.keys())
            query = self.person_to_query(patient, "||")
            if not candidate_in_answers(all_id_list, true_id_list):
                continue
            for i,drug in top_k_drugs.items():
                evidence_text = self.get_evidence_text(drug)
                if i in true_id_list:
                    positive_evidences.append([idx, query, evidence_text, 1])
                    test_dataset.append([idx, query, evidence_text, 1])
                else:
                    #negative_evidences.append([idx, query, evidence_text, 0])
                    test_dataset.append([idx, query, evidence_text, 0])
            positive_list[idx] = positive_evidences
        return test_dataset,positive_list


