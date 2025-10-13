import json
import numpy as np
import random
import torch
import pickle
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
import jieba


def chinese_tokenizer(text):
    return [jieba.lcut(doc) for doc in text]


def person_to_query(person, delimiter):
    onmedicine = ','.join([item['name'] for item in person['on_medicine']])
    symptom = ','.join(person['symptom'])
    diagnosis = ','.join(person['diagnosis'])
    antecedents = ','.join(person['antecedents'])
    allergen = ','.join(person['allergen']) 
    query = f"{person['age']} {delimiter} {person['group']} {delimiter} {person['gender']} {delimiter} {diagnosis} {delimiter} {symptom} {delimiter} {antecedents} {delimiter} {onmedicine} {delimiter} {allergen}"
    return query


def collate_fn(batch):
    """Collate the input data for the batch."""
    def _is_vector(obj):
        return (type(obj) is torch.Tensor) or (
            type(obj).__module__ == np.__name__ and obj.dtype == np.float32
        )

    elem = batch[0]
    instances = {
        key: default_collate([d[key] for d in batch]) for key in elem if _is_vector(elem[key])
    }
    mappings = {key: [d[key] for d in batch] for key in elem if not _is_vector(elem[key])}
    instances.update(mappings)
    return instances


class ContinueWithNext(Exception):
    pass


class DatasetGNN(Dataset):
    def __init__(self, data_path, max_entities=50, max_evidences=100, train=False):
        self.max_entities = max_entities
        self.max_evidences = max_evidences
        self.instances = self.prepare_data(data_path, train)

    def __getitem__(self, idx):
        return self.instances[idx]

    def __len__(self):
        return len(self.instances)

    def prepare_data(self, data_path, train=False):
        """Create matrix representations of each instance."""
        with open(data_path, "rb") as f3:
            people_data = pickle.load(f3)
        
        instances = list()
        print("Building adjacency matrix...")
        num = 0
        for idx, turn in tqdm(people_data.items()):
            if num >= 10:
                break
            num += 1
            try:
                instance = self.prepare_turn(idx, turn, train)
                instances.append(instance)
            except ContinueWithNext:
                continue

        ans_pres_list = [1 if sum(instance["entity_labels"]) > 0 else 0 for instance in instances]
        ans_pres = sum(ans_pres_list) / len(ans_pres_list)
        print(f"Answer presence: {ans_pres:.3f}, Number of questions: {len(ans_pres_list)}")
        return instances

    def prepare_turn(self, key, turn, train=False):
        """Prepare a single turn for training/inference."""
        question = person_to_query(turn['people'], "||")
        all_select_drug = list(turn["top_k_drugs"].values())
        true_id_list = [item['drugid'] for item in turn['people']['medicine']]
        onmedicine = turn['people']["on_medicine"]
        
        evidences = list()
        evidences.extend(onmedicine)
        evidences.extend(all_select_drug)

        if len(evidences) > self.max_entities:
            evidences = evidences[:self.max_entities]

        # Initialize mappings and matrices
        entity_to_id = dict()
        evidence_to_id = dict()
        id_to_entity = np.zeros(self.max_entities, dtype=object)
        id_to_evidence = np.zeros(self.max_evidences, dtype=object)
        
        entities_list = list()
        evidences_list = list()
        
        ent_to_ev = np.zeros((self.max_entities, self.max_evidences), dtype=np.float32)
        ev_to_ent = np.zeros((self.max_evidences, self.max_entities), dtype=np.float32)
        
        entity_labels = np.zeros(self.max_entities, dtype=np.float32)
        evidence_labels = np.zeros(self.max_evidences, dtype=np.float32)
        
        num_entities = 0
        num_evidences = 0

        # Process each drug (entity)
        for i, drug in enumerate(evidences):
            drug_id = drug["drugid"]
            en_id = num_entities
            entity_to_id[drug_id] = en_id
            id_to_entity[en_id] = drug_id
            num_entities += 1
            
            contain_evidences = list()
            
            # Process different types of evidence
            evidence_types = [
                ("treat", "treat"),
                ("caution", "crowd"), 
                ("interaction", "name"),
                ("ingredients", "ingredient")
            ]
            
            for etype, key_name in evidence_types:
                for item in drug[etype]:
                    label = item[key_name]
                    ev_id = evidence_to_id.get(label)
                    if ev_id is None:
                        if num_evidences >= (self.max_evidences - 1):
                            continue
                        ev_id = num_evidences
                        evidence_to_id[label] = num_evidences
                        id_to_evidence[ev_id] = label
                        num_evidences += 1
                        record = {'id': ev_id, 'label': label}
                        evidences_list.append(record)
                    
                    contain_evidences.append({"id": ev_id, "label": label})
                    ent_to_ev[en_id, ev_id] = 1
                    ev_to_ent[ev_id, en_id] = 1

            # Create entity record
            temp = {
                "id": en_id,
                "name": drug['name'],
                "instruction": drug,
                "connect_property": contain_evidences,
                "is_answer": drug_id in true_id_list
            }
            
            if drug_id in true_id_list:
                entity_labels[en_id] = 1
                
            entities_list.append(temp)

        # Convert to tensors
        ent_to_ev = torch.from_numpy(ent_to_ev)
        ev_to_ent = torch.from_numpy(ev_to_ent)
        entity_labels = torch.from_numpy(entity_labels).type(torch.LongTensor)
        evidence_labels = torch.from_numpy(evidence_labels).type(torch.LongTensor)

        # Create masks
        entity_mask = torch.FloatTensor(num_entities * [1] + (self.max_entities - num_entities) * [0])
        evidence_mask = torch.FloatTensor(num_evidences * [1] + (self.max_evidences - num_evidences) * [0])

        # Padding
        entities_list = entities_list + (self.max_entities - num_entities) * [
            {"id": "", "name": "", "instruction": "", "connect_property": "", "is_answer": ""}
        ]
        evidences_list = evidences_list + (self.max_evidences - num_evidences) * [
            {"id": "", "label": ""}
        ]

        # Check if we have answers for training
        entity_labels_sum = torch.sum(entity_labels)
        if train and not entity_labels_sum:
            raise ContinueWithNext("No answers available for training")

        # Normalize adjacency matrices
        vec = torch.sum(ent_to_ev, dim=0)
        vec[vec == 0] = 1
        ent_to_ev = ent_to_ev / vec

        vec = torch.sum(ev_to_ent, dim=0)
        vec[vec == 0] = 1
        ev_to_ent = ev_to_ent / vec

        return {
            "question_id": key,
            "tsf": question,
            "entities": entities_list,
            "entity_mask": entity_mask,
            "evidences": evidences_list,
            "evidence_mask": evidence_mask,
            "ent_to_ev": ent_to_ev,
            "ev_to_ent": ev_to_ent,
            "entity_labels": entity_labels,
            "evidence_labels": evidence_labels,
            "id_to_entity": id_to_entity,
            "id_to_evidence": id_to_evidence,
            "gold_answers": true_id_list
        } 