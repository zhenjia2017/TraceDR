import json
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
import pickle
import jieba
from py2neo import Graph
import os
import time
from bm25_es import BM25Scoring

# 连接图数据库
uri = "http://localhost:7474"

def chinese_tokenizer(text):
    string = text.replace(",", " ")
    return jieba.lcut(string)

def person_to_query(person, delimiter):
    onmedicine = ','.join([item['name'] for item in person['on_medicine']])
    group = ','.join(person['group'])
    symptom = ','.join(person['symptom'])
    diagnosis = ','.join(person['diagnosis'])
    antecedents = ','.join(person['antecedents'])
    allergen = ','.join(person['allergen']) 
    query = f"{person['age']} {delimiter} {group} {delimiter} {person['gender']} {delimiter} {diagnosis} {delimiter} {symptom} {delimiter} {antecedents} {delimiter} {onmedicine} {delimiter} {allergen}"
    return query

def get_evidence_text(evidence):
    treat = evidence["treat"]
    caution = evidence["caution"]
    interaction = evidence["interaction"]
    ingredients = evidence["ingredients"]

    treat_values = [item['treat'] for item in treat if item['treat'] is not None]
    if not treat_values:
        treatments_string = 'None'
    else:
        treatments_string = ', '.join(treat_values)

    caution_values = [item['crowd'] + item['caution_level'] for item in caution 
                     if item.get('crowd') is not None and item.get('caution_level') is not None]
    caution_string = ', '.join(caution_values) if caution_values else 'None'

    interaction_values = [item['name'] for item in interaction if item.get('name') is not None]
    interaction_string = ', '.join(interaction_values) if interaction_values else 'None'

    ingredients_values = [item['ingredient'] for item in ingredients if item.get('ingredient') is not None]
    ingredients_string = ', '.join(ingredients_values) if ingredients_values else 'None'
   
    name = evidence['name']
    query = f"药名:{name} || 治疗:{treatments_string} || 禁用:{caution_string} || 成分:{ingredients_string} || 相互作用:{interaction_string}"
    return query

class ContinueWithNext(Exception):
    pass

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

class BM25Retriever:
    def __init__(self):
        self.graph = Graph(uri)
        self.bm25 = self._load_bm25_model()
    
    def _load_bm25_model(self):
        """Instantiate retrieve stage of ExplaiDR pipeline."""
        drug_list = list()
        corpus = list()
        # get all the drugids and treatment
        drug_data = self.get_treatment_from_mkg()
        # construct corpus and mapping for bm25
        for drug_id, treatment in drug_data.items():
            for treat in treatment:
                after_token = chinese_tokenizer(treat['treat'])
                corpus.append(after_token)
                drug_list.append(drug_id)
        # construct bm25 model
        bm25_select = BM25Scoring(corpus, drug_list, is_retrived=True)
        return bm25_select

    def get_treatment_from_mkg(self):
        """retrieve all drugid and treatment from MKG for bm25 retrievel"""
        drugMsg_dict = dict()
        # retrieve All drug ids from the MKG
        query1 = """MATCH (n:`药品`) RETURN id(n) AS drugid"""
        search = self.graph.run(query1).data()
        # retrieve the treatment associated with each drug
        for item in search:
            drugid = item["drugid"]
            query2 = f"""
                    MATCH p1=(drug:`药品`)-[:治疗*0..3]->(treatment:`病症`)
                    WHERE id(drug) = {drugid}
                    RETURN id(treatment) AS treatid,treatment.name AS treat
                    """
            treatment_list = self.graph.run(query2).data()
            drugMsg_dict[drugid] = treatment_list
        return drugMsg_dict

    def retrieve_drugs(self, patient):
        """retrieve topk drugs according to patient"""
        diagnosis = patient['diagnosis']
        symptom = patient['symptom']
        query = diagnosis + " " + symptom
        topk_drugs = self.bm25.get_top_evidences(query)
        return topk_drugs

    def get_drugmsg_from_mkg(self, drugid):
        """get drug message"""
        search = self.graph.run(
            """
            MATCH (drug:`药品`)
            WHERE id(drug) = $drugid
            
            WITH drug, drug.name AS name, drug.number AS CMAN
            OPTIONAL MATCH p1=(drug)-[:用药*0..2]->(fact:`知识组`)-[:用药]->(crowd:`人群`),
                          p2=(fact)-[:用药结果]->(useResult:`用药结果级别`)
            WITH drug, name, CMAN, 
                 collect(DISTINCT {crowdid: id(crowd), crowd: crowd.name, useresultid: id(useResult), useresult: useResult.name}) AS crowdInfo
            
            OPTIONAL MATCH p3=(drug)-[:治疗*0..3]->(treatment:`病症`)
            WITH drug, name, CMAN, crowdInfo,
                 collect(DISTINCT {treatid: id(treatment), treat: treatment.name}) AS treatmentInfo
            
            OPTIONAL MATCH p4=(drug)-[:成分*0..3]->(ingre:`药物`)
            WITH drug, name, CMAN, crowdInfo, treatmentInfo,
                 collect(DISTINCT {ingredientId: id(ingre), ingredient: ingre.name}) AS ingredients
            
            OPTIONAL MATCH p5=(drug)-[:相互作用*0..3]->(inter:`药物`)
            WITH drug, name, CMAN, crowdInfo, treatmentInfo, ingredients,
                 collect(DISTINCT {interactionId: id(inter), interaction: inter.name}) AS interactions
            
            RETURN name, CMAN, crowdInfo, treatmentInfo, ingredients, interactions
            """, drugid=drugid)
        result = search.data()[0]
        
        # 处理结果
        caution = [{"crowd_id": item["crowdid"], "crowd": item["crowd"], 
                   "caution_levelid": item["useresultid"], "caution_level": item["useresult"]} 
                   for item in result["crowdInfo"]]
        
        treat = [{"treat_id": item["treatid"], "treat": item["treat"]} 
                for item in result["treatmentInfo"]]
        
        ingredients_list = [{"ingredient_id": item['ingredientId'], "ingredient": item["ingredient"]} 
                           for item in result["ingredients"]]
        
        interaction_list = [{"interaction_id": item['interactionId'], "name": item["interaction"]} 
                           for item in result["interactions"]]
            
        record = {
            "drugid": drugid,
            "name": result["name"],
            "CMAN": result["CMAN"],
            "treat": treat,
            "caution": caution,
            "ingredients": ingredients_list,
            "interaction": interaction_list
        }
        return record

class TraceDRDataset(Dataset):
    def __init__(self, data_path, max_entities=100, max_evidences=50, train=False, 
                 tsf_delimiter=" || ", max_pos_evidences=10):
        self.max_entities = max_entities
        self.max_evidences = max_evidences
        self.train = train
        self.tsf_delimiter = tsf_delimiter
        self.max_pos_evidences = max_pos_evidences
        
        # 加载数据
        self.instances = self.prepare_data(data_path, train)

    def __getitem__(self, idx):
        return self.instances[idx]

    def __len__(self):
        return len(self.instances)

    def prepare_data(self, data_path, train=False):
        """prepare data for training"""
        with open(data_path, "rb") as f:
            dataset = pickle.load(f)

        instances = list()
        # num = 0
        for idx, turn in tqdm(dataset.items()):
            try:
                instance = self.prepare_turn(idx, turn, train)
                instances.append(instance)
            except ContinueWithNext:
                continue
        ans_pres_list = [1 if sum(instance["evidence_labels"]) > 0 else 0 for instance in instances]
        ans_pres = sum(ans_pres_list) / len(ans_pres_list)
        print(f"Answer presence: {ans_pres:.3f}, Number of questions: {len(ans_pres_list)}")
        return instances

    def prepare_turn(self, idx, turn, train=False):
        """prepare one sample"""
        # Basic information
        tsf = person_to_query(turn["people"], self.tsf_delimiter)
        question = person_to_query(turn["people"], self.tsf_delimiter)
        topk_drugs = list(turn["top_k_drugs"].values())
        gold_answers = [item['drugid'] for item in turn['people']['medicine']]
        onmedicine = list(turn['people']["on_medicine"])

        evidences = list()
        evidences.extend(onmedicine)
        evidences.extend(topk_drugs)

        if len(evidences) > self.max_evidences - len(onmedicine):
            evidences = evidences[:(self.max_evidences - len(onmedicine))]

        # check if the answer exists
        if train:
            has_answer = any(evidence["drugid"] in gold_answers for evidence in evidences)
            if not has_answer:
                raise ContinueWithNext("No answer present in top evidences.")

        # mappings from entities and evidences to their index in the matrices
        entity_to_id = dict()
        evidence_to_id = dict()

        # mappings from local ids in matrixes to entity/evidence
        id_to_entity = np.zeros(self.max_entities, dtype=object)
        id_to_evidence = np.zeros(self.max_evidences, dtype=object)

        # initialize evidence and entity matrixes: |E| x d, |Ɛ| x d
        entities_list = list()
        evidences_list = list()

        # mappings from local ids in matrixes to entity/evidence
        ent_to_ev = np.zeros((self.max_entities, self.max_evidences), dtype=np.float32)
        ev_to_ent = np.zeros((self.max_evidences, self.max_entities), dtype=np.float32)
        
        # gold answer distribution
        entity_labels = np.zeros(self.max_entities, dtype=np.float32)
        evidence_labels = np.zeros(self.max_evidences, dtype=np.float32)

        num_entities = 0
        num_evidences = 0

        # iterate through evidences retrieved
        for evidence in evidences:
            evidence_text = get_evidence_text(evidence)
            evidence["evidence_text"] = evidence_text
            contain_entities = list()
            
            # add evidence
            g_ev_id = evidence_to_id.get(evidence["drugid"])
            if g_ev_id is None:
                g_ev_id = num_evidences
                id_to_evidence[g_ev_id] = evidence
                num_evidences += 1
                evidence_to_id[evidence["drugid"]] = g_ev_id
                evidence["g_id"] = g_ev_id
                evidences_list.append(evidence)

            # add drug entity
            drug_entity = {
                "id": evidence["drugid"],
                "g_id": num_entities,
                "drugid": evidence["drugid"],
                "label": evidence["name"],
                "CMAN": evidence["CMAN"],
                "type": "药品"
            }
            contain_entities.append(drug_entity)

            if num_entities < (self.max_entities - 1):
                entity_id = evidence["drugid"]
                id_to_entity[num_entities] = drug_entity
                entity_to_id[entity_id] = num_entities
                entities_list.append(drug_entity)
                
                # set answer label
                if evidence["drugid"] in gold_answers:
                    entity_labels[num_entities] = 1
                    drug_entity["is_answer"] = True
                else:
                    drug_entity["is_answer"] = False
                    
                #  set entries in adj. matrixes
                ent_to_ev[num_entities, g_ev_id] = 1
                ev_to_ent[g_ev_id, num_entities] = 1
                num_entities += 1

            # add other entity types
            for entity_type, entities_data in [
                ("治疗", evidence["treat"]), 
                ("禁用", evidence["caution"]),
                ("成分", evidence["ingredients"]),
                ("相互作用", evidence["interaction"])
            ]:
                for entity in entities_data:
                    if num_entities >= (self.max_entities - 1):
                        break
                        
                    entity["type"] = entity_type
                    if entity_type == "治疗":
                        entity["id"] = entity["treat_id"]
                        entity_id = entity["treat"]
                        entity["label"] = entity_id
                    elif entity_type == "禁用":
                        entity["id"] = entity["crowd_id"]
                        entity_id = entity["crowd"]
                        entity["label"] = entity_id
                    elif entity_type == "成分":
                        entity["id"] = entity.get("ingredient_id")
                        entity_id = entity.get("ingredient")
                        entity["label"] = entity_id
                    elif entity_type == "相互作用":
                        entity["id"] = entity.get("interaction_id")
                        entity_id = entity["name"]
                        entity["label"] = entity_id

                    g_ent_id = entity_to_id.get(entity_id)
                    if g_ent_id is None:
                        g_ent_id = num_entities
                        id_to_entity[g_ent_id] = entity
                        num_entities += 1
                        entity_to_id[entity_id] = g_ent_id
                        entity["g_id"] = g_ent_id
                        entities_list.append(entity)
                    else:
                        entity["g_id"] = g_ent_id

                    entity["is_answer"] = False
                    contain_entities.append(entity)
                    ent_to_ev[g_ent_id, g_ev_id] = 1
                    ev_to_ent[g_ev_id, g_ent_id] = 1

            evidence["contain_entities"] = contain_entities
            
            # 标记answering evidence
            has_answer_entity = any(entity["drugid"] in gold_answers for entity in contain_entities if entity["type"] == "药品")
            if has_answer_entity:
                evidence_labels[g_ev_id] = 1
                evidence["is_answering_evidence"] = True
            else:
                evidence["is_answering_evidence"] = False

        # transform to tensor
        ent_to_ev = torch.from_numpy(ent_to_ev).to_sparse()
        ent_to_ev.requires_grad = False
        ev_to_ent = torch.from_numpy(ev_to_ent).to_sparse()
        ev_to_ent.requires_grad = False
        entity_labels = torch.from_numpy(entity_labels).type(torch.LongTensor)
        entity_labels.requires_grad = False
        evidence_labels = torch.from_numpy(evidence_labels).type(torch.LongTensor)
        evidence_labels.requires_grad = False

        # create mask
        entity_mask = num_entities * [1] + (self.max_entities - num_entities) * [0]
        entity_mask = torch.FloatTensor(entity_mask)
        evidence_mask = num_evidences * [1] + (self.max_evidences - num_evidences) * [0]
        evidence_mask = torch.FloatTensor(evidence_mask)

        # padding
        entities_list = entities_list + (self.max_entities - num_entities) * [{"id": "", "label": "", "type": ""}]
        evidences_list = evidences_list + (self.max_evidences - num_evidences) * [{"evidence_text": "", "contain_entities": []}]

        # check if there is an answer in training
        entity_labels_sum = torch.sum(entity_labels)
        if train and not entity_labels_sum:
            raise ContinueWithNext(f"Answer pruned via max_entities restriction")

        # normalize adjacency matrix
        vec = torch.sum(ent_to_ev.to_dense(), dim=0)
        vec[vec == 0] = 1
        ent_to_ev = ent_to_ev.to_dense() / vec

        vec = torch.sum(ev_to_ent.to_dense(), dim=0)
        vec[vec == 0] = 1
        ev_to_ent = ev_to_ent.to_dense() / vec

        instance = {
            "question_id": idx,
            "on_medicine": turn['people']['on_medicine'],
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
            "tsf": tsf,
            "question": question,
            "gold_answers": gold_answers,
        }
        return instance

def prepare_intermediate_data(data_dir, benchmark, output_dir, part='train'):
    """prepare intermediate data"""
    retriever = BM25Retriever()
    
    data_path = os.path.join(data_dir, benchmark, f"{part}.pkl")
    with open(data_path, "rb") as f:
        patients_data = pickle.load(f)
    
    processed_data = dict()
    for idx, patient in tqdm(patients_data.items(), desc=f"Processing {part} data"):
        topk_drugs_id = retriever.retrieve_drugs(patient)
        topk_drugs_msg = dict()
        for drug_id in topk_drugs_id:
            drug_msg = retriever.get_drugmsg_from_mkg(drug_id)
            topk_drugs_msg[drug_id] = drug_msg
        
        record = {
            'people': patient,
            'top_k_drugs': topk_drugs_msg
        }
        processed_data[idx] = record

    # Save processed data
    save_directory = os.path.join(output_dir, benchmark)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    save_path = os.path.join(save_directory, f"{part}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(processed_data, f)
    
    print(f"Saved {part} data to {save_path}")
    return save_path 