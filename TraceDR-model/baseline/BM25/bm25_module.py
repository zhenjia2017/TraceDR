import pickle
import jieba
from tqdm import tqdm
import evaluation
from ExplaiDR.library.utils import get_logger,get_config
from bm25_es import BM25Scoring

class BM25Module:
    def __init__(self, config):
        self.config = config
        self.faith = self.config["faith_or_unfaith"]
        self.benchmark = self.config["benchmark"]

    def chinese_tokenizer(self,text):
        string = text.replace(",", " ")
        return jieba.lcut(string)

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

    def predata(self):
        """Preprocess data to generate mappings, corpus, and query dictionaries."""
        with open("top50-interaction-109/test.pkl", "rb") as fp:
            patient_data = pickle.load(fp)

        mapping_all, corpus_all, query_dict, people_data = {}, {}, {}, {}

        for idx, data in patient_data.items():
            topk_drug = data['top_k_drugs']
            corpus, mapping = [], {}
            for drugid, drug in topk_drug.items():
                drug_text = self.get_evidence_text(drug)
                if drug_text not in mapping:
                    corpus.append(drug_text)
                    mapping[drug_text] = [drug["drugid"]]
                else:
                    mapping[drug_text].append(drug["drugid"])

            corpus_all[idx], mapping_all[idx] = corpus, mapping
            people_data[idx], person = patient_data[idx]["people"], patient_data[idx]["people"]
            query_dict[idx] = self.person_to_query(person, '||')

        return mapping_all, corpus_all, query_dict, people_data

    def bm25_select_topk_drugs(self):
        """Use the BM25 to select the most relevant drugs."""
        mapping, corpus, query_dict, patient_data = self.predata()

        # Initialize evaluation lists
        metrics_lists = {
            'p_at_1_list': [],
            'mrr_list': [],
            'hit5_list': [],
            'jaccard_list': [],
            'precison_list': [],
            'recall_list': [],
            'f1_list': [],
            'ddi_list': [],
            'ddi_1_list': []
        }

        for idx, query in tqdm(query_dict.items()):
            gold_answers = list(patient_data[idx].medication.keys())
            bm25_select = BM25Scoring(corpus[idx], mapping[idx], is_retrived=False)
            topk_drug = bm25_select.get_top_evidences(query)
            select_drug = [drug for drugs in topk_drug for drug in drugs]

            # Calculate and collect various metrics
            metrics = evaluation.calculate_metrics(select_drug, gold_answers, idx)
            metrics_lists['p_at_1_list'].append(evaluation.precision_at_1(select_drug, gold_answers))
            metrics_lists['mrr_list'].append(evaluation.mrr_score(select_drug, gold_answers))
            metrics_lists['hit5_list'].append(evaluation.hit_at_5(select_drug, gold_answers))
            metrics_lists['jaccard_list'].append(metrics['jaccard_similarity'])
            metrics_lists['precison_list'].append(metrics['average_precision'])
            metrics_lists['recall_list'].append(metrics['average_recall'])
            metrics_lists['f1_list'].append(metrics['average_f1'])
            metrics_lists['ddi_list'].append(metrics['DDI_rate'])
            metrics_lists['ddi_1_list'].append(metrics['DDI_rate@1'])

        # Output averages
        for key, values in metrics_lists.items():
            print(f"{key.split('_')[0]}: {sum(values) / len(values)}")

if __name__ == "__main__":
    config = get_config('trainGCN_50_05_05.yml')
    bm25 = BM25Module(config)
    bm25.bm25_select_topk_drugs()