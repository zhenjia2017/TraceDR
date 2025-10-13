import torch
import re
import pickle
with open("intermediate_data/DrugRec0716/dev.pkl","rb") as f1:
    patient_data = pickle.load(f1)

def candidate_in_answers(answer_candidate, gold_answers):
    """Check if candidate is answer."""
    answer_candidate_id = answer_candidate["id"]
    return answer_candidate_id in gold_answers


def precision_at_1(answers, gold_answers):
    """Compute P@1 score for given answers and gold answers."""
    if not answers:
        return 0.0
    
    for answer in answers:
        if float(answer["rank"]) > 1.0:
            break
        elif candidate_in_answers(answer["answer"], gold_answers):
            return 1.0
    return 0.0


def mrr_score(answers, gold_answers):
    """Compute MRR score for given answers and gold answers."""
    if not answers:
        return 0.0
    
    for answer in answers:
        if candidate_in_answers(answer["answer"], gold_answers):
            return 1.0 / float(answer["rank"])
    return 0.0


def hit_at_k(answers, gold_answers, k=5):
    """Compute Hit@K score for given answers and gold answers."""
    if not answers:
        return 0.0
    
    for answer in answers:
        if float(answer["rank"]) > float(k):
            break
        elif candidate_in_answers(answer["answer"], gold_answers):
            return 1.0
    return 0.0


def calculate_p_at_k(predicted_ids, gold_answers, k=5):
    """
    calculate precision@K
    predicted_ids: the predicted drug ids list
    gold_answers: the correct drug ids list
    """
    if not predicted_ids:
        return 0.0
    
    # only consider the first k predictions
    pred_k = predicted_ids[:k]
    correct = sum(1 for pred_id in pred_k if pred_id in gold_answers)
    return correct / len(pred_k)


def calculate_recall_at_k(predicted_ids, gold_answers, k=5):
    """
    calculate recall@K
    predicted_ids: the predicted drug ids list
    gold_answers: the correct drug ids list
    """
    if not gold_answers:
        return 1.0
    
    # only consider the first k predictions
    pred_k = predicted_ids[:k]
    correct = sum(1 for gold_id in gold_answers if gold_id in pred_k)
    return correct / len(gold_answers)


def calculate_f1_at_k(precision, recall):
    """calculate F1@K score"""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def jaccard_similarity(predicted_ids, gold_answers, k=5):
    """calculate Jaccard similarity"""
    pred_set = set(predicted_ids[:k])
    gold_set = set(gold_answers)
    
    if not pred_set and not gold_set:
        return 1.0
    
    intersection = len(pred_set & gold_set)
    union = len(pred_set | gold_set)
    
    return intersection / union if union > 0 else 0.0


def answer_presence(answers, gold_answers):
    """Check if any of the gold answers is present in the predictions."""
    if not answers:
        return 0.0
    
    for answer in answers:
        if candidate_in_answers(answer["answer"], gold_answers):
            return 1.0
    return 0.0

def population_allergen_check(answers,idx):
    """
        DDI: the number of drugs that have interactions with the currently used drugs in the top k recommendations / k
        answers: the drug ids in the top k recommendations
        idx: the index of the patient
        """
    topk_drugs = patient_data[idx]['top_k_drugs']
    population = patient_data[idx]['people']["group"]
    allergen = patient_data[idx]['people']["allergen"]
    interaction_list = list()
    DDI_list = [0,0,0,0,0]
    # create a fuzzy matching pattern for each patient's population
    population_patterns = [f"(?i).*{pop}.*" for pop in population]

    for idx,answer in enumerate(answers):
        now = topk_drugs.get(answer,[])
        if not now:
            continue
        cautions = now.get('caution',[])
        ingredients_list = topk_drugs[answer].get('ingredients',[])
        if cautions:
            populations = list()
            for caution in cautions:
                populations.append(caution['crowd'])
            drug_populations = set(populations)
            for pattern in population_patterns:
                for pop in drug_populations:
                    if re.search(pattern, pop, re.IGNORECASE):  # fuzzy matching
                        DDI_list[idx] = 1
                        break
        if ingredients_list and allergen:
            drug_data = list()
            for ingredient in ingredients_list:
                drug_data.append(ingredient['ingredient'])
            common_allergens = set(drug_data) & set(allergen)
            if common_allergens:
                DDI_list[idx] = 1
    return DDI_list


def calculate_DDI(answers,idx):
    """
    DDI: the number of drugs that have interactions with the currently used drugs in the top k recommendations / k
    answers: the drug ids in the top k recommendations
    idx: the index of the patient
    """
    topk_drugs = patient_data[idx]['top_k_drugs']
    on_medicine = patient_data[idx]['people']["on_medicine"]
    interaction_list = list()
    DDI_list = [0,0,0,0,0]

    # add all the interactions of the currently used drugs to the list
    # if there is currently used drugs, then calculate DDI, otherwise the DDI is 0
    if on_medicine:
        for medicine in on_medicine:
            interaction = medicine.get('interaction',[])
            if interaction:
                for ingre in interaction:
                    interaction_list.append(ingre['name'])
        # check if the ingredients of the top k recommendations have interactions with the currently used drugs
        for idx,answer in enumerate(answers):
            now = topk_drugs.get(answer,[])
            if not now:
                continue
            ingredients = now['ingredients']
            for ingre in ingredients:
                if ingre['ingredient'] in interaction_list:
                    DDI_list[idx] = 1
                    break
    return DDI_list

def calculate_metrics(idx, answers, gold_answers, k=5):
    """
    calculate the complete evaluation metrics, refer to the calculate_metrics function in evaluation.py
    
    Args:
        answers: Sorted list of answers, each element containing {"answer": {"id": ..., "label": ...}, "rank": ..., "score": ...}
        gold_answers: correct answer IDs list.
        k: The value of k when calculating top-k indicators
    
    Returns:
        dict: dict containing various evaluation indicators
    """
    if not answers:
        predicted_ids = []
    else:
        # extract the first k predicted ids
        predicted_ids = []
        for answer in answers:
            if float(answer["rank"]) > float(k):
                break
            predicted_ids.append(answer["answer"]["id"])
    
    # basic metrics
    p_at_1 = precision_at_1(answers, gold_answers)
    mrr = mrr_score(answers, gold_answers)
    h_at_5 = hit_at_k(answers, gold_answers, k=5)
    ans_presence = answer_presence(answers, gold_answers)
    
    # Precision, Recall, F1 @ K
    precision_k = calculate_p_at_k(predicted_ids, gold_answers, k)
    recall_k = calculate_recall_at_k(predicted_ids, gold_answers, k)
    f1_k = calculate_f1_at_k(precision_k, recall_k)
    
    # Jaccard similarity
    jaccard = jaccard_similarity(predicted_ids, gold_answers, k)

    #DDI 
    DDI_list1 = calculate_DDI(predicted_ids ,idx)
    DDI_list2 = population_allergen_check(predicted_ids ,idx)

    DDI_result = [bool(a) or bool(b) for a, b in zip(DDI_list1, DDI_list2)]
    DDI_rate = sum(DDI_list1) / len(DDI_list1)
    group_rate = sum(DDI_list2) / len(DDI_list2)
    CMS = sum(DDI_result) / len(DDI_result)
    
    return {
        "p_at_1": p_at_1,
        "mrr": mrr,
        "h_at_5": h_at_5,
        "answer_presence": ans_presence,
        "jaccard": jaccard,
        "precision_at_k": precision_k,
        "recall_at_k": recall_k,
        "f1_at_k": f1_k,
        # 'CMS': CMS,
        # 'DDI_rate@1':DDI_result[0],
        # 'DDI_rate': DDI_rate,
        # 'group_rate': group_rate
    }


def evaluate_predictions(batch, answer_predictions, evidence_predictions=None):
    """
    evaluate the model prediction results
    """
    qa_metrics = []
    
    for b in range(len(batch["gold_answers"])):
        gold_answers = batch["gold_answers"][b]
        predictions = answer_predictions[b] if answer_predictions else []
        
        # calculate the metrics
        metrics = calculate_metrics(batch["question_id"][b], predictions, gold_answers, k=5)
        qa_metrics.append(metrics)
    
    return qa_metrics


def aggregate_metrics(metrics_list):
    """
    aggregate the evaluation metrics of multiple samples
    """
    if not metrics_list:
        return {}
    
    # get all the possible metric names
    all_keys = set()
    for metrics in metrics_list:
        all_keys.update(metrics.keys())
    
    # calculate the average of each metric
    aggregated = {}
    for key in all_keys:
        values = [metrics.get(key, 0.0) for metrics in metrics_list if key in metrics]
        if values:
            aggregated[key] = sum(values) / len(values)
        else:
            aggregated[key] = 0.0
    
    # add the number of samples
    aggregated["num_questions"] = len(metrics_list)
    
    return aggregated 