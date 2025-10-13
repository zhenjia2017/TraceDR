import re
import pickle
from Levenshtein import distance as levenshtein_distance

# Load patient data for DDI calculations
with open("../../intermediate_data/DrugRec0716/test.pkl", "rb") as f1:
    patient_data = pickle.load(f1)


def candidate_in_answers(answer_candidate, gold_answers):
    """Check if candidate is answer."""
    if answer_candidate in gold_answers:
        return True
    return False


def mrr_score(answers, gold_answers):
    """Compute MRR score for given answers and gold answers."""
    if not answers:
        return 0.0
    
    for answer in answers:
        if candidate_in_answers(answer["drug_id"], gold_answers):
            return 1.0 / float(answer["rank"])
    return 0.0


def precision_at_1(answers, gold_answers):
    """Compute P@1 score for given answers and gold answers."""
    if not answers:
        return 0.0
    
    for answer in answers:
        if float(answer["rank"]) > float(1.0):
            break
        elif candidate_in_answers(answer["drug_id"], gold_answers):
            return 1.0
    return 0.0


def hit_at_5(answers, gold_answers):
    """Compute Hit@5 score for given answers and gold answers."""
    if not answers:
        return 0.0
    
    for answer in answers:
        if float(answer["rank"]) > float(5.0):
            break
        elif candidate_in_answers(answer["drug_id"], gold_answers):
            return 1.0
    return 0.0


def calculate_p_at_k(answers, gold_answers):
    """Calculate precision@K"""
    p_list = list()
    if not gold_answers:
        return 1.0
    
    for answer in answers:
        if answer in gold_answers:
            p_list.append(1.0)
        else:
            p_list.append(0.0)
    return sum(p_list) / len(p_list)


def calculate_recall_at_k(answers, gold_answers):
    """Calculate recall@K"""
    recall_scores = []
    if not gold_answers:
        return 1.0
    
    for answer in gold_answers:
        if answer in answers:
            recall_scores.append(1.0)
        else:
            recall_scores.append(0.0)
    return sum(recall_scores) / len(gold_answers)


def calculate_f1_at_k(precision, recall):
    """Calculate F1@K"""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def calculate_metrics(answers, y_gt, idx):
    """Calculate comprehensive metrics including DDI and safety measures."""
    y_pred = list()
    
    # Extract top 5 predictions
    for answer in answers:
        if float(answer["rank"]) > float(5.0):
            break
        else:
            y_pred.append(answer["drug_id"])

    # Convert to sets for Jaccard similarity
    set_pred = set(y_pred)
    set_gt = set(y_gt)

    # Calculate Jaccard similarity
    jaccard_similarity = len(set_pred.intersection(set_gt)) / len(set_pred.union(set_gt)) if set_pred.union(set_gt) else 0.0
    
    # Calculate DDI metrics
    DDI_list1 = calculate_DDI(y_pred, idx)
    DDI_list2 = population_allergen_check(y_pred, idx)
    
    # Calculate precision, recall, F1
    precision_at_k = calculate_p_at_k(y_pred, y_gt)
    recall_at_k = calculate_recall_at_k(y_pred, y_gt)
    f1_at_k = calculate_f1_at_k(precision_at_k, recall_at_k)

    # Combine DDI results
    DDI_result = [bool(a) or bool(b) for a, b in zip(DDI_list1, DDI_list2)]
    
    # Calculate rates
    DDI_rate = sum(DDI_list1) / len(DDI_list1) if DDI_list1 else 0.0
    group_rate = sum(DDI_list2) / len(DDI_list2) if DDI_list2 else 0.0
    CMS = sum(DDI_result) / len(DDI_result) if DDI_result else 0.0
    
    return {
        'CMS': CMS,
        'group_rate': group_rate,
        'DDI_rate': DDI_rate,
        'DDI_rate@1': DDI_result[0] if DDI_result else 0.0,
        'jaccard': jaccard_similarity,
        'avg_p': precision_at_k,
        'avg_re': recall_at_k,
        'avg_f1': f1_at_k,
    }


def population_allergen_check(answers, idx):
    """Check for population and allergen contraindications."""
    if idx not in patient_data:
        return [0, 0, 0, 0, 0]
    
    topk_drugs = patient_data[idx]['top_k_drugs']
    population = patient_data[idx]['people']["group"]
    allergen = patient_data[idx]['people']["allergen"]
    
    DDI_list = [0, 0, 0, 0, 0]
    
    # Create fuzzy matching patterns for population
    population_patterns = [f"(?i).*{pop}.*" for pop in population]

    for idx_drug, answer in enumerate(answers):
        if idx_drug >= 5:  # Only check top 5
            break
            
        now = topk_drugs.get(answer, [])
        if not now:
            continue
            
        cautions = now.get('caution', [])
        ingredients_list = topk_drugs[answer].get('ingredients', [])
        
        # Check contraindicated populations
        if cautions:
            populations = list()
            for caution in cautions:
                populations.append(caution['crowd'])
            drug_populations = set(populations)
            
            for pattern in population_patterns:
                for pop in drug_populations:
                    if re.search(pattern, pop, re.IGNORECASE):
                        DDI_list[idx_drug] = 1
                        break
        
        # Check allergens
        if ingredients_list and allergen:
            drug_data = list()
            for ingredient in ingredients_list:
                drug_data.append(ingredient['ingredient'])
            common_allergens = set(drug_data) & set(allergen)
            if common_allergens:
                DDI_list[idx_drug] = 1
                
    return DDI_list


def calculate_DDI(answers, idx):
    """Calculate Drug-Drug Interactions."""
    if idx not in patient_data:
        return [0, 0, 0, 0, 0]
    
    topk_drugs = patient_data[idx]['top_k_drugs']
    on_medicine = patient_data[idx]['people']["on_medicine"]
    
    interaction_list = list()
    DDI_list = [0, 0, 0, 0, 0]

    # Collect all interaction components from current medications
    if on_medicine:
        for medicine in on_medicine:
            interaction = medicine.get('interaction', [])
            if interaction:
                for ingre in interaction:
                    interaction_list.append(ingre['name'])
        
        # Check if recommended drugs interact with current medications
        for idx_drug, answer in enumerate(answers):
            if idx_drug >= 5:  # Only check top 5
                break
                
            now = topk_drugs.get(answer, [])
            if not now:
                continue
                
            ingredients = now['ingredients']
            for ingre in ingredients:
                if ingre['ingredient'] in interaction_list:
                    DDI_list[idx_drug] = 1
                    break
                    
    return DDI_list



