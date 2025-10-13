from Levenshtein import distance as levenshtein_distance
import pickle
import re
with open("top50-interaction-109/test.pkl","rb") as f1:
    patient_data = pickle.load(f1)

def answer_presence(evidences, answers, relaxed=False):
    """
    Compute the answer presence for a set of evidences
    and a parsed answer dict, and return a list of
    answering evidences.
    Return format: (boolean, [evidence-dict, ...])
    """
    # initialize
    answer_present = False
    answering_evidences = list()

    # go through evidences
    for evidence in evidences:
        evidence['id'] = evidence['drugid']
        if candidate_in_answers(evidence, answers):
            # remember evidence
            answer_present = True
            answering_evidences.append(evidence)
    # return results
    return (answer_present, answering_evidences)


def evidence_has_answer(evidence, gold_answers, relaxed=False):
    """Check whether the given evidence has any of the answers."""
    for answer_candidate in evidence["contain_entities"]:
        if answer_candidate["type"] != "药品":
            continue
        # check for year in case the item is a timestamp
        # answer_candidate_id = answer_candidate["id"]
        # if relaxed and StringLibrary.is_timestamp(answer_candidate_id):
        #     year = StringLibrary.get_year(answer_candidate_id)
        #     if candidate_in_answers({"id": year, "label": year}, gold_answers):
        #         return True
        # check if answering candidate
        if candidate_in_answers(answer_candidate, gold_answers):
            return True

    return False


def candidate_in_answers(answer_candidate, gold_answers):
    """Check if candidate is answer."""
    # get ids
    answer_candidate_id = answer_candidate
    gold_answer_ids = gold_answers

    # normalize
    #answer_candidate_id = answer_candidate_id.lower().strip().replace('"', "").replace("+", "")
    # gold_answer_ids = [
    #     answer.lower().strip().replace('"', "").replace("+", "") for answer in gold_answer_ids
    # ]

    # perform check
    if answer_candidate_id in gold_answer_ids:
        return True

    # no match found
    return False


def mrr_score(answers, gold_answers):
    """Compute MRR score for given answers and gold answers."""
    # check if any answer was given
    if not answers:
        return 0.0
    # go through answer candidates
    for idx,answer in enumerate(answers):
        if candidate_in_answers(answer, gold_answers):
            return 1.0 / float(idx+1)
    return 0.0


def precision_at_1(answers, gold_answers):
    """Compute P@1 score for given answers and gold answers."""
    # check if any answer was given
    if not answers:
        return 0.0
    # go through answer candidates
    if candidate_in_answers(answers[0], gold_answers):
        return 1.0
    return 0.0


def hit_at_5(answers, gold_answers):
    """Compute Hit@5 score for given answers and gold answers."""
    # check if any answer was given
    if not answers:
        return 0.0
    # go through answer candidates
    for answer in answers:
        if candidate_in_answers(answer, gold_answers):
            return 1.0
    return 0.0

def calculate_p_at_k(answers, gold_answers):
    """
    计算precision@K
    answers:预测的药品id
    gold_answers:正确的药品id
    """
    p_list = list()
    for answer in answers:
        if answer in gold_answers:
            p_list.append(1.0)
        else:
            p_list.append(0.0)
    return sum(p_list) / len(p_list)

def calculate_recall_at_k(answers, gold_answers):
    """
    计算recall@K
    answers:预测的药品id
    gold_answers:正确的药品id
    """
    recall_scores = []
    for answer in gold_answers:
        if answer in answers:
            recall_scores.append(1.0)
        else:
            recall_scores.append(0.0)
    return sum(recall_scores) / len(gold_answers)

def calculate_f1_at_k(precision,recall):
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def population_allergen_check(answers,idx):
    """
        综合DDI，从人群和相互作用看
        DDI:推荐药品前k个中与正在用药有相互作用的个数/k
        answers: rank后的药品id
        idx: 病人id
        """
    topk_drugs = patient_data[idx]['top_k_drugs']
    population = patient_data[idx]['people'].group
    allergen = patient_data[idx]['people'].allergen
    interaction_list = list()
    DDI_list = [0,0,0,0,0]
    # 为每个用户提供的人群词创建一个模糊匹配的模式
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
                    if re.search(pattern, pop, re.IGNORECASE):  # 进行模糊匹配
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
    DDI:推荐药品前k个中与正在用药有相互作用的个数/k
    answers: rank后的药品id
    idx: 病人id
    """
    topk_drugs = patient_data[idx]['top_k_drugs']
    on_medicine = patient_data[idx]['people'].on_medicine
    interaction_list = list()
    DDI_list = [0,0,0,0,0]

    # 把所有正在用药的相互作用成分添加到列表中
    # 如果有正在用药才算DDI，否则该病人DDI为0
    if on_medicine:
        for idx,medicine in on_medicine.items():
            interaction = medicine.get('interaction',[])
            if interaction:
                for ingre in interaction:
                    interaction_list.append(ingre['name'])
        # 检查推荐的前k个药品的成分中，是否有与on_medicine的相互作用成分相同的药,说明它们会发生相互作用
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

def calculate_metrics(pred, y_gt,idx):
    # if not answers:
    #     return 0.0
    y_pred = list()
    # go through answer candidates
    answers = pred[:5]
    # for answer in pred_id:
    #     pred_id.append((answer["answer"]["drugid"]))
    # 转换为集合以便计算Jaccard相似度
    set_pred = set(answers)
    set_gt = set(y_gt)

    # 计算Jaccard相似度
    jaccard_similarity = len(set_pred.intersection(set_gt)) / len(set_pred.union(set_gt))
    DDI_list1 = calculate_DDI(answers ,idx)
    DDI_list2 = population_allergen_check(answers ,idx)
    precision_at_k = calculate_p_at_k(answers,y_gt)
    recall_at_k = calculate_recall_at_k(answers,y_gt)
    f1_at_k = calculate_f1_at_k(precision_at_k,recall_at_k)

    DDI_result = [bool(a) or bool(b) for a, b in zip(DDI_list1, DDI_list2)]
    DDI_rate = sum(DDI_result) / len(DDI_result)

    return {
        'DDI_rate': DDI_rate,
        'DDI_rate@1':DDI_result[0],
        'jaccard_similarity': jaccard_similarity,
        'average_precision': precision_at_k,
        'average_recall': recall_at_k,
        'average_f1': f1_at_k,
    }





def get_ranked_answers(config, generated_answer, turn):
    """
    Convert the predicted answer text to a Wikidata ID (or Yes/No),
    and return the ranked answers.
    Can be used for any method that predicts an answer string (instead of a KB item).
    """
    # check if existential (special treatment)
    question = turn["question"]
    if question_is_existential(question):
        ranked_answers = [
            {"answer": {"id": "yes", "label": "yes"}, "score": 1.0, "rank": 1},
            {"answer": {"id": "no", "label": "no"}, "score": 0.5, "rank": 2},
        ]
    # no existential
    else:
        # return dummy answer in case None was found (if no evidences found)
        if generated_answer is None:
            return [{"answer": {"id": "None", "label": "None"}, "rank": 1, "score": 0.0}]
        smallest_diff = 100000
        all_answers = list()
        mentions = set()
        for evidence in turn["top_evidences"]:
            for disambiguation in evidence["disambiguations"]:
                mention = disambiguation[0]
                id = disambiguation[1]
                if id is None or id == False:
                    continue

                # skip duplicates
                ans = str(mention) + str(id)
                if ans in mentions:
                    continue
                mentions.add(ans)
                # exact match
                if generated_answer == mention:
                    diff = 0
                # otherwise compute edit distance
                else:
                    diff = levenshtein_distance(generated_answer, mention)

                all_answers.append({"answer": {"id": id, "label": mention}, "score": diff})

        sorted_answers = sorted(all_answers, key=lambda j: j["score"])
        ranked_answers = [
            {"answer": answer["answer"], "score": answer["score"], "rank": i + 1}
            for i, answer in enumerate(sorted_answers)
        ]

    # don't return all answers
    max_answers = config["ha_max_answers"]
    ranked_answers = ranked_answers[:max_answers]
    if not ranked_answers:
        ranked_answers = [{"answer": {"id": "None", "label": "None"}, "rank": 1, "score": 0.0}]
    return ranked_answers


def question_is_existential(question):
    existential_keywords = [
        "is",
        "are",
        "was",
        "were",
        "am",
        "be",
        "being",
        "been",
        "did",
        "do",
        "does",
        "done",
        "doing",
        "has",
        "have",
        "had",
        "having",
    ]
    lowercase_question = question.lower()
    lowercase_question = lowercase_question.strip()
    for keyword in existential_keywords:
        if lowercase_question.split()[0] == keyword:
            return True
    return False
