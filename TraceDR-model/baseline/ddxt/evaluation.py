import json
# from library.string_library import StringLibrary
from Levenshtein import distance as levenshtein_distance


def candidate_in_answers(answer_candidate, gold_answers):
    """Check if candidate is answer."""
    # get ids
    answer_candidate_id = answer_candidate["drugid"]
    gold_answer_ids = gold_answers

    # normalize
    # answer_candidate_id = answer_candidate_id.lower().strip().replace('"', "").replace("+", "")
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
    for answer in answers:
        if candidate_in_answers(answer["answer"], gold_answers):
            return 1.0 / float(answer["rank"])
    return 0.0


def precision_at_1(answers, gold_answers):
    """Compute P@1 score for given answers and gold answers."""
    # check if any answer was given
    if not answers:
        return 0.0
    # go through answer candidates
    for answer in answers:
        if float(answer["rank"]) > float(1.0):
            break
        elif candidate_in_answers(answer["answer"], gold_answers):
            return 1.0
    return 0.0


# def precision_at_1(answers, gold_answers):
#     """Compute P@1 score for given answers and gold answers."""
#     # 检查是否有推荐答案
#     if not answers:
#         return 0.0
#     # 检查排名第1的答案是否正确
#     top_answer = answers[0]["answer"]  # 获取排名第1的答案
#     return 1.0 if top_answer in gold_answers else 0.0


def hit_at_5(answers, gold_answers):
    """Compute Hit@5 score for given answers and gold answers."""
    # check if any answer was given
    if not answers:
        return 0.0
    # go through answer candidates
    for answer in answers:
        if float(answer["rank"]) > float(10.0):
            break
        elif candidate_in_answers(answer["answer"], gold_answers):
            return 1.0
    return 0.0


def calculate_p_at_k(answers, gold_answers):
    """
    计算precision@K
    answers:预测的药品id
    gold_answers:正确的药品id
    """
    if not answers:
        return 0.0
    if not gold_answers:
        return 1.0
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
    if not gold_answers:
        return 1.0
    recall_scores = []
    for answer in gold_answers:
        if answer in answers:
            recall_scores.append(1.0)
        else:
            recall_scores.append(0.0)
    return sum(recall_scores) / len(gold_answers)


def calculate_f1_at_k(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def calculate_metrics(answers, y_gt):
    # if not answers:
    #     return 0.0
    y_pred = list()
    # go through answer candidates
    for answer in answers:
        if float(answer["rank"]) > float(20.0):
            break
        else:
            y_pred.append(answer["answer"]["id"])

    # for answer in pred_id:
    #     pred_id.append((answer["answer"]["drugid"]))
    # 转换为集合以便计算Jaccard相似度
    set_pred = set(y_pred)
    set_gt = set(y_gt)

    # 计算Jaccard相似度
    jaccard_similarity = len(set_pred.intersection(set_gt)) / len(set_pred.union(set_gt))

    precision_at_k = calculate_p_at_k(y_pred, y_gt)
    recall_at_k = calculate_recall_at_k(y_pred, y_gt)
    f1_at_k = calculate_f1_at_k(precision_at_k, recall_at_k)

    return {
        'jaccard_similarity': jaccard_similarity,
        'average_precision': precision_at_k,
        'average_recall': recall_at_k,
        'average_f1': f1_at_k,
    }
	
	
def load_json(file_path):
    with open(file_path, 'rb') as f:
        return json.load(f)


def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
