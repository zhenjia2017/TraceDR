from Levenshtein import distance as levenshtein_distance
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
    # 检查answer_candidate中是否有任一元素出现在gold_answers中
    for candidate in answer_candidate:
        answer_candidate_id = candidate
        if answer_candidate_id in gold_answers:
            return True
    return False

def find_first_correct_index(sorted_texts, mappings):
    for index, text in enumerate(sorted_texts):
        if mappings.get(text[1], 0) == 1:  # 如果文本在字典中并且是True
            return index  # 返回索引
    #return 7

def mrr_score(answers, mappings):
    """Compute MRR score for given answers and gold answers."""
    # check if any answer was given
    index = find_first_correct_index(answers,mappings)
    return 1.0 / float(index+1)


def precision_at_1(answers, mappings):
    """Compute P@1 score for given answers and gold answers."""
    # check if any answer was given
    if mappings[answers[0][1]] == 1:#['query','drug']
        return 1.0
    else:
        return 0.0

def calculate_p_at_k(answers, mappings):
    """
    计算precision@K
    answers:预测的药品id
    gold_answers:正确的药品id
    """
    p_list = list()
    for answer in answers:
        if mappings[answer] == 1:
            p_list.append(1.0)
        else:
            p_list.append(0.0)
    return sum(p_list) / len(p_list)

def calculate_recall_at_k(answers, mappings,positive_evidences):
    """
    计算recall@K
    answers:预测的药品id
    gold_answers:正确的药品id
    """
    recall_scores = []
    for answer in answers:
        if mappings[answer] == 1:
            recall_scores.append(1.0)
        else:
            recall_scores.append(0.0)
    return sum(recall_scores) / float(len(positive_evidences))

def calculate_f1_at_k(precision,recall):
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def calculate_metrics(answers, positive_evidences,mapping):
    # if not answers:
    #     return 0.0
    y_pred = list()
    y_gt = list()
    # go through answer candidates
    num = 0
    for answer in answers:
        if num > 5:
            break
        num += 1
        y_pred.append(answer[1])
        # else:
        #     y_pred.append(answer["answer"]["id"])
    for item in positive_evidences:
        y_gt.append(item[2])
    # for answer in pred_id:
    #     pred_id.append((answer["answer"]["drugid"]))
    # 转换为集合以便计算Jaccard相似度
    set_pred = set(y_pred)
    set_gt = set(y_gt)

    # 计算Jaccard相似度
    jaccard_similarity = len(set_pred.intersection(set_gt)) / len(set_pred.union(set_gt))

    precision_at_k = calculate_p_at_k(y_pred,mapping)
    recall_at_k = calculate_recall_at_k(y_pred,mapping, positive_evidences)
    f1_at_k = calculate_f1_at_k(precision_at_k,recall_at_k)

    return {
        'jaccard_similarity': jaccard_similarity,
        'average_precision': precision_at_k,
        'average_recall': recall_at_k,
        'average_f1': f1_at_k,
    }


def hit_at_5(answers, mappings):
    """Compute Hit@5 score for given answers and gold answers."""
    # check if any answer was given
    if not answers:
        return 0.0
    # go through answer candidates
    for idx,answer in enumerate(answers):
        if float(idx) > float(5.0):
            break
        elif mappings[answer[1]] == 1:
            return 1.0
    return 0.0


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
