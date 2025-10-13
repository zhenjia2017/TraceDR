import numpy as np
import json
from evaluation import *
import re
import csv
#from llms.utils import load_json, save_json
def load_json(file_path):
    with open(file_path, 'rb') as f:
        return json.load(f)


def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# 定义评估函数
def evaluate_metrics(answers, gold_answers):
    all_metrics = {
        "MRR": 0.0,
        "Precision@1": 0.0,
        "Hit@5": 0.0,
        "Jaccard Similarity": 0.0,
        "Precision@K": 0.0,
        "Recall@K": 0.0,
        "F1@K": 0.0,
    }
    num_samples = len(answers)

    for recommended, answer in zip(answers, gold_answers):
        # 计算 MRR、Precision@1 和 Hit@5
        answers_with_rank = [{"answer": {"drugid": rec}, "rank": idx + 1} for idx, rec in enumerate(recommended)]
        # print(answers_with_rank)
        # print(answer)
        mrr = mrr_score(answers_with_rank, answer)
        p1 = precision_at_1(answers_with_rank, answer)
        hit5 = hit_at_5(answers_with_rank, answer)

        # 计算 Jaccard 相似度、Precision@K、Recall@K 和 F1@K
        metrics = calculate_metrics(
            [{"answer": {"id": rec}, "rank": idx + 1} for idx, rec in enumerate(recommended)],
            answer,
        )

        # 累加指标
        all_metrics["MRR"] += mrr
        all_metrics["Precision@1"] += p1
        all_metrics["Hit@5"] += hit5
        all_metrics["Jaccard Similarity"] += metrics["jaccard_similarity"]
        all_metrics["Precision@K"] += metrics["average_precision"]
        all_metrics["Recall@K"] += metrics["average_recall"]
        all_metrics["F1@K"] += metrics["average_f1"]

    # 计算平均值
    for key in all_metrics:
        all_metrics[key] /= num_samples

    return all_metrics


def ddi1():
    with open("../DrugRec0716/test.pkl", "rb") as f1:
        patient_data = pickle.load(f1)
    with open("../results/merged_drug_data.pkl", "rb") as f2:
        drug_info = pickle.load(f2)
    # with open("../results/ddxt_result.json", "rb") as f3:
    with open("../results/ddxt_result.json", "rb") as f3:
        qwen_data = json.load(f3)
    All_DDI_list = []
    topk_drugs = drug_info
    i = 0
    for key, value in patient_data.items():
        # print(key)
        on_medicine = value['people']["on_medicine"]
        interaction_list = list()
        # DDI_list = [0, 0, 0, 0, 0]
        DDI_list = []  # 初始化为空列表，动态填充

        # 把所有正在用药的相互作用成分添加到列表中
        # 如果有正在用药才算DDI，否则该病人DDI为0
        if on_medicine:
            for medicine in on_medicine:
                interaction = medicine.get('interaction', [])
                if interaction:
                    for ingre in interaction:
                        interaction_list.append(ingre['name'])
            # 检查推荐的前k个药品的成分中，是否有与on_medicine的相互作用成分相同的药,说明它们会发生相互作用
            # if isinstance(qwen_data[i]['recommended'], str):
            #     answers = list(re.split(r'[、，]', qwen_data[i]['recommended']))
            # else:
            #     answers = qwen_data[i]['recommended']
            # answers = answers[:5]

            # Handle None or empty recommended field
            recommended = qwen_data[i].get('recommended')
            if recommended is None:
                recommended = ""

            # Convert to list whether it's string or already list
            if isinstance(recommended, str):
                answers = list(re.split(r'[、，,]', recommended))
            else:  # assume it's already a list
                answers = list(recommended)

            # Take first 5 items (or fewer if not available)
            answers = answers[:5] if answers else []
            for idx, answer in enumerate(answers):
                now = topk_drugs.get(answer, [])
                # print(now)
                if not now:
                    continue
                ingredients = now.get('成分', [])
                has_ddi = 0
                if ingredients:
                    for ingre in ingredients:
                        if ingre in interaction_list:
                            # DDI_list[idx] = 1
                            has_ddi = 1
                            break
                DDI_list.append(has_ddi)  # 动态添加DDI结果
        All_DDI_list.append(DDI_list)
        i += 1
    return All_DDI_list


def ddi2():
    with open("../DrugRec0716/test.pkl", "rb") as f1:
        patient_data = pickle.load(f1)
    with open("../results/merged_drug_data.pkl", "rb") as f2:
        drug_info = pickle.load(f2)
    # with open("../results/ddxt_result.json", "rb") as f3:
    with open("../results/ddxt_result.json", "rb") as f3:
        qwen_data = json.load(f3)
    All_DDI_list = []
    topk_drugs = drug_info
    i = 0
    for key, value in patient_data.items():
        # print(key)
        # on_medicine = value['people']["on_medicine"]
        allergen = value['people']["allergen"]
        population = value['people']["group"]
        interaction_list = list()
        DDI_list = []  # 初始化为空列表，动态填充
        # DDI_list = [0, 0, 0, 0, 0]
        # 为每个用户提供的人群词创建一个模糊匹配的模式
        population_patterns = [f"(?i).*{pop}.*" for pop in population]
        # 把所有正在用药的相互作用成分添加到列表中
        # 如果有正在用药才算DDI，否则该病人DDI为0

        # 检查推荐的前k个药品的成分中，是否有与on_medicine的相互作用成分相同的药,说明它们会发生相互作用
        # if isinstance(qwen_data[i]['recommended'], str):
        #     answers = list(re.split(r'[、，]', qwen_data[i]['recommended']))
        # else:
        #     answers = qwen_data[i]['recommended']
        # # answers = list(re.split(r'[、，]', qwen_data[i]['recommended']))
        # answers = answers[:5]

        # Handle None or empty recommended field
        recommended = qwen_data[i].get('recommended')
        if recommended is None:
            recommended = ""

        # Convert to list whether it's string or already list
        if isinstance(recommended, str):
            answers = list(re.split(r'[、，,]', recommended))
        else:  # assume it's already a list
            answers = list(recommended)

        # Take first 5 items (or fewer if not available)
        answers = answers[:5] if answers else []
        for idx, answer in enumerate(answers):
            now = topk_drugs.get(answer, [])
            # print(now)
            if not now:
                continue
            cautions = now.get('禁用', [])
            ingredients_list = now.get('成分', [])
            has_ddi = 0
            if cautions:
                populations = list()
                for caution in cautions:
                    populations.append(caution[0])
                drug_populations = set(populations)
                for pattern in population_patterns:
                    for pop in drug_populations:
                        if re.search(pattern, pop, re.IGNORECASE):  # 进行模糊匹配
                            # DDI_list[idx] = 1
                            has_ddi = 1
                            break
            if ingredients_list and allergen:
                drug_data = list()
                for ingredient in ingredients_list:
                    drug_data.append(ingredient)
                common_allergens = set(drug_data) & set(allergen)
                if common_allergens:
                    has_ddi = 1
                    # DDI_list[idx] = 1
            DDI_list.append(has_ddi)
        All_DDI_list.append(DDI_list)
        i += 1
    return All_DDI_list


def main():
    # llms评估
    # file_pairs = [
    #     # 'data/gml_4_air_result.json',
    #     # '../results/qwen_test_result.json',
    #     # '../results/glm_plus_test_result.json',
    #     # '../results/glm_air_test_result.json'
    #     '../results/mimic_glm4air_test_result.json'
    #     # 'data/test.json'
    # ]
    # for input_file in file_pairs:
    #     # 加载数据
    #     data = load_json(input_file)
    #     pred_data = []
    #     true_data = []
    #     for item in data:
    #         recommended_str = item["recommended"] or ""  # Convert None to empty string
    #         recommended = list(re.split(r'[、，,]', recommended_str))
    #         # recommended = list(re.split(r'[、，,]', item["recommended"]))
    #         answer = list(re.split(r'[、，,]', item["answer"]))  # 标准答案也转为列表以统一处理
    #         pred_data.append(recommended)
    #         true_data.append(answer)
    #     metrics = evaluate_metrics(pred_data, true_data)
    #     print(f'文件：{input_file}')
    #     print("Evaluation Metrics:")
    #     for metric, value in metrics.items():
    #         print(f"{metric}: {value:.4f}")

    data = load_json('../results/ddxt_result.json')
    pred_data = []
    true_data = []
    for item in data:
        recommended = item["recommended"]
        answer = item["answer"]
        if not recommended:
            print(item["id"])
        pred_data.append(recommended)
        true_data.append(answer)
    # print(len(pred_data))
    metrics = evaluate_metrics(pred_data, true_data)
    #print(f'文件：{'transformer_result.json'}')
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")


if __name__ == "__main__":
    main()
    # DDI
    a = ddi1()
    b = ddi2()
    total_CMS = 0
    total_DDI_rate_1 = 0
    total_DDI_rate = 0
    total_group_rate = 0

    for l1, l2 in zip(a, b):
        DDI_result = [bool(a) or bool(b) for a, b in zip(l1, l2)]
        DDI_rate = sum(l1) / len(l1) if len(l1) > 0 else 0
        group_rate = sum(l2) / len(l2) if len(l2) > 0 else 0
        CMS = sum(DDI_result) / len(DDI_result) if len(DDI_result) > 0 else 0
        # if DDI_rate != 0 and DDI_rate != 0.2:
        #     print(f"{count}: {DDI_rate:}")

        total_CMS += CMS
        total_DDI_rate_1 += DDI_result[0] if DDI_result != [] else 0
        total_DDI_rate += DDI_rate
        total_group_rate += group_rate

    average_result = {
        'CMS': total_CMS / len(a),
        'DDI_rate@1': total_DDI_rate_1 / len(a),
        'DDI_rate': total_DDI_rate / len(a),
        'group_rate': total_group_rate / len(b),
    }
    print(total_DDI_rate)
    for metric, value in average_result.items():
        print(f"{metric}: {value:.3f}")


    # print(average_result)
    # row_means = [sum(row) / len(row) for row in a]
    # test = sum(row_means) / len(row_means)
    #
    # print(test)

