import numpy as np
import torch
import warnings
from sklearn.metrics import jaccard_score, roc_auc_score, precision_score, f1_score, average_precision_score
from sklearn.exceptions import UndefinedMetricWarning
import dill

# �����ض�����
warnings.filterwarnings("ignore", 
                        message="No positive class found in y_true, recall is set to one for all thresholds.")
def multi_label_metric(y_gt, y_pred, y_prob):

    def jaccard(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            union = set(out_list) | set(target)
            jaccard_score = 0 if len(union) == 0 else len(inter) / len(union)
            score.append(jaccard_score)
        return np.mean(score)

    def average_prc(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            prc_score = 0 if len(out_list) == 0 else len(inter) / len(out_list)
            score.append(prc_score)
        return score

    def average_recall(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            recall_score = 0 if len(target) == 0 else len(inter) / len(target)
            score.append(recall_score)
        return score

    def average_f1(average_prc, average_recall):
        score = []
        for idx in range(len(average_prc)):
            if average_prc[idx] + average_recall[idx] == 0:
                score.append(0)
            else:
                score.append(2*average_prc[idx]*average_recall[idx] / (average_prc[idx] + average_recall[idx]))
        return score

    def f1(y_gt, y_pred):
        all_micro = []
        for b in range(y_gt.shape[0]):
            all_micro.append(f1_score(y_gt[b], y_pred[b], average='macro'))
        return np.mean(all_micro)

    def roc_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(roc_auc_score(y_gt[b], y_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(average_precision_score(y_gt[b], y_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_at_k(y_gt, y_prob, k=3):
        precision = 0
        sort_index = np.argsort(y_prob, axis=-1)[:, ::-1][:, :k]
        for i in range(len(y_gt)):
            TP = 0
            for j in range(len(sort_index[i])):
                if y_gt[i, sort_index[i, j]] == 1:
                    TP += 1
            precision += TP / len(sort_index[i])
        return precision / len(y_gt)

    # roc_auc
    #print(y_gt.shape) #(1,450)
    #print(y_prob.shape) #(1,450)
    try:
        auc = roc_auc(y_gt, y_prob)
    except:
        auc = 0
    # precision
    p_1 = precision_at_k(y_gt, y_prob, k=1)
    p_3 = precision_at_k(y_gt, y_prob, k=3)
    p_5 = precision_at_k(y_gt, y_prob, k=5)
    # macro f1
    f1 = f1(y_gt, y_pred)
    # precision
    prauc = precision_auc(y_gt, y_prob)
    # jaccard
    ja = jaccard(y_gt, y_pred)
    # pre, recall, f1
    avg_prc = average_prc(y_gt, y_pred)
    avg_recall = average_recall(y_gt, y_pred)
    avg_f1 = average_f1(avg_prc, avg_recall)

    return ja, prauc, np.mean(avg_prc), np.mean(avg_recall), np.mean(avg_f1)

def ddi_rate_score(record, ddi_A):
    all_cnt = 0
    dd_cnt = 0
    for patient in record:
        for adm in patient:
            med_code_set = adm  # 直接使用1-based索引
            for i, med_i in enumerate(med_code_set):
                for j, med_j in enumerate(med_code_set):
                    if j <= i:
                        continue
                    all_cnt += 1
                    # ddi_A 使用0-based索引，因此需要减1
                    if ddi_A[med_i - 1, med_j - 1] == 1 or ddi_A[med_j - 1, med_i - 1] == 1:
                        dd_cnt += 1

    return dd_cnt / all_cnt if all_cnt != 0 else 0

def calculate_rank_metrics(scores, n_drugs, true_drugs, device="cpu"):
    """
    Calculate ranking metrics for drug recommendation.
    
    Args:
        scores: numpy array of predicted scores for each drug (shape: [n_drugs])
        n_drugs: total number of drugs
        true_drugs: tensor of true drug indices
        device: device to use for computations
        
    Returns:
        p_at_1: precision at 1
        mrr: mean reciprocal rank
        hit_at_5: hit rate at 5
        precision_at_5: precision at 5
        recall_at_5: recall at 5
        f1_at_5: f1 score at 5
    """
    # Filter out invalid drug indices and convert to numpy
    true_drugs = true_drugs.cpu().numpy().astype(int)
    true_set = set(true_drugs)
    
    if len(true_set) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    # Get ranked list of drugs (descending order)
    ranked_drugs = np.argsort(scores)[::-1]
    
    # Calculate P@1
    p_at_1 = 1.0 if ranked_drugs[0] in true_set else 0.0
    
    # Calculate MRR
    mrr = 0.0
    for rank, drug in enumerate(ranked_drugs, 1):
        if drug in true_set:
            mrr = 1.0 / rank
            break
    
    # Calculate HIT@5
    hit_at_5 = 1.0 if len(set(ranked_drugs[:5]) & true_set) > 0 else 0.0
    
    # Calculate precision, recall, F1@5
    top5_drugs = ranked_drugs[:5]
    true_positives = len(set(top5_drugs) & true_set)
    
    precision_at_5 = true_positives / len(top5_drugs) if len(top5_drugs) > 0 else 0.0
    recall_at_5 = true_positives / len(true_set) if len(true_set) > 0 else 0.0
    
    if (precision_at_5 + recall_at_5) > 0:
        f1_at_5 = 2 * (precision_at_5 * recall_at_5) / (precision_at_5 + recall_at_5)
    else:
        f1_at_5 = 0.0
    
    return p_at_1, mrr, hit_at_5, precision_at_5, recall_at_5, f1_at_5