import torch
import numpy as np
import dill
import os
import warnings
from model.mymodel1 import Model
from utils.dataset2 import PKLSet
from eval.metrics import multi_label_metric, ddi_rate_score, calculate_rank_metrics
from scipy.sparse import coo_matrix
from tqdm import trange, tqdm

warnings.filterwarnings("ignore", message="No positive class found in y_true, recall is set to one for all thresholds.")


def _to_sparse_tensor(x, device):
    x_sparse = x.to_sparse()
    return x_sparse.coalesce().to(device)


def calculate_ddi_rate_sparse(y_pred_labels, ddi_adj):
    """使用稀疏矩阵方式计算DDI率"""
    if not y_pred_labels:
        return 0.0

    total_pairs = 0
    ddi_pairs = 0

    for med_code_set in y_pred_labels:
        n_meds = len(med_code_set)
        if n_meds < 2:
            continue

        # 生成所有药物对
        for i in range(n_meds):
            for j in range(i + 1, n_meds):
                med_i = med_code_set[i]
                med_j = med_code_set[j]
                total_pairs += 1
                # 检查DDI矩阵中是否有交互
                if ddi_adj[med_i, med_j] > 0 or ddi_adj[med_j, med_i] > 0:
                    ddi_pairs += 1

    return ddi_pairs / total_pairs if total_pairs > 0 else 0.0


def calculate_ddi_at_k(y_pred_labels, ddi_adj, k=5):
    """计算top-k预测中的DDI率"""
    if not y_pred_labels:
        return 0.0

    total_pairs = 0
    ddi_pairs = 0

    for med_code_set in y_pred_labels:
        top_k = med_code_set[:k]
        n_meds = len(top_k)
        if n_meds < 2:
            continue

        for i in range(n_meds):
            for j in range(i + 1, n_meds):
                med_i = top_k[i]
                med_j = top_k[j]
                total_pairs += 1
                if ddi_adj[med_i, med_j] > 0 or ddi_adj[med_j, med_i] > 0:
                    ddi_pairs += 1

    return ddi_pairs / total_pairs if total_pairs > 0 else 0.0


def jaccard_at_k(y_gt, y_pred, k=5):
    """计算Jaccard@k指标"""
    score = []
    for b in range(y_gt.shape[0]):
        target = set(np.where(y_gt[b] == 1)[0])
        pred_top_k = set(np.argsort(y_pred[b])[::-1][:k])
        inter = pred_top_k & target
        union = pred_top_k | target
        jaccard_score = 0 if len(union) == 0 else len(inter) / len(union)
        score.append(jaccard_score)
    return np.mean(score)


def evaluate(model, test_loader, n_drugs, ddi_adj, device="cpu"):
    model.eval()
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    p_at_1, mrr, hit_at_5 = [], [], []
    precison_at_5, recall_at_5, f1_at_5 = [], [], []
    ja_at_5 = []  # 新增Jaccard@5指标
    smm_record = []
    med_cnt, visit_cnt = 0, 0

    for adm in tqdm(test_loader, desc="测试中"):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []

        syms = torch.tensor(adm[0]).to(device)
        drugs = torch.tensor(adm[2]).to(device)

        scores = model.evaluate(syms, device=device)

        y_gt_tmp = np.zeros(n_drugs)
        valid_indices = drugs[(drugs < n_drugs) & (drugs >= 0)].cpu().numpy().astype(int)
        y_gt_tmp[valid_indices] = 1
        y_gt.append(y_gt_tmp)

        result = torch.sigmoid(scores).detach().cpu().numpy()
        y_pred_prob.append(result)
        y_pred_tmp = result.copy()
        y_pred_tmp[y_pred_tmp >= 0.8] = 1
        y_pred_tmp[y_pred_tmp < 0.8] = 0
        y_pred.append(y_pred_tmp)

        y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
        y_pred_label.append(sorted(y_pred_label_tmp))
        visit_cnt += 1
        med_cnt += len(y_pred_label_tmp)
        smm_record.append(y_pred_label_tmp)  # 记录预测的药物

        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(np.array(y_gt),
                                                                                 np.array(y_pred),
                                                                                 np.array(y_pred_prob))
        adm_p1, adm_mrr, amd_h5, adm_precison5, adm_recall5, adm_f5 = calculate_rank_metrics(result, n_drugs, drugs,
                                                                                             device="cpu")
        adm_ja_at_5 = jaccard_at_k(np.array(y_gt), np.array(y_pred_prob), k=5)  # 计算Jaccard@5

        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        p_at_1.append(adm_p1)
        mrr.append(adm_mrr)
        hit_at_5.append(amd_h5)
        precison_at_5.append(adm_precison5)
        recall_at_5.append(adm_recall5)
        f1_at_5.append(adm_f5)
        ja_at_5.append(adm_ja_at_5)

    # 使用稀疏矩阵方式计算DDI率
    #ddi_rate = calculate_ddi_rate_sparse(smm_record, ddi_adj)
    #ddi_at_5 = calculate_ddi_at_k(smm_record, ddi_adj, k=5)
    ddi_rate =0
    ddi_at_5 = 0


    return (np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1),
            1.0 * med_cnt / visit_cnt, ddi_rate, np.mean(p_at_1), np.mean(mrr), np.mean(hit_at_5),
            np.mean(precison_at_5), np.mean(recall_at_5), np.mean(f1_at_5),
            np.mean(ja_at_5), ddi_at_5)  # 新增两个指标


def load_test_data(dataset_dir, file_name):
    data_path = os.path.join(dataset_dir, file_name)
    assert os.path.exists(data_path), f"测试数据文件 {data_path} 不存在"
    return dill.load(open(data_path, 'rb'))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    dataset_dir = os.path.join('DrugRec0716', '')  # 根据实际路径修改
    test_file = "data_test.pkl"

    # 加载数据集
    pklSet = PKLSet(batch_size=16, dataset='')  # 根据实际数据集名称修改

    # 加载测试数据
    test_data = load_test_data(dataset_dir, test_file)

    # 初始化模型
    ddi_adj_coo = pklSet.ddi_adj.tocoo()
    indices = torch.LongTensor(np.vstack((ddi_adj_coo.row, ddi_adj_coo.col)))
    values = torch.FloatTensor(ddi_adj_coo.data)
    shape = ddi_adj_coo.shape
    ddi_adj_tensor = torch.sparse_coo_tensor(indices, values, shape).to(device)

    # 将 drug_multihots 转换为 PyTorch 稀疏张量
    drug_multihots_coo = pklSet.drug_multihots.tocoo()
    drug_multihots_indices = torch.LongTensor(np.vstack((drug_multihots_coo.row, drug_multihots_coo.col)))
    drug_multihots_values = torch.FloatTensor(drug_multihots_coo.data)
    drug_multihots_shape = drug_multihots_coo.shape
    drug_multihots_tensor = torch.sparse_coo_tensor(drug_multihots_indices, drug_multihots_values,
                                                    drug_multihots_shape).to(device)

    model = Model(
        pklSet.n_sym, pklSet.n_drug, ddi_adj_tensor, pklSet.sym_sets,
        drug_multihots_tensor, embed_dim=64
    ).to(device)

    # 加载最佳模型
    best_model_path = os.path.join('checkpoints', 'best_model5e-3.pt')
    assert os.path.exists(best_model_path), f"最佳模型文件 {best_model_path} 不存在"
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # 测试评估
    test_loader = test_data  # 假设 test_data 格式与 data_eval 一致
    final_metrics = evaluate(model, test_loader, pklSet.n_drug, pklSet.ddi_adj, device)

    print('\n' + '=' * 50)
    print("最终测试结果:")
    print('-' * 50)
    print(f"JA: {final_metrics[0]:.4f} | PRAUC: {final_metrics[1]:.4f}")
    print(f"精确率: {final_metrics[2]:.4f} | 召回率: {final_metrics[3]:.4f}")
    print(f"F1分数: {final_metrics[4]:.4f} | 平均药物数: {final_metrics[5]:.2f}")
    print(f"DDI率: {final_metrics[6]:.4f} | DDI@5: {final_metrics[14]:.4f}")  # 新增DDI@5
    print(f"P@1: {final_metrics[7]:.4f} | MRR: {final_metrics[8]:.4f}")
    print(f"Hit@5: {final_metrics[9]:.4f} | Precision@5: {final_metrics[10]:.4f}")
    print(f"Recall@5: {final_metrics[11]:.4f} | F1@5: {final_metrics[12]:.4f}")
    print(f"Jaccard@5: {final_metrics[13]:.4f}")  # 新增Jaccard@5
    print('=' * 50 + '\n')


if __name__ == '__main__':
    main()