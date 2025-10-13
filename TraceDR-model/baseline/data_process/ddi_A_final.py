import dill
import numpy as np
from scipy import sparse
import pickle
from collections import defaultdict

def build_top50_ddi_matrix(voc_path, ddi_adj_path, output_path = "dev"):
    """
    TOP50里面每个药的name+CMAN,对应一个编号，再对应到大ddi_adj
    :param voc_path:
    :param interactions_path:
    :param output_path:
    :return:
    """
    voc = dill.load(open(voc_path, 'rb'))
    med_voc = voc['med_voc']
    drug2idx = med_voc['word2idx']
    n_drugs = len(med_voc['idx2word'])
    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
    with open(f"DrugRec0716/{output_path}.pkl","rb") as f1:
        data = pickle.load(f1)
    for idx, item in data.items():
        topk_drugs = item["top_k_drugs"].values()
        ddi = build_interaction_matrix(topk_drugs, ddi_adj, drug2idx)
        item["ddi_adj"] = ddi
    with open(f"DrugRec0716/drugrec_withddi/{output_path}.pkl","wb") as f2:
        pickle.dump(data, f2)

def build_interaction_matrix(drug_list, large_ddi_matrix, drug2idx):
    """
    构建药品子集的相互作用矩阵

    参数:
    drug_list: 包含药品信息的列表，每个元素应包含'name'和'CMAN'字段
    large_interaction_df: 大相互作用矩阵，行和列索引应为药品标识

    返回:
    50x50的相互作用矩阵，以及药品顺序列表
    """
    # 提取药品的唯一标识 - 这里假设使用(name, CMAN)元组作为唯一键
    drug_ids = [f"{drug['name']}||{drug['CMAN']}" for drug in drug_list]

    # 初始化50x50的矩阵
    n = 50
    interaction_matrix = np.zeros((n, n))

    # 填充相互作用矩阵
    for i in range(len(drug_ids)):
        for j in range(len(drug_ids)):
            drug1 = drug_ids[i]
            drug2 = drug_ids[j]

            # 获取药品在大矩阵中的索引
            idx1 = drug2idx.get(drug1, -1)
            idx2 = drug2idx.get(drug2, -1)

            # 如果两个药品都在大矩阵中，获取相互作用值
            if idx1 != -1 and idx2 != -1:
                interaction = large_ddi_matrix[idx1, idx2]
            else:
                interaction = 0  # 默认无相互作用

            interaction_matrix[i, j] = interaction

    return interaction_matrix, drug_ids

def build_ddi_matrix2(voc_path, interactions_path, output_path):
    """不使用稀疏矩阵"""
    # 加载药物词汇表
    voc = dill.load(open(voc_path, 'rb'))
    med_voc = voc['med_voc']
    drug2idx = med_voc['word2idx']
    n_drugs = len(med_voc['idx2word'])

    # 初始化密集矩阵（全零）
    ddi_adj = np.zeros((n_drugs, n_drugs), dtype=np.int8)

    # 加载药物相互作用
    with open(interactions_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t{')
            if len(parts) < 2:
                continue
            drug_a = parts[0]
            drugs_b = parts[1].replace('{', '').replace('}', '').split('\t')
            for drug_b in drugs_b:
                if drug_b not in drug2idx or drug_a not in drug2idx:
                    continue  # 跳过无效药物
                i, j = drug2idx[drug_a], drug2idx[drug_b]
                if i != j:  # 避免自环
                    ddi_adj[i, j] = 1
                    ddi_adj[j, i] = 1  # 对称赋值

    # 保存为pkl
    with open(output_path, 'wb') as f:
        dill.dump(ddi_adj, f)

    print(f"密集DDI矩阵已保存至 {output_path}")
    print(f"矩阵维度: {ddi_adj.shape}, 非零元素数量: {np.sum(ddi_adj > 0)}")
    return ddi_adj


def build_ddi_matrix(voc_path, interactions_path, output_path):
    """使用稀疏矩阵"""
    # 加载药物词汇表
    voc = dill.load(open(voc_path, 'rb'))
    med_voc = voc['med_voc']
    drug2idx = med_voc['word2idx']
    n_drugs = len(med_voc['idx2word'])

    # 使用稀疏矩阵 (COO格式)
    rows, cols = [], []

    # 加载药物相互作用
    with open(interactions_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t{')
            if len(parts) < 2:
                continue
            drug_a = parts[0]
            drugs_b = parts[1].replace('{', '').replace('}', '').split('\t')
            for drug_b in drugs_b:
                if drug_b not in drug2idx or drug_a not in drug2idx:
                    continue  # 跳过无效药物
                i, j = drug2idx[drug_a], drug2idx[drug_b]
                rows.extend([i, j])  # 对称填充
                cols.extend([j, i])

    # 创建稀疏矩阵 (对称邻接矩阵)
    data = np.ones(len(rows), dtype=np.int8)
    ddi_adj = sparse.coo_matrix((data, (rows, cols)), shape=(n_drugs + 1, n_drugs + 1))
    ddi_adj = ddi_adj.tocsr()  # 转换为CSR格式节省内存

    # 保存为pkl
    with open(output_path, 'wb') as f:
        dill.dump(ddi_adj, f)

    print(f"稀疏DDI矩阵已保存至 {output_path}")
    print(f"矩阵维度: {ddi_adj.shape}, 非零元素数量: {ddi_adj.nnz}")


if __name__ == "__main__":
    # with open("drugrec_withddi/dev.pkl","rb") as f1:
    #     data = pickle.load(f1)
    # print("ok")
    # build_top50_ddi_matrix(
    #     voc_path = 'datasets/4S/voc_final.pkl',
    #     ddi_adj_path = 'datasets/4S/ddi_A_final.pkl',
    #     output_path="test")

    build_ddi_matrix(
        voc_path='DrugRec0716/voc_final.pkl',
        interactions_path='drug_interactions.txt',
        output_path='DrugRec0716/ddi_A_final.pkl'
    )


