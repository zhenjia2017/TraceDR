import dill
import pickle
import os
import numpy as np
from tqdm import tqdm
from scipy.sparse import lil_matrix, csr_matrix
train_path = 'DrugRec0716/data_train.pkl'
dev_path = 'DrugRec0716/data_eval.pkl'
test_path = 'DrugRec0716/data_test.pkl'
voc_path = 'DrugRec0716/voc_final.pkl'

def construct_kgdnet_adj():
    # 加载数据
    train = dill.load(open(train_path, 'rb'))
    dev = dill.load(open(dev_path, 'rb'))
    test = dill.load(open(test_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    all_data = train + dev + test

    diag_voc, sym_voc, med_voc = voc['diag_voc'], voc['sym_voc'], voc['med_voc']
    diag_idx_arr_len = len(diag_voc["idx2word"]) + 1
    sym_idx_arr_len = len(sym_voc["idx2word"]) + 1
    med_idx_arr_len = len(med_voc["idx2word"]) + 1
    # 确保路径存在
    path = 'KGDnetdata/AdjMatrices'
    os.makedirs(path, exist_ok=True)
    
    # 初始化稀疏邻接矩阵 (LIL 格式，便于逐个添加)
    diag_adj = lil_matrix((diag_idx_arr_len, diag_idx_arr_len), dtype=np.uint8)
    sym_adj = lil_matrix((sym_idx_arr_len, sym_idx_arr_len), dtype=np.uint8)
    diag_sym_adj = lil_matrix((diag_idx_arr_len, sym_idx_arr_len), dtype=np.uint8)
    sym_diag_adj = lil_matrix((sym_idx_arr_len, diag_idx_arr_len), dtype=np.uint8)
    prescriptions_adj = lil_matrix((med_idx_arr_len, med_idx_arr_len), dtype=np.uint8)

    for adm in tqdm(all_data):
        diag_set = adm[1]
        sym_set = adm[0]
        med_set = adm[2]

        if diag_set:
            for i, diag_i in enumerate(diag_set):
                for j, diag_j in enumerate(diag_set):
                    if j <= i:
                        continue
                    diag_adj[diag_i, diag_j] = 1
                    diag_adj[diag_j, diag_i] = 1

        if sym_set:
            for i, sym_i in enumerate(sym_set):
                for j, sym_j in enumerate(sym_set):
                    if j <= i:
                        continue
                    sym_adj[sym_i, sym_j] = 1
                    sym_adj[sym_j, sym_i] = 1

        if diag_set and sym_set:
            for diag in diag_set:
                for sym in sym_set:
                    diag_sym_adj[diag, sym] = 1
                    sym_diag_adj[sym, diag] = 1

        if med_set:
            for i, med_i in enumerate(med_set):
                for j, med_j in enumerate(med_set):
                    if j <= i:
                        continue
                    prescriptions_adj[med_i, med_j] = 1
                    prescriptions_adj[med_j, med_i] = 1

    # 转换为CSR格式
    diag_adj = diag_adj.tocsr()
    sym_adj = sym_adj.tocsr()
    diag_sym_adj = diag_sym_adj.tocsr()
    sym_diag_adj = sym_diag_adj.tocsr()
    prescriptions_adj = prescriptions_adj.tocsr()

    

    # 保存
    with open(os.path.join(path, 'diag_adj.pkl'), 'wb') as f:
        pickle.dump(diag_adj, f)
    with open(os.path.join(path, 'proc_adj.pkl'), 'wb') as f:
        pickle.dump(sym_adj, f)
    with open(os.path.join(path, 'diag_proc_adj.pkl'), 'wb') as f:
        pickle.dump(diag_sym_adj, f)
    with open(os.path.join(path, 'proc_diag_adj.pkl'), 'wb') as f:
        pickle.dump(sym_diag_adj, f)
    with open(os.path.join(path, 'prescriptions_adj.pkl'), 'wb') as f:
        pickle.dump(prescriptions_adj, f)

    print("稀疏邻接矩阵已成功构建并保存。")

if __name__ == "__main__":
    construct_kgdnet_adj()



