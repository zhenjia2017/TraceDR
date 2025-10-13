import dill
import numpy as np
import torch
import os
from scipy.sparse import lil_matrix, vstack
from tqdm import tqdm
from collections import defaultdict
from itertools import chain

class PKLSet(object):
    def __init__(self, batch_size, dataset):
        base_dir = os.path.join('DrugRec0716', dataset)
        self.eval_path = os.path.join(base_dir, 'data_eval.pkl')
        self.voc_path = os.path.join(base_dir, 'voc_final.pkl')
        self.ddi_adj_path = os.path.join(base_dir, 'ddi_A_final.pkl')

        print(f"\n[Loading] 尝试加载DDI邻接矩阵: {self.ddi_adj_path}")
        assert os.path.exists(self.ddi_adj_path), "DDI邻接矩阵文件不存在"
        self.ddi_adj = dill.load(open(self.ddi_adj_path, 'rb'))
        # 直接切片取前 450 行和前 450 列
        
        print("[Success] DDI邻接矩阵加载成功")
        self.n_drug = 100065
        self.n_sym = 5611
        self.ddi_adj = self.ddi_adj[:self.n_drug, :self.n_drug]  # shape: (450, 450)
        self.sym_train, self.drug_train, self.data_eval = self.check_file(batch_size, dataset)
        self.sym_sets, self.drug_multihots = self.mat_train_data(dataset)
        self.similar_sets_idx = self.find_similar_set_by_ja(self.sym_train)

    def check_file(self, batch_size, dataset):
        base_dir = os.path.join('DrugRec0716', dataset)
        sym_path = os.path.join(base_dir, f'sym_train_{batch_size}.pkl')
        drug_path = os.path.join(base_dir, f'drug_train_{batch_size}.pkl')

        if not os.path.exists(sym_path):
            print(f"\n[Generating] 未找到批次数据，开始生成: {sym_path}")
            self.gen_batch_data(batch_size, dataset)
        else:
            print(f"\n[Loading] 直接加载现有批次数据: {sym_path}")

        data_eval = dill.load(open(self.eval_path, 'rb'))
        #print("data_eval 原始症状索引（前5个 adm[0]）：", [adm[0] for adm in data_eval[:5]])
        #print("data_eval 原始药物索引（前5个 adm[2]）：", [adm[2][:10] for adm in data_eval[:5]])
        all_syms = [idx for adm in data_eval for idx in adm[0]]
        all_drugs = [idx for adm in data_eval for idx in adm[2]]
        print("data_eval 症状索引范围:", [min(all_syms), max(all_syms)])
        print("data_eval 药物索引范围:", [min(all_drugs), max(all_drugs)])

        return self.load_data(sym_path, drug_path)

    def load_data(self, sym_path, drug_path):
        print(f"\n[Loading] 加载词汇表: {self.voc_path}")
        assert os.path.exists(self.voc_path), "词汇表文件不存在"
        voc = dill.load(open(self.voc_path, 'rb'))
        sym_voc, med_voc = voc['sym_voc'], voc['med_voc']
        #self.n_sym = len(sym_voc['idx2word'])  # 5531
        #self.n_drug = 10000  # 99480
        print(f"[Info] 症状数量: {self.n_sym}, 药物数量: {self.n_drug}")

        print(f"\n[Loading] 加载症状训练数据: {sym_path}")
        assert os.path.exists(sym_path), "症状训练文件不存在"
        sym_train = dill.load(open(sym_path, 'rb'))
        #print("原始数据样本（前5个）：", sym_train[:5])
        # 验证症状索引范围 [1, 5531]
        for i, batch in enumerate(sym_train):
            for j, sym_list in enumerate(batch):
                if not sym_list:
                    print(f"警告: 批次 {i} 的症状列表 {j} 为空")
                    continue
                if any(idx < 1 or idx > self.n_sym for idx in sym_list):
                    print(f"警告: 批次 {i} 的症状列表 {j} 包含无效索引: {sym_list}")

        print(f"\n[Loading] 加载药物训练数据: {drug_path}")
        assert os.path.exists(drug_path), "药物训练文件不存在"
        drug_train = dill.load(open(drug_path, 'rb'))
        #print("原始数据样本（前5个）：", drug_train[:5])
        # 验证药物索引范围 [1, 99480]
        '''
        for i, batch in enumerate(drug_train):
            for j, drug_array in enumerate(batch):
                if isinstance(drug_array, np.ndarray):
                    drugs = np.where(drug_array)[0].tolist()  # 注意：这里仍可能是0-based，需后续调整
                else:
                    drugs = drug_array
                if not drugs:
                    print(f"警告: 批次 {i} 的药物列表 {j} 为空，已跳过")
                    continue
                if any(idx < 1 or idx > self.n_drug for idx in drugs):
                    print(f"警告: 批次 {i} 的药物列表 {j} 包含无效索引: {drugs}")
        '''
        print(f"\n[Loading] 加载评估数据: {self.eval_path}")
        assert os.path.exists(self.eval_path), "评估数据文件不存在"
        data_eval = dill.load(open(self.eval_path, 'rb'))
        #print("数据样本（前5个）：", data_eval[:5])
        return sym_train, drug_train, data_eval
        
    def count_sym(self, dataset):
        base_dir = os.path.join('DrugRec0716', dataset)
        train_path = os.path.join(base_dir, 'data_train.pkl')
        print(f"\n[Loading] 统计症状出现次数: {train_path}")
        assert os.path.exists(train_path), "训练数据文件不存在"
        data = dill.load(open(train_path, 'rb'))

        countings = np.zeros(self.n_sym + 1)  # +1 因为索引从1到5531
        max_sym = 0
        for adm in data:
            syms = adm[0]  # 保持原始索引
            
            if syms:
                current_max = max(syms)
                if current_max > self.n_drug:
                    continue
                if current_max > max_sym:
                    max_sym = current_max
            countings[syms] += 1  # 直接使用1-based索引
        print(f"[Debug] 最大症状索引: {max_sym}, n_sym: {self.n_sym}")
        return countings

    def mat_train_data(self, dataset):
        base_dir = os.path.join('DrugRec0716', dataset)
        train_path = os.path.join(base_dir, 'data_train.pkl')
        sym_sets_path = os.path.join(base_dir, 'sym_sets.pkl')
        drug_multihots_path = os.path.join(base_dir, 'drug_multihots.pkl')

        if os.path.exists(sym_sets_path) and os.path.exists(drug_multihots_path):
            print(f"\n[Loading] 直接加载预处理数据: {sym_sets_path}")
            with open(sym_sets_path, 'rb') as f:
                sym_sets = dill.load(f)
            with open(drug_multihots_path, 'rb') as f:
                drug_multihots = dill.load(f)
            return sym_sets, drug_multihots

        print(f"\n[Processing] 生成预处理数据并保存...")
        data_train = dill.load(open(train_path, 'rb'))
        sym_sets, drug_sets_multihot = [], []

        for adm in tqdm(data_train, desc="Processing data_train"):
            syms = adm[0]  # 保持原始1-based索引
            drugs = adm[2]  # 保持原始1-based索引
            sym_sets.append(syms)
            drug_multihot = lil_matrix((1, self.n_drug), dtype=np.int64)
            drug_multihot[0, [d - 1 for d in drugs if d <= self.n_drug]] = 1  # lil_matrix 使用0-based，需减1
            drug_sets_multihot.append(drug_multihot)

        drug_multihots = vstack(drug_sets_multihot)

        with open(sym_sets_path, 'wb') as f:
            dill.dump(sym_sets, f)
        with open(drug_multihots_path, 'wb') as f:
            dill.dump(drug_multihots, f)
        print(f"[Success] 预处理数据已保存至: {sym_sets_path}")

        return sym_sets, drug_multihots

    def gen_batch_data(self, batch_size, dataset):
        base_dir = os.path.join('DrugRec0716', dataset)
        print(f"\n[Generating] 初始化批次数据生成...")
        voc = dill.load(open(self.voc_path, 'rb'))
        sym_voc, med_voc = voc['sym_voc'], voc['med_voc']
        #self.n_sym =  # 5531
        #self.n_drug = 450  # 99480

        sym_count = self.count_sym(dataset)
        size_dict, drug_dict = defaultdict(list), defaultdict(list)
        sym_sets, drug_sets = [], []
        s_set_num = 0

        train_path = os.path.join(base_dir, 'data_train.pkl')
        data = dill.load(open(train_path, 'rb'))
        max_drug = 0

        for adm in data:
            syms = adm[0]  # 保持原始索引
            drugs = adm[2]  # 保持原始索引1
            sym_sets.append(syms)
            drug_sets.append(drugs)
            s_set_num += 1

            if drugs:
                current_max = max(drugs)
                if current_max > max_drug:
                    max_drug = current_max
        print(f"[Debug] 训练集中最大药物索引: {max_drug}, n_drug: {self.n_drug}")

        for adm in data:
            syms = adm[0]
            drugs = adm[2]
            drug_multihot = np.zeros(self.n_drug, dtype=np.bool_)
            drug_multihot[[d - 1 for d in drugs if d <= 450]] = 1  # numpy 数组使用0-based
            key = len(syms)
            size_dict[key].append(syms)
            drug_dict[key].append(drug_multihot)

        

        sym_train, drug_train = [], []
        keys = sorted(size_dict.keys())
        for size in keys:
            data_size = size_dict[size]
            num_batches = (len(data_size) + batch_size - 1) // batch_size
            for i in range(num_batches):
                start = i * batch_size
                end = start + batch_size
                sym_batch = data_size[start:end]
                drug_batch = drug_dict[size][start:end]
                sym_train.append(sym_batch)
                drug_train.append(drug_batch)

        sym_save_path = os.path.join(base_dir, f'sym_train_{batch_size}.pkl')
        drug_save_path = os.path.join(base_dir, f'drug_train_{batch_size}.pkl')
        with open(sym_save_path, 'wb') as f:
            dill.dump(sym_train, f)
        with open(drug_save_path, 'wb') as f:
            dill.dump(drug_train, f)
        print(f"[Success] 批次数据已保存至: {sym_save_path}")
        print(f"生成的药物训练数据长度: {len(drug_train)}")
        if not drug_train:
            raise ValueError("生成的药物训练数据为空，请检查数据生成逻辑")

    def find_similar_set_by_ja(self, sym_train):
        #print("\n[Processing] 计算相似症状集...")
        similar_sets = [[] for _ in range(len(sym_train))]
        for idx, sym_batch in enumerate(sym_train):
            if len(sym_batch) == 0 or len(sym_batch[0]) <= 2:
                continue
            batch_sets = [set(s) for s in sym_batch]
            similar_sets[idx] = [0] * len(sym_batch)
            for i in range(len(batch_sets)):
                max_overlap = -1
                best_j = i
                for j in range(len(batch_sets)):
                    if i == j:
                        continue
                    overlap = len(batch_sets[i] & batch_sets[j])
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_j = j
                similar_sets[idx][i] = best_j
        #print("[Success] 相似症状集计算完成")
        return similar_sets