import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import scipy.sparse as sp

from .aggregation1 import Attention
import torch.sparse as tsp

class Model(nn.Module):
    def __init__(self, n_sym, n_drug, ddi_adj, sym_sets, drug_multihots, embed_dim=32, dropout=0.4):
        super(Model, self).__init__()
        print("正在初始化模型...")
        self.n_sym, self.n_drug = n_sym, n_drug  # 5531, 99480
        self.embed_dim, self.dropout = embed_dim, dropout
        self.sym_sets, self.drug_multihots = sym_sets, drug_multihots
        self.sym_embeddings = nn.Embedding(self.n_sym + 1, self.embed_dim)
        self.drug_embeddings = nn.Embedding(self.n_drug + 1, self.embed_dim)
        self.sym_agg = Attention(self.embed_dim)

        # 调整DDI邻接矩阵以匹配 n_drug
        ddi_adj = ddi_adj.coalesce()
        if ddi_adj.shape[0] > n_drug:
            print(f"调整DDI矩阵大小从 {ddi_adj.shape} 到 ({n_drug}, {n_drug})")
            indices = ddi_adj.indices()
            values = ddi_adj.values()
            mask = (indices[0] < n_drug) & (indices[1] < n_drug)
            new_indices = indices[:, mask]
            new_values = values[mask]
            ddi_adj = torch.sparse_coo_tensor(new_indices, new_values, (n_drug, n_drug)).coalesce()

        self.register_buffer('tensor_ddi_adj', ddi_adj)
        self.init_parameters()
        print("模型初始化完成")

    def init_parameters(self):
        print("正在初始化参数...")
        stdv = 1.0 / math.sqrt(self.embed_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        print("参数初始化完成")

    def _sparse_bmm(self, x, y):
        if x.is_sparse:
            x = x.to_dense()
        if y.is_sparse:
            y = y.to_dense()
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if y.dim() == 2:
            y = y.unsqueeze(0)
        return torch.bmm(x, y)

    def sparse_bmm_chunked(self, sparse_mat, adj, chunk_size=10000):
        n_drug = adj.shape[1]
        total_sum = torch.tensor(0.0, device=sparse_mat.device)
        adj = adj.coalesce()

        if sparse_mat.dim() == 1:
            sparse_mat = sparse_mat.unsqueeze(0)
        if sparse_mat.dim() != 2:
            raise ValueError(f"sparse_mat 必须是二维的，实际维度为 {sparse_mat.dim()}")

        if not sparse_mat.is_sparse:
            raise ValueError(f"sparse_mat 必须是稀疏张量，实际类型为 {sparse_mat.type()}")

        #print(f"sparse_bmm_chunked: sparse_mat shape: {sparse_mat.shape}, adj shape: {adj.shape}")

        for i in range(0, n_drug, chunk_size):
            end = min(i + chunk_size, n_drug)
            mask = (adj.indices()[1] >= i) & (adj.indices()[1] < end)
            sub_indices = adj.indices()[:, mask]
            sub_values = adj.values()[mask]

            if sub_indices.size(1) == 0:
                continue

            sub_indices[1] = sub_indices[1] - i
            sub_adj = torch.sparse_coo_tensor(sub_indices, sub_values, (n_drug, end - i),device=sparse_mat.device).coalesce()
            chunk = torch.sparse.mm(sparse_mat, sub_adj).coalesce()
            total_sum += chunk.values().sum()

        return total_sum

    def _merge_sparse_tensors(self, sparse_tensors, size):
        """合并多个稀疏张量为一个"""
        if not sparse_tensors:
            return torch.sparse_coo_tensor(
                torch.empty(2, 0, device=sparse_tensors[0].device if sparse_tensors else 'cpu'),
                torch.empty(0, device=sparse_tensors[0].device if sparse_tensors else 'cpu'),
                size
            ).coalesce()

        all_indices = []
        all_values = []
        for t in sparse_tensors:
            all_indices.append(t.indices())
            all_values.append(t.values())

        indices = torch.cat(all_indices, dim=1)
        values = torch.cat(all_values)
        return torch.sparse_coo_tensor(indices, values, size).coalesce()

    def forward(self, syms, drugs, similar_idx, device="cpu"):
        #print("开始前向传播...")
        #print(f"DDI 矩阵是否合并: {self.tensor_ddi_adj.is_coalesced()}")
        #print(f"syms shape: {syms.shape}")

        # 获取症状嵌入并聚合
        sym_embeds = self.sym_embeddings(syms.long())  # [batch_size, num_syms, embed_dim]
        s_set_embeds = self.sym_agg(sym_embeds)  # [batch_size, embed_dim] 或 [embed_dim]
        if s_set_embeds.dim() == 1:
            s_set_embeds = s_set_embeds.unsqueeze(0)  # 确保 [batch_size, embed_dim]
        s_set_embeds = F.normalize(s_set_embeds, p=2, dim=-1)

        # 分块计算稀疏 scores
        chunk_size = 10000
        scores_list = []
        for i in range(0, self.n_drug, chunk_size):
            end = min(i + chunk_size, self.n_drug)
            drug_ids = torch.arange(i + 1, end + 1, device=device)  # 1-based
            drug_embeds_chunk = self.drug_embeddings(drug_ids)  # [chunk_size, embed_dim]
            drug_embeds_chunk = F.normalize(drug_embeds_chunk, p=2, dim=-1)

            # 计算稠密 chunk_scores 并转换为稀疏形式
            chunk_scores = torch.matmul(s_set_embeds, drug_embeds_chunk.T)  # [batch_size, chunk_size]
            chunk_scores = torch.clamp(chunk_scores, -10, 10)
            chunk_prob = torch.sigmoid(chunk_scores)  # [batch_size, chunk_size]
            #scores_list.append(chunk_prob)
            # 转换为稀疏张量，只保留大于阈值的预测
            threshold = 0.5
            mask = chunk_prob > threshold
            indices = mask.nonzero(as_tuple=False)  # [num_nonzero, 2]
            if indices.size(0) > 0:
                values = chunk_prob[mask]
                indices[:, 1] = indices[:, 1] + i
                sparse_chunk = torch.sparse_coo_tensor(
                    indices.T, values, (s_set_embeds.shape[0], self.n_drug)
                ).coalesce()
                scores_list.append(sparse_chunk)

        # 合并所有稀疏块
        scores = self._merge_sparse_tensors(scores_list, (s_set_embeds.shape[0], self.n_drug))

        # 确保 neg_pred_sp 是二维的
        neg_pred_sp = scores
        if neg_pred_sp.dim() == 1:
            neg_pred_sp = neg_pred_sp.unsqueeze(0)
        #print(f"neg_pred_sp shape: {neg_pred_sp.shape}")

        # 确保 tensor_ddi_adj 是合并的
        if not self.tensor_ddi_adj.is_coalesced():
            self.tensor_ddi_adj = self.tensor_ddi_adj.coalesce()
            #print("DDI 矩阵未合并，已执行 coalesce 操作")

        # 计算 batch_neg
        batch_neg = 0.000001 * self.sparse_bmm_chunked(neg_pred_sp, self.tensor_ddi_adj)

        scores_aug = torch.tensor(0.0, device=device)
        if syms.shape[0] > 2 and syms.shape[1] > 2:
            all_drugs = torch.arange(1, self.n_drug + 1, device=device)
            all_drug_embeds = self.drug_embeddings(all_drugs)
            all_drug_embeds = F.normalize(all_drug_embeds, p=2, dim=-1)
            scores_aug = self.intraset_augmentation(syms, drugs, all_drug_embeds, similar_idx, device)
            batch_neg += self.interset_ddi(syms, s_set_embeds, drugs, all_drug_embeds, similar_idx, device)

        return scores, scores_aug, batch_neg

    def evaluate(self, syms, device='cpu'):
        sym_embeds, drug_embeds = self.sym_embeddings(syms.long()), self.drug_embeddings(torch.arange(0, self.n_drug).long().to(device))
        s_set_embed = self.sym_agg(sym_embeds).unsqueeze(0)
        # s_set_embed = torch.mean(sym_embeds, dim=0).unsqueeze(0)
        scores = torch.mm(s_set_embed, drug_embeds.transpose(-1, -2)).squeeze(0)

        return scores


    def intraset_augmentation(self, syms, drugs, all_drug_embeds, similar_idx, device):
        #print("开始集内增强...")
        if similar_idx is None or len(similar_idx) == 0:
            return torch.tensor(0.0, device=device)
        selected_drugs = drugs[similar_idx].to_sparse()
        common_drug = drugs.to_sparse().mul(selected_drugs)
        diff_drug = (drugs - selected_drugs.to_dense()).clamp(min=0).to_sparse()

        #print("计算交叉熵...")
        common_embeds = self._get_common_embeds(syms, similar_idx)
        x = self.sym_agg(common_embeds)
        y = all_drug_embeds.t()
        scores = torch.matmul(x, y)
        #print("计算交叉熵完成")
        #print("集内增强完成")
        return F.binary_cross_entropy_with_logits(scores, common_drug.to_dense())

    def _get_common_embeds(self, syms, similar_idx):
        #print("提取共有特征...")
        common_sym = (syms * syms[similar_idx]).to_sparse()
        batch_size = syms.shape[0]
        indices = common_sym.indices()
        values = common_sym.values()

        embeds = self.sym_embeddings(indices[1].long())
        embeds = embeds * values.unsqueeze(-1)

        batch_indices = indices[0]
        common_embeds = torch.zeros(batch_size, syms.shape[1], self.embed_dim, device=syms.device)
        common_embeds[batch_indices, indices[1]] = embeds
        #print("提取共有特征完成")
        return common_embeds

    def interset_ddi(self, syms, s_set_embed, drugs, all_drug_embeds, similar_idx, device):
        #print("计算跨集DDI...")
        diff_drug = (drugs - drugs[similar_idx]).abs().to_sparse()
        diff_drug = diff_drug.coalesce()

        indices = diff_drug.indices()
        values = diff_drug.values()
        batch_idx = indices[0]
        sum_diff = torch.zeros(drugs.shape[0], 1, device=device)
        sum_diff.scatter_add_(0, batch_idx.unsqueeze(-1), values.unsqueeze(-1))

        diff_embed = torch.sparse.mm(diff_drug, all_drug_embeds) / (sum_diff + 1e-6)
        common_embed = self.sym_agg(self._get_common_embeds(syms, similar_idx))
        #print("跨集DDI计算完成")
        return 0.0001 * torch.sigmoid((common_embed * diff_embed)).sum()