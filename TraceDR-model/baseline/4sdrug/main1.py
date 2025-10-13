import torch
import torch.nn.functional as F
from eval.metrics import multi_label_metric, ddi_rate_score, calculate_rank_metrics
import numpy as np
from scipy.sparse import coo_matrix
from tqdm import trange, tqdm
from model.mymodel1 import Model
from model.radm1 import RAdam
import argparse
import os
import time
import csv
from datetime import datetime
from utils.dataset2 import PKLSet
import scipy.sparse as sp
import dill

if torch.cuda.is_available():
    torch.cuda.set_device(0)

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(description="Run")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--beta', type=float, default=1.0)#DDI 损失项的权重
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--score_threshold', type=float, default=0.8)# 预测阈值
    parser.add_argument('--alpha', type=float, default=0.5)#BPR 损失项的权重
    parser.add_argument('--dataset', type=str, default='')
    return parser.parse_known_args()


def _to_sparse_tensor(x, device):
    x_sparse = x.to_sparse()
    return x_sparse.coalesce().to(device)


def evaluate(model, test_loader, n_drugs, ddi_adj, device="cpu"):
    model.eval()
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    p_at_1, mrr, hit_at_5 = [],[],[]
    precison_at_5, recall_at_5, f1_at_5 = [], [], []
    smm_record = []
    med_cnt, visit_cnt = 0, 0
    for step, adm in tqdm(enumerate(test_loader)):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []

        syms, drugs = torch.tensor(adm[0]).to(device), torch.tensor(adm[2]).to(device)
        # print(syms, drugs)
        # print(syms.shape, drugs.shape)
        scores = model.evaluate(syms, device=device)
        # scores = 2 * torch.softmax(scores, dim=-1) - 1

        y_gt_tmp = np.zeros(n_drugs)
        #y_gt_tmp = np.zeros(n_drugs)
        valid_indices = drugs[(drugs < n_drugs) & (drugs >= 0)].cpu().numpy().astype(int)
        #gold_answers = drugs.cpu().numpy()
        y_gt_tmp[valid_indices] = 1

        ######y_gt_tmp[drugs.cpu().numpy()] = 1
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
        #print(result)
        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(np.array(y_gt),
                                                                                 np.array(y_pred),
                                                                                 np.array(y_pred_prob))
        adm_p1, adm_mrr, amd_h5, adm_precison5, adm_recall5, adm_f5 =  calculate_rank_metrics(result, n_drugs, drugs, device="cpu")
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
    # print(y_pred_label)
    #ddi_rate = ddi_rate_score(smm_record, ddi_adj)
    ddi_rate = 0
    return (np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), 1.0 * med_cnt / visit_cnt, ddi_rate,np.mean(p_at_1), np.mean(mrr), np.mean(hit_at_5), np.mean(precison_at_5), np.mean(recall_at_5), np.mean(f1_at_5))


    # ddi_rate = ddi_rate_score(smm_record, ddi_adj)
   
if __name__ == '__main__':
    print("程序初始化中...")
    args, unknown = parse_args()

    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint5e-3.pt')
    best_model_path = os.path.join(checkpoint_dir, 'best_model5e-3.pt')
    
    # 创建评估结果记录文件
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f'eval_results_{timestamp}.csv')
    
    # 初始化CSV文件，写入表头
    with open(results_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        headers = [
            'Epoch', 'Timestamp', 'JA', 'PRAUC', 'Precision', 'Recall', 
            'F1', 'Avg_Drugs', 'DDI_Rate', 'P@1', 'MRR', 'Hit@5', 
            'Precision@5', 'Recall@5', 'F1@5', 'Is_Best'
        ]
        writer.writerow(headers)
    
    print(f"评估结果将保存到: {results_file}")

    print("加载数据集...")
    pklSet = PKLSet(args.batch_size, args.dataset)

    print(f"药物训练数据长度: {len(pklSet.drug_train)}")
    if not pklSet.drug_train:
        raise ValueError("药物训练数据为空，请检查数据加载逻辑")

    print(f"\n[Info] 每个 epoch 的迭代次数（批次数）: {len(pklSet.sym_train)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    print(f"使用设备: {device}")

    print("处理DDI邻接矩阵...")
    ddi_adj_coo = pklSet.ddi_adj.tocoo()
    indices = torch.LongTensor(np.vstack((ddi_adj_coo.row, ddi_adj_coo.col)))
    values = torch.FloatTensor(ddi_adj_coo.data)
    shape = ddi_adj_coo.shape
    ddi_adj_tensor = torch.sparse_coo_tensor(indices, values, shape).to(device)

    print("处理药物特征...")
    drug_multihots = pklSet.drug_multihots
    coo = drug_multihots.tocoo()
    indices = torch.LongTensor(np.vstack([coo.row, coo.col]))
    values = torch.FloatTensor(coo.data)
    shape = coo.shape
    drug_multihots_sparse = torch.sparse_coo_tensor(indices, values, shape).to(device)

    print("初始化模型...")
    model = Model(
        pklSet.n_sym, pklSet.n_drug, ddi_adj_tensor, pklSet.sym_sets,
        drug_multihots_sparse, args.embedding_dim
    ).to(device)

    optimizer = RAdam(model.parameters(), lr=args.lr)
    print(f"模型总参数量: {sum(p.numel() for p in model.parameters())}")

    best_ja = -np.inf
    start_epoch = 0
    start_step = 0
    completed_epochs = 0

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        start_step = checkpoint.get('step', 0)
        completed_epochs = checkpoint.get('completed_epochs', 0)
        best_ja = checkpoint['best_ja']
        model.tensor_ddi_adj = model.tensor_ddi_adj.coalesce()
        print(f"已加载检查点: 已完成 {completed_epochs} 个epoch, 从 epoch {start_epoch} 的 step {start_step} 恢复")
    else:
        print("未找到检查点，从头开始训练")

    if completed_epochs >= args.epoch:
        print(f"训练已完成（共 {completed_epochs} 个epoch），直接进行最终评估...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        final_metrics = evaluate(model, pklSet.data_eval, pklSet.n_drug, pklSet.ddi_adj, device)
        print('\n' + '=' * 50)
        print("最终评估结果:")
        print('-' * 50)
        print(f"JA: {final_metrics[0]:.4f} | PRAUC: {final_metrics[1]:.4f}")  # ✔️
        print(f"精确率: {final_metrics[2]:.4f} | 召回率: {final_metrics[3]:.4f}")  # ✔️
        print(f"F1分数: {final_metrics[4]:.4f} | 平均药物数: {final_metrics[5]:.2f}")  # 修正为索引5
        print(f"DDI率: {final_metrics[6]:.4f}")  # 修正为索引6
        print(f"P@1: {final_metrics[7]:.4f} | MRR: {final_metrics[8]:.4f}")  # 修正为索引7和8
        print(f"hit@5: {final_metrics[9]:.4f} | precision@5: {final_metrics[10]:.4f}")  # 修正为索引9和10
        print(f"recall@5: {final_metrics[11]:.4f} | f1@5: {final_metrics[12]:.4f}")  # 修正为索引11和12
        print('=' * 50 + '\n')
        
        # 将最终评估结果写入CSV文件
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(results_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            row = [
                f"Final({completed_epochs})",  # Epoch
                current_time,                 # Timestamp  
                f"{final_metrics[0]:.4f}",   # JA
                f"{final_metrics[1]:.4f}",   # PRAUC
                f"{final_metrics[2]:.4f}",   # Precision
                f"{final_metrics[3]:.4f}",   # Recall
                f"{final_metrics[4]:.4f}",   # F1
                f"{final_metrics[5]:.2f}",   # Avg_Drugs
                f"{final_metrics[6]:.4f}",   # DDI_Rate
                f"{final_metrics[7]:.4f}",   # P@1
                f"{final_metrics[8]:.4f}",   # MRR
                f"{final_metrics[9]:.4f}",   # Hit@5
                f"{final_metrics[10]:.4f}",  # Precision@5
                f"{final_metrics[11]:.4f}",  # Recall@5
                f"{final_metrics[12]:.4f}",  # F1@5
                'Final'                       # Is_Best
            ]
            writer.writerow(row)
        
        print(f"最终评估结果已保存到: {results_file}")
    else:
        print(f"\n开始训练，从 epoch {start_epoch} 的 step {start_step} 开始...")
        
        # 创建总的进度条
        total_training_steps = (args.epoch - start_epoch) * len(pklSet.sym_train)
        training_progress = tqdm(total=total_training_steps, desc="训练进度", unit="batch")

        sym_train = list(pklSet.sym_train)
        drug_train = list(pklSet.drug_train)
        similar_sets_idx = list(pklSet.similar_sets_idx)
        train_data = list(zip(sym_train, drug_train, similar_sets_idx))

        for epoch in range(start_epoch, args.epoch):
            model.train()
            total_loss = 0.0

            if epoch == start_epoch and start_step > 0:
                current_data = train_data[start_step:]
                print(f"从 epoch {epoch + 1} 的 step {start_step} 恢复，剩余 {len(current_data)} 个批次")
            else:
                current_data = train_data
                start_step = 0

            # 使用同一个进度条来更新每个 batch 的进度
            for step, (syms, drugs, similar_idx) in enumerate(current_data):
                global_step = step + (start_step if epoch == start_epoch else 0)

                syms_dense = np.array(syms)
                syms_tensor = torch.from_numpy(syms_dense).to(device).long()
                drugs_dense = np.array(drugs)
                drugs_tensor = torch.from_numpy(drugs_dense).to(device).float()
                if drugs_tensor.max() > 1 or drugs_tensor.min() < 0:
                    drugs_tensor = (drugs_tensor > 0).float()
                similar_idx_tensor = torch.tensor(similar_idx).to(device)
                model.zero_grad()
                optimizer.zero_grad()
                scores, bpr, loss_ddi = model(syms_tensor, drugs_tensor, similar_idx_tensor, device)

                # 处理稀疏张量的 sigmoid 和损失计算
                scores_indices = scores.indices()  # [2, num_nonzero]
                scores_values = scores.values()  # [num_nonzero]
                sig_scores_values = torch.sigmoid(scores_values)  # 只对值应用 sigmoid

                # 构建完整的 sig_scores 稀疏张量
                sig_scores = torch.sparse_coo_tensor(
                    scores_indices, sig_scores_values, scores.size()
                ).coalesce()
                
                scores_values = scores_values.to(device)  # ✅ 正确：赋值后 scores_values 在 GPU
                scores_indices = (scores_indices[0].to(device), scores_indices[1].to(device))  # ✅ 元组索引也要转移
                #print("scores_values device:", scores_values.device)
                #print("drugs_tensor device:", drugs_tensor.device)
                #print("scores_indices device:", scores_indices[0].device, scores_indices[1].device)
                #sig_scores_values.to(device)
                #print(scores_values.device)
                #print(drugs_tensor.device)
                # 计算 BCE 损失（仅对非零值部分）
                bce_loss = F.binary_cross_entropy_with_logits(
                    scores_values, drugs_tensor[scores_indices[0], scores_indices[1]], reduction='sum'
                ) / model.n_drug

                # 计算熵（仅对非零值部分）
                entropy = -torch.mean(sig_scores_values * torch.log(sig_scores_values + 1e-8))

                # 组合损失
                loss = bce_loss + 0.5 * entropy +  args.alpha * bpr +  args.beta * loss_ddi

                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # 更新进度条
                training_progress.update(1)
                training_progress.set_postfix({
                    'epoch': epoch + 1,
                    'batch_loss': f"{loss.item():.4f}",
                    'avg_loss': f"{total_loss / (step + 1):.4f}"
                })

                if (global_step + 1) % 1000 == 0:
                    torch.save({
                        'epoch': epoch,
                        'step': global_step + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_ja': best_ja,
                        'completed_epochs': completed_epochs,
                        'loss': total_loss / (step + 1)
                    }, checkpoint_path)
                    print(f"检查点已保存至 {checkpoint_path} (epoch {epoch + 1}, step {global_step + 1})")

            completed_epochs = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'step': 0,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_ja': best_ja,
                'completed_epochs': completed_epochs,
                'loss': total_loss / len(current_data)
            }, checkpoint_path)
            print(f"Epoch {epoch + 1} 完成，检查点已保存")

            # 更新进度条描述
            training_progress.set_description(f"Epoch {epoch + 1}/{args.epoch} 完成")

            if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epoch:
                print(f"\n开始第 {epoch + 1} 轮评估...")
                eval_start = time.time()
                final_metrics = evaluate(model, pklSet.data_eval, pklSet.n_drug, pklSet.ddi_adj, device)
                print('\n' + '=' * 50)
                print("最终评估结果:")
                print('-' * 50)
                print(f"JA: {final_metrics[0]:.4f} | PRAUC: {final_metrics[1]:.4f}")  # ✔️
                print(f"精确率: {final_metrics[2]:.4f} | 召回率: {final_metrics[3]:.4f}")  # ✔️
                print(f"F1分数: {final_metrics[4]:.4f} | 平均药物数: {final_metrics[5]:.2f}")  # 修正为索引5
                print(f"DDI率: {final_metrics[6]:.4f}")  # 修正为索引6
                print(f"P@1: {final_metrics[7]:.4f} | MRR: {final_metrics[8]:.4f}")  # 修正为索引7和8
                print(f"hit@5: {final_metrics[9]:.4f} | precision@5: {final_metrics[10]:.4f}")  # 修正为索引9和10
                print(f"recall@5: {final_metrics[11]:.4f} | f1@5: {final_metrics[12]:.4f}")  # 修正为索引11和12
                print('=' * 50 + '\n')

                ja = final_metrics[0]
                is_best = False
                if ja > best_ja:
                    print(f"✨ 发现更好模型! JA从 {best_ja:.4f} 提升到 {ja:.4f}")
                    print(f"保存最佳模型到 {best_model_path}")
                    torch.save(model.state_dict(), best_model_path)
                    best_ja = ja
                    is_best = True
                
                # 将评估结果写入CSV文件
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(results_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    row = [
                        epoch + 1,                    # Epoch
                        current_time,                 # Timestamp  
                        f"{final_metrics[0]:.4f}",   # JA
                        f"{final_metrics[1]:.4f}",   # PRAUC
                        f"{final_metrics[2]:.4f}",   # Precision
                        f"{final_metrics[3]:.4f}",   # Recall
                        f"{final_metrics[4]:.4f}",   # F1
                        f"{final_metrics[5]:.2f}",   # Avg_Drugs
                        f"{final_metrics[6]:.4f}",   # DDI_Rate
                        f"{final_metrics[7]:.4f}",   # P@1
                        f"{final_metrics[8]:.4f}",   # MRR
                        f"{final_metrics[9]:.4f}",   # Hit@5
                        f"{final_metrics[10]:.4f}",  # Precision@5
                        f"{final_metrics[11]:.4f}",  # Recall@5
                        f"{final_metrics[12]:.4f}",  # F1@5
                        'Yes' if is_best else 'No'   # Is_Best
                    ]
                    writer.writerow(row)
                
                print(f"评估结果已保存到: {results_file}")

        # 训练完成后关闭进度条
        training_progress.close()

        print("\n训练完成! 正在加载最佳模型进行最终评估...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        final_metrics = evaluate(model, pklSet.data_eval, pklSet.n_drug, pklSet.ddi_adj, device)

        print('\n' + '=' * 50)
        print("最终评估结果:")
        print('-' * 50)
        print(f"JA: {final_metrics[0]:.4f} | PRAUC: {final_metrics[1]:.4f}")  # ✔️
        print(f"精确率: {final_metrics[2]:.4f} | 召回率: {final_metrics[3]:.4f}")  # ✔️
        print(f"F1分数: {final_metrics[4]:.4f} | 平均药物数: {final_metrics[5]:.2f}")  # 修正为索引5
        print(f"DDI率: {final_metrics[6]:.4f}")  # 修正为索引6
        print(f"P@1: {final_metrics[7]:.4f} | MRR: {final_metrics[8]:.4f}")  # 修正为索引7和8
        print(f"hit@5: {final_metrics[9]:.4f} | precision@5: {final_metrics[10]:.4f}")  # 修正为索引9和10
        print(f"recall@5: {final_metrics[11]:.4f} | f1@5: {final_metrics[12]:.4f}")  # 修正为索引11和12
        print('=' * 50 + '\n')
        
        # 将训练完成后的最终评估结果写入CSV文件
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(results_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            row = [
                f"Best_Model",                # Epoch
                current_time,                 # Timestamp  
                f"{final_metrics[0]:.4f}",   # JA
                f"{final_metrics[1]:.4f}",   # PRAUC
                f"{final_metrics[2]:.4f}",   # Precision
                f"{final_metrics[3]:.4f}",   # Recall
                f"{final_metrics[4]:.4f}",   # F1
                f"{final_metrics[5]:.2f}",   # Avg_Drugs
                f"{final_metrics[6]:.4f}",   # DDI_Rate
                f"{final_metrics[7]:.4f}",   # P@1
                f"{final_metrics[8]:.4f}",   # MRR
                f"{final_metrics[9]:.4f}",   # Hit@5
                f"{final_metrics[10]:.4f}",  # Precision@5
                f"{final_metrics[11]:.4f}",  # Recall@5
                f"{final_metrics[12]:.4f}",  # F1@5
                'Best_Final'                  # Is_Best
            ]
            writer.writerow(row)
        
        print(f"最终评估结果已保存到: {results_file}")
        print(f"所有评估结果记录在: {results_file}")