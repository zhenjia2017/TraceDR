import numpy as np
from tqdm import tqdm
import random
import pickle
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, jaccard_score, confusion_matrix, precision_score, recall_score

def ddi_score(outputs, output_thresh, ddi_gt, ddi_adj, ddi_thresh):
  num_adm = len(outputs)

  ## Patient DDI Calc. ##
  med_pair_ct = 0
  ddi_pair_ct = 0
  for i in range(146):
    for j in range(i+1, 146):
      if ddi_gt[i] != 0 and ddi_gt[j] != 0:
        med_pair_ct += 1
        ddi_pair_ct += int(ddi_adj[i][j])
  try:
    patient_ddi_rate = ddi_pair_ct / med_pair_ct
  except: patient_ddi_rate = 0

  ddi_loss_param = 1 if patient_ddi_rate <= ddi_thresh else max(
    0, 1-((patient_ddi_rate - ddi_thresh)/0.05) # 如果患者的真实 DDI 率 ≤ 阈值，ddi_loss_param = 1，说明没问题，可以专注推荐准确性。
  ) # 如果 DDI 率超过阈值，则线性降低 ddi_loss_param

  ## DDI Rate Calc. ##
  ddi_rates = [0] * num_adm

  for adm in range(num_adm):
    med_pair_ct = 0
    ddi_pair_ct = 0
    for i in range(len(output_thresh[adm])):
      for j in range(i+1, len(output_thresh[adm])):
        if output_thresh[adm][i] == 1 and output_thresh[adm][j] == 1:
          med_pair_ct += 1 # 总共同时开的药物对数
          ddi_pair_ct += int(ddi_adj[i][j]) # 这些组合中实际发生DDI的数量（根据 ddi_adj）
    try: ddi_rates[adm] = ddi_pair_ct / med_pair_ct
    except: ddi_rates[adm] = 0
  ddi_loss = np.mean(ddi_rates)

  return ddi_loss_param, ddi_loss

def eval(model, ehr_eval, patient_info_eval, ddi_KG, ddi_adj, ddi_thresh, device, med_num):
  model.eval()

  epoch_roc_auc = 0
  epoch_prauc = 0
  epoch_f1 = 0
  epoch_jaccard = 0
  epoch_precision = 0
  epoch_recall = 0
  epoch_meds = 0
  epoch_ddi = 0

  epoch_p_at_1 = 0
  epoch_mrr = 0
  epoch_hit_at_5 = 0
  epoch_precision_at_5 = 0
  epoch_recall_at_5 = 0
  epoch_f1_at_5 = 0
  epoch_jaccard_at_5 = 0

  num_patients = len(ehr_eval)

  for subj_id in tqdm(range(num_patients)):
    roc_auc = 0
    prauc = 0
    f1 = 0
    jaccard = 0
    num_meds = 0
    precision = 0
    recall = 0

    p_at_1 = 0
    mrr = 0
    hit_at_5 = 0
    precision_at_5 = 0
    recall_at_5 = 0
    f1_at_5 = 0
    jaccard_at_5 = 0

    patient_data = ehr_eval[subj_id]
    num_adm = len(patient_data)

    ## Get ground truths
    ground_truths = []

    ddi_gt = torch.zeros(med_num).to(device)

    for i in range(num_adm):
      gt = torch.zeros(med_num).to(device)

      try: 
        meds = patient_info_eval[subj_id][i][('patient', 'prescribed_to', 'medicine')]['edges'][1]
        gt[meds] = 1
        # print(meds)
        # print(gt)
      except: 
        gt[0] = 1
        

      ddi_gt += gt

      ground_truths.append(gt)

    outputs = model(patient_data, ddi_KG)

    output_thresh = [np.zeros(med_num) for _ in range(num_adm)]
    for i in range(num_adm):
      for j in range(med_num):
        output_thresh[i][j] = int((outputs[i][0][0][j].item() > -0.95))
      output_thresh[i] = output_thresh[i].astype(int)

    for i in range(num_adm):
      output_numpy = outputs[i][0].cpu().detach().numpy().squeeze()
      ground_truth_numpy = ground_truths[i].cpu().detach().numpy()
      output_prob = (output_numpy + 1) / 2  # 如果你用 tanh
      # print(output_prob)
      # print(ground_truth_numpy)

      roc_auc += roc_auc_score(ground_truth_numpy, output_prob)
      prauc += average_precision_score(ground_truth_numpy, output_prob)
      f1 += f1_score(ground_truth_numpy, output_thresh[i], zero_division=0.0)
      jaccard += jaccard_score(ground_truth_numpy, output_thresh[i], zero_division=0.0)

      precision += precision_score(ground_truth_numpy, output_thresh[i], zero_division=0.0)
      recall += recall_score(ground_truth_numpy, output_thresh[i], zero_division=0.0)

      cm = confusion_matrix(ground_truth_numpy, output_thresh[i])
      num_meds += cm[0][1] + cm[1][1]

      # Ranking metrics
      topk = 5
      sorted_indices = np.argsort(-output_prob)
      gt_indices = np.where(ground_truth_numpy == 1)[0]
      gt_set = set(gt_indices)

      # P@1
      if sorted_indices[0] in gt_set:
        p_at_1 += 1

      # MRR
      for rank, idx in enumerate(sorted_indices):
        if idx in gt_set:
          mrr += 1 / (rank + 1)
          break

      # Top-5 predictions
      pred_topk = set(sorted_indices[:topk])
      true_positives = len(pred_topk & gt_set)
      pred_union = pred_topk | gt_set
      pred_intersection = pred_topk & gt_set

      # Hit@5
      hit_at_5 += int(len(pred_intersection) > 0)

      # Precision@5
      precision_at_5 += true_positives / topk

      # Recall@5
      recall_at_5 += true_positives / len(gt_set) if len(gt_set) > 0 else 0

      # F1@5
      prec5 = true_positives / topk
      rec5 = true_positives / len(gt_set) if len(gt_set) > 0 else 0
      f1_5 = 2 * prec5 * rec5 / (prec5 + rec5) if (prec5 + rec5) > 0 else 0
      f1_at_5 += f1_5

      # Jaccard@5
      jaccard_5 = len(pred_intersection) / len(pred_union) if len(pred_union) > 0 else 0
      jaccard_at_5 += jaccard_5

    #_, ddi_loss = ddi_score(outputs, output_thresh, ddi_gt, ddi_adj, ddi_thresh)

    roc_auc = roc_auc / num_adm
    prauc = prauc / num_adm
    f1 = f1 / num_adm
    jaccard = jaccard / num_adm
    precision = precision / num_adm
    recall = recall / num_adm
    num_meds = num_meds / num_adm

    p_at_1 /= num_adm
    mrr /= num_adm
    hit_at_5 /= num_adm
    precision_at_5 /= num_adm
    recall_at_5 /= num_adm
    f1_at_5 /= num_adm
    jaccard_at_5 /= num_adm

    epoch_roc_auc += roc_auc
    epoch_prauc += prauc
    epoch_f1 += f1
    epoch_jaccard += jaccard
    epoch_precision += precision
    epoch_recall += recall
    epoch_meds += num_meds
    #epoch_ddi += ddi_loss
    epoch_ddi += 0

    epoch_p_at_1 += p_at_1
    epoch_mrr += mrr
    epoch_hit_at_5 += hit_at_5
    epoch_precision_at_5 += precision_at_5
    epoch_recall_at_5 += recall_at_5
    epoch_f1_at_5 += f1_at_5
    epoch_jaccard_at_5 += jaccard_at_5

  epoch_roc_auc = epoch_roc_auc / num_patients
  epoch_prauc = epoch_prauc / num_patients
  epoch_f1 = epoch_f1 / num_patients
  epoch_jaccard = epoch_jaccard / num_patients
  epoch_precision = epoch_precision / num_patients
  epoch_recall = epoch_recall / num_patients
  epoch_meds = epoch_meds / num_patients
  epoch_ddi = epoch_ddi / num_patients


  epoch_p_at_1 = epoch_p_at_1 / num_patients
  epoch_mrr = epoch_mrr / num_patients
  epoch_hit_at_5 = epoch_hit_at_5 / num_patients
  epoch_precision_at_5 = epoch_precision_at_5 / num_patients
  epoch_recall_at_5 = epoch_recall_at_5 / num_patients
  epoch_f1_at_5 = epoch_f1_at_5 / num_patients
  epoch_jaccard_at_5 = epoch_jaccard_at_5 / num_patients

  return epoch_roc_auc, epoch_prauc, epoch_f1, epoch_jaccard, epoch_precision, epoch_recall, epoch_meds, epoch_ddi, epoch_p_at_1, epoch_mrr, epoch_hit_at_5, epoch_precision_at_5, epoch_recall_at_5, epoch_f1_at_5,epoch_jaccard_at_5

def test(model, ehr_test, patient_info_test, ddi_KG, ddi_adj, ddi_thresh, device, med_num):
  metrics = []

  data_len = len(ehr_test)
  req_len = int(0.8 * data_len)

  for _ in 10:
    idx_list = list(range(data_len))
    random.shuffle(idx_list)
    idx_list = idx_list[:req_len]

    iter_ehr = ehr_test[idx_list]
    iter_patient_info = patient_info_test[idx_list]

    roc_auc, prauc, f1, jaccard, precision, recall, meds, ddi, p_at_1, mrr, hit_at_5, precison_5, recall_5, f1_5,ja_at_5 = eval(
      model, iter_ehr, iter_patient_info, ddi_KG, ddi_adj, ddi_thresh, device, med_num
    )

    print(f"ValAUC: {roc_auc:.4f}, PRAUC: {prauc:.4f}, F1: {f1:.4f}, Jaccard: {jaccard:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, DDI: {ddi:.4f}, num_meds: {meds:.4f}")
    print(f"p@1: {p_at_1:.4f}, mrr: {mrr:.4f}, hit@5: {hit_at_5:.4f}, precision@5: {precison_5:.4f}, recall@5: {recall_5:.4f}, f1@5: {f1_5:.4f}, ja@5: {ja_at_5:.4f}")
    metrics.append([roc_auc, prauc, f1, jaccard, precision, recall, meds, ddi, p_at_1, mrr, hit_at_5, precison_5, recall_5, f1_5,ja_at_5])

  with open('test_metrics.pkl', 'wb') as f:
    pickle.dump(metrics, f)

  return

def test2(model, ehr_test, patient_info_test, ddi_KG, ddi_adj, ddi_thresh, device, med_num):
  roc_auc, prauc, f1, jaccard, precision, recall, meds, ddi, p_at_1, mrr, hit_at_5, precison_5, recall_5, f1_5,ja_at_5 = eval(
    model, ehr_test, patient_info_test, ddi_KG, ddi_adj, ddi_thresh, device, med_num
  )

  print(f"ValAUC: {roc_auc:.4f}, PRAUC: {prauc:.4f}, F1: {f1:.4f}, Jaccard: {jaccard:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, DDI: {ddi:.4f}, num_meds: {meds:.4f}")
  print(f"p@1: {p_at_1:.4f}, mrr: {mrr:.4f}, hit@5: {hit_at_5:.4f}, precision@5: {precison_5:.4f}, recall@5: {recall_5:.4f}, f1@5: {f1_5:.4f}, ja@5: {ja_at_5:.4f}")
  #   metrics.append([roc_auc, prauc, f1, jaccard, precision, recall, meds, ddi, p_at_1, mrr, hit_at_5, precison_5, recall_5, f1_5,ja_at_5])

  # with open('test_metrics.pkl', 'wb') as f:
  #   pickle.dump(metrics, f)
