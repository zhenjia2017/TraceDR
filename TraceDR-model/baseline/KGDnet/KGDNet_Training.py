import pickle
import numpy as np
import argparse
import tqdm
import json
import torch
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import sys
sys.path.append("..")

from KGDNet_Model import KGDNet
from util import ddi_score, eval, test,test2
from dataloader import get_ehr_data

model_name = 'kgdnet0718'
resume_path = 'saved/kgdnet0718/Epoch_1_p@1_0.0009_ja_0.0007.model'

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true', default=False, help='Test mode')
parser.add_argument('--lr', type=int, default=32, help='Set the learning rate')
parser.add_argument('--batch_size', type=int, default=8, help='Set the batch size')
parser.add_argument('--model_name', type=str, default=model_name, help="model name")
parser.add_argument('--embed_dim', type=int, default=64, help='Embedding dimension size')
parser.add_argument('--ddi_thresh', type=float, default=0.08, help="Set DDI threshold")
parser.add_argument('--resume_path', type=str, default=resume_path, help='Resume path')
parser.add_argument('--save_path', type=str, default='best_model.pt', help='Path to save the best model')
args = parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
  if not os.path.exists(os.path.join("saved", model_name)):
    os.makedirs(os.path.join("saved", model_name))

  patient_info, ehr_KGs, ddi_KG, ddi_adj, voc, num_clinical_nodes, num_med_nodes = get_ehr_data(device)
  #ehr_KGs, patient_info, ddi_KG, ddi_adj, voc = []

  split_point = int(len(ehr_KGs) * 3 / 5)
  eval_len = int(len(ehr_KGs[split_point:]) / 2)

  ehr_train = ehr_KGs[:split_point]
  ehr_test = ehr_KGs[split_point:split_point + eval_len]
  ehr_eval = ehr_KGs[split_point+eval_len:]

  med_num = len(voc['med_voc']['idx2word'])+1

  patient_info_train = patient_info[:split_point]
  patient_info_test = patient_info[split_point:split_point + eval_len]
  patient_info_eval = patient_info[split_point+eval_len:]

  ## Training ##
  metrics_arr = []

  model = KGDNet(embed_dim=args.embed_dim, batch_size=args.batch_size,clinical_vocab_size=num_clinical_nodes, medicine_vocab_size=num_med_nodes).to(device)
  print(f"Total trainable parameters: {count_parameters(model):,}")
  if args.test:
    print("test..")
    model.load_state_dict(torch.load(open(args.resume_path, 'rb')))
    test2(model, ehr_test, patient_info_test, ddi_KG, ddi_adj, args.ddi_thresh, device,med_num)
    return

  optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-4, alpha=0.995)

  num_epochs = 5
  best_jaccard = 0

  for epoch in range(num_epochs):
    model.train()

    epoch_loss = 0
    epoch_roc_auc = 0
    epoch_prauc = 0
    epoch_f1 = 0
    epoch_jaccard = 0
    epoch_precision = 0
    epoch_recall = 0
    epoch_meds = 0
    epoch_ddi = 0

    num_patients = len(ehr_train)

    for subj_id in tqdm.tqdm(range(num_patients)):
      patient_data = ehr_train[subj_id]
      #patient_data = [patient_data]
      num_adm = len(patient_data)

      ## Get ground truths
      ground_truths = []

      ddi_gt = torch.zeros(med_num).to(device)

      for i in range(num_adm):
        gt = torch.zeros(med_num).to(device)
        # meds = patient_info_train[subj_id][i][('patient', 'prescribed_to', 'medicine')]['edges'][1]
        # gt[meds] = 1
        try:
          meds = patient_info_train[subj_id][i][('patient', 'prescribed_to', 'medicine')]['edges'][1]
          #print(meds)
          gt[meds] = 1
          #print(gt)
        except: gt[0] = 1

        ddi_gt += gt

        ground_truths.append(gt)

      ##############################
      # Zero the gradients
      optimizer.zero_grad()

      outputs = model(patient_data, ddi_KG)
      #print(outputs)
      #print(outputs.shape)

      ##############################

      ## Apply threshold ##
      output_thresh = [np.zeros(med_num) for _ in range(num_adm)]
      for i in range(num_adm):
        for j in range(med_num):
          output_thresh[i][j] = int((outputs[i][0][0][j].item() > -0.95))
        output_thresh[i] = output_thresh[i].astype(int)

      loss = 0
      bce_loss = 0
      mlm_loss = 0
      ddi_loss = 0

      ## Recommendation Loss ##
      for i in range(num_adm):
        pred = outputs[i][0].squeeze(0)
        #print(pred.shape)
        bce_loss += F.binary_cross_entropy_with_logits(pred, ground_truths[i])
        #mlm_loss += F.multilabel_margin_loss(pred, ground_truths[i].long())
      bce_loss = bce_loss / num_adm
      #mlm_loss = mlm_loss / num_adm

      #ddi_loss_param, ddi_loss = ddi_score(outputs, output_thresh, ddi_gt, ddi_adj, args.ddi_thresh)

      ## Total Loss ##
      rec_loss_param = 0.95
      # loss = ddi_loss_param * (
      #     rec_loss_param * bce_loss + (1 - rec_loss_param) * mlm_loss
      # ) + (1 - ddi_loss_param) * ddi_loss
      loss = 0.9 * bce_loss

      loss.backward()
      # with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
      #   loss.backward()
      # print(prof.key_averages().table(sort_by="cuda_time_total"))
      nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
      optimizer.step()

      epoch_loss += loss
      #epoch_ddi += ddi_loss
      epoch_ddi += 0

    epoch_loss = loss / num_patients

    epoch_roc_auc, epoch_prauc, epoch_f1, 
    epoch_jaccard, epoch_precision, 
    roc_auc, prauc, f1, jaccard, precision, recall, meds, ddi, p_at_1, mrr, hit_at_5, precison_5, recall_5, f1_5,ja_at_5 = eval(
      model, ehr_eval, patient_info_eval, ddi_KG, ddi_adj, args.ddi_thresh, device, med_num
    )

    metrics_arr.append([epoch+1, epoch_loss.item(), roc_auc, prauc, f1, jaccard, precision, recall, ddi, meds,p_at_1, mrr, hit_at_5, precison_5, recall_5, f1_5,ja_at_5])

    print(f"Epoch: {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, ValAUC: {roc_auc:.4f}, PRAUC: {prauc:.4f}, F1: {f1:.4f}, Jaccard: {jaccard:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, DDI: {ddi:.4f}, num_meds: {meds:.4f}")
    print(f"p@1: {p_at_1:.4f}, mrr: {mrr:.4f}, hit@5: {hit_at_5:.4f}, precision@5: {precison_5:.4f}, recall@5: {recall_5:.4f}, f1@5: {f1_5:.4f}, ja@5: {ja_at_5:.4f}")
    # Save the best model based on Jaccard
    if jaccard > best_jaccard:
      best_jaccard = jaccard
      torch.save(model.state_dict(), open( os.path.join('saved', model_name, 'Epoch_%d_p@1_%.4f_ja_%.4f.model' % (epoch, p_at_1, jaccard)), 'wb'))
      print(f"Saved new best model at epoch {epoch+1} with Jaccard: {jaccard:.4f}")
    with open(os.path.join('saved', model_name, f'epoch_{epoch+1}_metrics.json'), 'w') as f:
      json.dump(metrics_arr, f, indent=4)

  with open(os.path.join('saved', model_name, 'all_metrics.pkl'), 'wb') as f:
    pickle.dump(metrics_arr, f)


def test_cuda():
    device = torch.device('cuda:0')
    x = torch.randn(100, 100).to(device)
    y = torch.randn(100, 100).to(device)
    z = x @ y  # 简单矩阵乘法
    print("CUDA计算成功:", z.mean())
if __name__ == '__main__':
    main()
    #test_cuda()