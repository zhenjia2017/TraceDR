import torch
import torch.nn as nn
import numpy as np
import dill
import time
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import os
import torch.nn.functional as F
from collections import defaultdict

import sys
sys.path.append("..")
from models2 import Retain
from util import llprint, multi_label_metric, ddi_rate_score, get_n_params, calculate_rank_metrics

torch.manual_seed(1203)
model_name = 'Retain0718'
resume_name = 'Epoch_39_p@1_0.1676_ja_0.0778.model'

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
    

def eval(model, data_eval, voc_size, epoch):
    
    # evaluate
    print('')
    model.eval()
    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    p_at_1, mrr, hit_at_5 = [], [], []
    precision_at_5, recall_at_5, f1_at_5 = [], [], []
    ja_at_5 = []
    case_study = defaultdict(dict)
    med_cnt = 0
    visit_cnt = 0
    num = 0
    for step, input1 in enumerate(data_eval):
        # if num >= 10:
        #     break
        # else:
        #     num += 1
        # if len(input) < 2: # visit > 2
        #     continue
        y_gt = []
        y_pred = []
        y_pred_prob = []
        y_pred_label = []
        input = [input1]
        for i in range(0, len(input)):

            y_pred_label_tmp = []
            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[input[i][2]] = 1
            y_gt.append(y_gt_tmp)
            drugs = torch.tensor(input[i][2])

            target_output1 = model(input[:i+1])
            target_output1 = target_output1.unsqueeze(1)
            
            target_output1 = F.sigmoid(target_output1).detach().cpu().numpy()[0]
            target_output1 = target_output1.flatten()
            #print(target_output1)
             # Flatten the output to 1D array before thresholding
            y_pred_tmp = target_output1.copy()
            y_pred_prob.append(target_output1)
            y_pred_tmp[y_pred_tmp >= 0.3] = 1
            y_pred_tmp[y_pred_tmp < 0.3] = 0
            y_pred.append(y_pred_tmp)
            for idx, value in enumerate(y_pred_tmp):
                if value == 1:
                    y_pred_label_tmp.append(idx)
            y_pred_label.append(y_pred_label_tmp)
            med_cnt += len(y_pred_label_tmp)
            visit_cnt += 1

        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(np.array(y_gt), np.array(y_pred),
                                                                                   np.array(y_pred_prob))
        adm_p1, adm_mrr, amd_h5, adm_precison5, adm_recall5, adm_f5 = calculate_rank_metrics(target_output1, drugs)
        adm_ja_at_5 = jaccard_at_k(np.array(y_gt), np.array(y_pred_prob), k=5)
        case_study[adm_ja] = {'ja': adm_ja, 'patient':input, 'y_label':y_pred_label}
        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        p_at_1.append(adm_p1)
        mrr.append(adm_mrr)
        hit_at_5.append(amd_h5)
        precision_at_5.append(adm_precison5)
        recall_at_5.append(adm_recall5)
        f1_at_5.append(adm_f5)
        ja_at_5.append(adm_ja_at_5)
        llprint('\rEval--Epoch: %d, Step: %d/%d' % (epoch, step, len(data_eval)))

    dill.dump(case_study, open(os.path.join('saved', model_name, 'case_study.pkl'), 'wb'))
    # ddi rate
    #ddi_rate = ddi_rate_score(smm_record)
    ddi_rate = 0
    llprint('\tDDI Rate: %.4f, Jaccard: %.4f,  PRAUC: %.4f, AVG_PRC: %.4f, AVG_RECALL: %.4f, AVG_F1: %.4f\n' % (
        ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1),
    ))
    llprint('\tp@1: %.4f, mrr: %.4f,  hit@5: %.4f, precision@5: %.4f, recall@5: %.4f, f1@5: %.4f, jaccard@5: %.4f\n' % (
        np.mean(p_at_1), np.mean(mrr), np.mean(hit_at_5), np.mean(precision_at_5), np.mean(recall_at_5),
        np.mean(f1_at_5),np.mean(ja_at_5)
    ))
    print('avg med', med_cnt / visit_cnt)

    return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), np.mean(p_at_1), np.mean(mrr), np.mean(hit_at_5), np.mean(precision_at_5), np.mean(recall_at_5), np.mean(f1_at_5),np.mean(ja_at_5)


def main():
    if not os.path.exists(os.path.join("saved", model_name)):
        os.makedirs(os.path.join("saved", model_name))

    train_path = '../../data/DrugRec0716/data_train.pkl'
    dev_path = '../../data/DrugRec0716/data_eval.pkl'
    test_path = '../../data/DrugRec0716/data_test.pkl'
    #data_path = '../data/records_final.pkl'
    voc_path = '../../data/DrugRec0716/voc_final.pkl'
    device = torch.device('cuda:0')

    # data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['sym_voc'], voc['med_voc']

    data_train = dill.load(open(train_path, 'rb'))
    # eval_len = int(len(data[split_point:]) / 2)
    data_test = dill.load(open(test_path, 'rb'))
    data_eval = dill.load(open(dev_path, 'rb'))
    voc_size = (len(diag_voc["idx2word"])+1, len(pro_voc["idx2word"])+1, len(med_voc["idx2word"])+1)

    EPOCH = 40
    LR = 0.0002
    TEST = True

    model = Retain(voc_size, device=device)
    if TEST:
        model.load_state_dict(torch.load(open(os.path.join("saved", model_name, resume_name), 'rb')))

    model.to(device=device)
    print('parameters', get_n_params(model))


    optimizer = Adam(model.parameters(), lr=LR)
    best_ja = 0
    if TEST:
        eval(model, data_test, voc_size, 0)
    else:
        history = defaultdict(list)
        for epoch in range(EPOCH):
            loss_record = []
            start_time = time.time()
            model.train()
            num = 0
            for step, input1 in enumerate(data_train):
                # if num >= 10:
                #     break
                # else:
                #     num += 1

                # if len(input) < 2:
                #     continue
                input = [input1]
                loss = 0
                for i in range(0, len(input)):
                    target = np.zeros((1, voc_size[2])) # 1行voc_size[2]列的矩阵
                    target[:, input[i][2]] = 1

                    output_logits = model(input[:i+1])
                    loss += F.binary_cross_entropy_with_logits(output_logits, torch.FloatTensor(target).unsqueeze(1).to(device))
                    loss_record.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                llprint('\rTrain--Epoch: %d, Step: %d/%d' % (epoch, step, len(data_train)))

            ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, p_at_1, mrr, hit_at_5, precison_5, recall_5, f1_5,ja_at_5 = eval(model, data_eval, voc_size, epoch)
            history['ja'].append(ja)
            history['ddi_rate'].append(ddi_rate)
            history['avg_p'].append(avg_p)
            history['avg_r'].append(avg_r)
            history['avg_f1'].append(avg_f1)
            history['prauc'].append(prauc)
            history['p@1'].append(p_at_1)
            history['mrr'].append(mrr)
            history['hit@5'].append(hit_at_5)
            history['precision@5'].append(precison_5)
            history['recall@5'].append(recall_5)
            history['f1@5'].append(f1_5)
            history['ja@5'].append(ja_at_5)
            end_time = time.time()
            elapsed_time = (end_time - start_time) / 60
            llprint('\tEpoch: %d, Loss1: %.4f, One Epoch Time: %.2fm, Appro Left Time: %.2fh\n' % (epoch,
                                                                                                np.mean(loss_record),
                                                                                                elapsed_time,
                                                                                                elapsed_time * (
                                                                                                            EPOCH - epoch - 1)/60))
            if best_ja < ja:
                best_epoch = epoch
                best_ja = ja
                torch.save(model.state_dict(), open( os.path.join('saved', model_name, 'Epoch_%d_p@1_%.4f_ja_%.4f.model' % (epoch, p_at_1, ja)), 'wb'))
            print('')

        dill.dump(history, open(os.path.join('saved', model_name, 'history.pkl'), 'wb'))

        # test
        torch.save(model.state_dict(), open(
            os.path.join('saved', model_name, 'final.model'), 'wb'))


if __name__ == '__main__':
    main()