import torch
import torch.nn as nn
import numpy as np
import dill
import time
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import os
import torch.nn.functional as F
import random
from collections import defaultdict

import sys
sys.path.append("..")
from models2 import Leap
from util import llprint, sequence_metric, sequence_output_process, ddi_rate_score, get_n_params, calculate_rank_metrics

torch.manual_seed(1203)

model_name = 'Leap0718'
resume_name = 'Epoch_29_p@1_0.0270_ja_0.0138.model'

def jaccard_at_k(y_gt, pred_top_k, k=5):
    """计算Jaccard@k指标"""
    true_drugs = y_gt.cpu().numpy().astype(int)
    #true_set = set(true_drugs)
    if len(pred_top_k) < 5:
        pred_top_k += [0] * (5 - len(pred_top_k))
    #target = y_gt
    #target = set(np.where(y_gt[b] == 1)[0])
   # pred_top_k = set(np.argsort(y_pred[b])[::-1][:k])
    inter = set(pred_top_k) & set(true_drugs)
    union = set(pred_top_k) | set(true_drugs)
    jaccard_score = 0 if len(union) == 0 else len(inter) / len(union)
        
    return jaccard_score

def eval(model, data_eval, voc_size, epoch):
    # evaluate
    print('')
    model.eval()

    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    p_at_1, mrr, hit_at_5 = [], [], []
    precision_at_5, recall_at_5, f1_at_5 = [], [], []
    ja_at_5 = []
    records = []
    med_cnt = 0
    visit_cnt = 0
    num = 0
    for step, input1 in enumerate(data_eval):
        # if num >= 10:
        #     break
        # else:
        #     num += 1
        y_gt = []
        y_pred = []
        y_pred_prob = []
        y_pred_label = []
        input = [input1]
        for adm in input:
            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)
            drugs = torch.tensor(adm[2])
            output_logits = model(adm)
            output_logits = output_logits.detach().cpu().numpy()
            #print(output_logits)
            out_list, sorted_predict = sequence_output_process(output_logits, [voc_size[2], voc_size[2]+1])
            #print(sorted(sorted_predict))
            #print(out_list)
            #print(sorted_predict)
            y_pred_label.append(sorted_predict)
            y_pred_prob.append(np.mean(output_logits[:, :-2], axis=0))

            y_pred_tmp = np.zeros(voc_size[2])
            y_pred_tmp[out_list] = 1
            y_pred.append(y_pred_tmp)
            visit_cnt += 1
            med_cnt += len(sorted_predict)
        records.append(y_pred_label)

        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = sequence_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob), np.array(y_pred_label))
        adm_p1, adm_mrr, amd_h5, adm_precison5, adm_recall5, adm_f5 = calculate_rank_metrics(sorted_predict, drugs, isleap = True)
        adm_ja_at_5 = jaccard_at_k(drugs, sorted_predict, k=5)
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

    # ddi rate
    # ddi_rate = ddi_rate_score(records)
    ddi_rate = 0
    llprint('\tDDI Rate: %.4f, Jaccard: %.4f,  PRAUC: %.4f, AVG_PRC: %.4f, AVG_RECALL: %.4f, AVG_F1: %.4f\n' % (
        ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)
    ))
    llprint('\tp@1: %.4f, mrr: %.4f,  hit@5: %.4f, precision@5: %.4f, recall@5: %.4f, f1@5: %.4f, ja@5: %.4f\n' % (
        np.mean(p_at_1), np.mean(mrr), np.mean(hit_at_5), np.mean(precision_at_5), np.mean(recall_at_5),
        np.mean(f1_at_5), np.mean(ja_at_5)
    ))
    print('avg med', med_cnt / visit_cnt)
    return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), np.mean(p_at_1), np.mean(mrr), np.mean(hit_at_5), np.mean(precision_at_5), np.mean(recall_at_5), np.mean(f1_at_5), np.mean(ja_at_5)

def main():
    if not os.path.exists(os.path.join("saved", model_name)):
        os.makedirs(os.path.join("saved", model_name))

    train_path = '../../data/DrugRec0716/data_train.pkl'
    dev_path = '../../data/DrugRec0716/data_eval.pkl'
    test_path = '../../data/DrugRec0716/data_test.pkl'
    # data_path = '../data/records_final.pkl'
    voc_path = '../../data/DrugRec0716/voc_final.pkl'
    device = torch.device('cuda:0')

    #data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['sym_voc'], voc['med_voc']

    data_train = dill.load(open(train_path, 'rb'))
    # eval_len = int(len(data[split_point:]) / 2)
    data_test = dill.load(open(test_path, 'rb'))
    data_eval = dill.load(open(dev_path, 'rb'))
    voc_size = (len(diag_voc["idx2word"])+1, len(pro_voc["idx2word"])+1, len(med_voc["idx2word"])+1)
    print(voc_size[2])
    EPOCH = 30
    LR = 0.0002
    TEST = False
    END_TOKEN = voc_size[2] + 1

    model = Leap(voc_size, device=device)
    if TEST:
        model.load_state_dict(torch.load(open(os.path.join("saved", model_name, resume_name), 'rb')))
        # pass

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
                input = [input1]
                for adm in input:
                    loss_target = adm[2] + [END_TOKEN]
                    output_logits = model(adm)
                    loss = F.cross_entropy(output_logits, torch.LongTensor(loss_target).to(device))

                    loss_record.append(loss.item())

                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
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
                torch.save(model.state_dict(), open(
                    os.path.join('saved', model_name, 'Epoch_%d_p@1_%.4f_ja_%.4f.model' % (epoch, p_at_1, ja)), 'wb'))
            print('')

        dill.dump(history, open(os.path.join('saved', model_name, 'history.pkl'), 'wb'))
        # test
        torch.save(model.state_dict(), open(
            os.path.join('saved', model_name, 'final.model'), 'wb'))

def fine_tune(fine_tune_name=''):
    data_path = '../../data/records_final.pkl'
    voc_path = '../../data/voc_final.pkl'
    device = torch.device('cuda:0')

    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    ddi_A = dill.load(open('../../data/ddi_A_final.pkl', 'rb'))

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    # data_eval = data[split_point+eval_len:]
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    model = Leap(voc_size, device=device)
    model.load_state_dict(torch.load(open(os.path.join("saved", model_name, fine_tune_name), 'rb')))
    model.to(device)

    EPOCH = 30
    LR = 0.0001
    END_TOKEN = voc_size[2] + 1

    optimizer = Adam(model.parameters(), lr=LR)
    ddi_rate_record = []
    for epoch in range(1):
        loss_record = []
        start_time = time.time()
        random_train_set = [ random.choice(data_train) for i in range(len(data_train))]
        for step, input in enumerate(random_train_set):
            model.train()
            K_flag = False
            for adm in input:
                target = adm[2]
                output_logits = model(adm)
                out_list, sorted_predict = sequence_output_process(output_logits.detach().cpu().numpy(), [voc_size[2], voc_size[2] + 1])

                inter = set(out_list) & set(target)
                union = set(out_list) | set(target)
                jaccard = 0 if union == 0 else len(inter) / len(union)
                K = 0
                for i in out_list:
                    if K == 1:
                        K_flag = True
                        break
                    for j in out_list:
                        if ddi_A[i][j] == 1:
                            K = 1
                            break

                loss = -jaccard * K * torch.mean(F.log_softmax(output_logits, dim=-1))


                loss_record.append(loss.item())

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            llprint('\rTrain--Epoch: %d, Step: %d/%d' % (epoch, step, len(data_train)))

            if K_flag:
                ddi_rate, ja, prauc, avg_p, avg_r, avg_f1 = eval(model, data_test, voc_size, epoch)


                end_time = time.time()
                elapsed_time = (end_time - start_time) / 60
                llprint('\tEpoch: %d, Loss1: %.4f, One Epoch Time: %.2fm, Appro Left Time: %.2fh\n' % (epoch,
                                                                                               np.mean(loss_record),
                                                                                               elapsed_time,
                                                                                               elapsed_time * (
                                                                                                       EPOCH - epoch - 1) / 60))

                torch.save(model.state_dict(),
                   open(os.path.join('saved', model_name, 'fine_Epoch_%d_JA_%.4f_DDI_%.4f.model' % (epoch, ja, ddi_rate)),
                        'wb'))
                print('')

    # test
    torch.save(model.state_dict(), open(
        os.path.join('saved', model_name, 'final.model'), 'wb'))



if __name__ == '__main__':
    main()
    #fine_tune(fine_tune_name='Epoch_26_JA_0.4465_DDI_0.0723.model')