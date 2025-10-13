import pickle
from tqdm import tqdm
import torch
import argparse
import numpy as np
import dill
import time
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import os
import torch.nn.functional as F
from collections import defaultdict

from models2 import GAMENet
from util import llprint, multi_label_metric, ddi_rate_score, get_n_params, calculate_rank_metrics

torch.manual_seed(1203)
np.random.seed(1203)

model_name = 'GAMENet0717'
resume_name = 'saved/GAMENet0717/Epoch_37_p@1_0.0132_ja_0.0052.model'

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--eval', action='store_true', default=False, help="eval mode")
parser.add_argument('--model_name', type=str, default=model_name, help="model name")
parser.add_argument('--resume_path', type=str, default=resume_name, help='resume path')
parser.add_argument('--ddi', action='store_true', default=False, help="using ddi")

args = parser.parse_args()
model_name = args.model_name
resume_name = args.resume_path

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
    
def eval(model, data_eval, voc_size, epoch, ddi_adj):
    # evaluate
    print('')
    model.eval()
    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)] #创建5个变量
    p_at_1, mrr, hit_at_5 = [],[],[]
    precision_at_5, recall_at_5, f1_at_5 = [], [], []
    ja_at_5 = []
    case_study = defaultdict(dict) #可以直接嵌套赋值，而无需先检查键是否存在
    med_cnt = 0
    visit_cnt = 0
    num = 0
    for step, input1 in enumerate(data_eval):
        # if num >= 10:
        #     break
        # else:
        #     num += 1
        input = [input1]
        y_gt = []
        y_pred = []
        y_pred_prob = []
        y_pred_label = []
        for adm_idx, adm in enumerate(input):

            target_output1 = model(input[:adm_idx+1])

            y_gt_tmp = np.zeros(voc_size[2]+1)
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)
            drugs = torch.tensor(adm[2])
            target_output1 = F.sigmoid(target_output1).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output1)
            y_pred_tmp = target_output1.copy()
            y_pred_tmp[y_pred_tmp>=0.5] = 1
            y_pred_tmp[y_pred_tmp<0.5] = 0
            y_pred.append(y_pred_tmp)
            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)

        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))
        adm_p1, adm_mrr, amd_h5, adm_precison5, adm_recall5, adm_f5 =  calculate_rank_metrics(target_output1, drugs)
        adm_ja_at_5 = jaccard_at_k(np.array(y_gt), np.array(y_pred_prob), k=5)
        case_study[adm_ja] = {'ja': adm_ja, 'patient': input, 'y_label': y_pred_label}

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
    #ddi_rate = ddi_rate_score(smm_record, ddi_adj)
    ddi_rate = 0

    llprint('\tDDI Rate: %.4f, Jaccard: %.4f,  PRAUC: %.4f, AVG_PRC: %.4f, AVG_RECALL: %.4f, AVG_F1: %.4f\n' % (
        ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)
    ))
    llprint('\tp@1: %.4f, mrr: %.4f,  hit@5: %.4f, precision@5: %.4f, recall@5: %.4f, f1@5: %.4f, jaccard@5: %.4f\n' % (
        np.mean(p_at_1), np.mean(mrr), np.mean(hit_at_5), np.mean(precision_at_5), np.mean(recall_at_5), np.mean(f1_at_5),np.mean(ja_at_5)
    ))
    dill.dump(obj=smm_record, file=open('../data/gamenet_records.pkl', 'wb'))
    dill.dump(case_study, open(os.path.join('saved', model_name, 'case_study.pkl'), 'wb'))

    # print('avg med', med_cnt / visit_cnt)

    return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), np.mean(p_at_1), np.mean(mrr), np.mean(hit_at_5), np.mean(precision_at_5), np.mean(recall_at_5), np.mean(f1_at_5),np.mean(ja_at_5)


def main():
    if not os.path.exists(os.path.join("saved", model_name)):
        os.makedirs(os.path.join("saved", model_name))

    train_path = '../data/DrugRec0716/data_train.pkl'
    dev_path = '../data/DrugRec0716/data_eval.pkl'
    test_path = '../data/DrugRec0716/data_test.pkl'
    #data_path = '../data/4sdata/records_final.pkl'
    voc_path = '../data/DrugRec0716/voc_final.pkl'

    ehr_adj_path = '../data/DrugRec0716/ehr_adj_final.pkl'
    ddi_adj_path = '../data/DrugRec0716/ddi_A_final.pkl'
    device = torch.device('cuda:0')
    print(device)

    ehr_adj = dill.load(open(ehr_adj_path, 'rb'))
    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
    print(type(ehr_adj))
    print(type(ddi_adj))
    #data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['sym_voc'], voc['med_voc']

    #split_point = int(len(data) * 2 / 3)
    data_train = dill.load(open(train_path, 'rb'))
    #eval_len = int(len(data[split_point:]) / 2)
    data_test = dill.load(open(test_path, 'rb'))
    data_eval = dill.load(open(dev_path, 'rb'))

    EPOCH = 40
    LR = 0.0002
    TEST = args.eval
    Neg_Loss = args.ddi
    DDI_IN_MEM = args.ddi
    TARGET_DDI = 0.05
    T = 0.5
    decay_weight = 0.85

    voc_size = (len(diag_voc["idx2word"]), len(pro_voc["idx2word"]), len(med_voc["idx2word"]))
    model = GAMENet(voc_size, ehr_adj, ddi_adj, emb_dim=64, device=device, ddi_in_memory=DDI_IN_MEM)
    if TEST:
        model.load_state_dict(torch.load(open(resume_name, 'rb')))
    model.to(device=device)

    print('parameters', get_n_params(model))
    optimizer = Adam(list(model.parameters()), lr=LR)

    if TEST:
        eval(model, data_test, voc_size,0, ddi_adj)
    else:
        history = defaultdict(list)
        best_epoch = 0
        best_ja = 0
        for epoch in range(EPOCH):
            loss_record1 = []
            start_time = time.time()
            model.train()
            prediction_loss_cnt = 0
            neg_loss_cnt = 0
            num = 0 
            for step, input1 in enumerate(data_train):
                # if num >= 10:
                #     break
                # else:
                #     num += 1
                input = [input1]
                for idx, adm in enumerate(input):
                    seq_input = input[:idx+1]
                    # print(input)
                    # print(seq_input)
                    loss1_target = np.zeros((1, voc_size[2]+1))
                    loss1_target[:, adm[2]] = 1
                    loss3_target = np.full((1, voc_size[2]+1), -1)
                    #print(input)
                    for idx, item in enumerate(adm[2]):
                        loss3_target[0][idx] = item

                    target_output1, batch_neg_loss = model(seq_input)

                    loss1 = F.binary_cross_entropy_with_logits(target_output1, torch.FloatTensor(loss1_target).to(device))
                    loss3 = F.multilabel_margin_loss(F.sigmoid(target_output1), torch.LongTensor(loss3_target).to(device))
                    if Neg_Loss:
                        target_output1 = F.sigmoid(target_output1).detach().cpu().numpy()[0]
                        target_output1[target_output1 >= 0.5] = 1
                        target_output1[target_output1 < 0.5] = 0
                        y_label = np.where(target_output1 == 1)[0]
                        current_ddi_rate = ddi_rate_score([[y_label]])
                        if current_ddi_rate <= TARGET_DDI:
                            loss = 0.9 * loss1 + 0.01 * loss3
                            prediction_loss_cnt += 1
                        else:
                            rnd = np.exp((TARGET_DDI - current_ddi_rate)/T)
                            if np.random.rand(1) < rnd:
                                loss = batch_neg_loss
                                neg_loss_cnt += 1
                            else:
                                loss = 0.9 * loss1 + 0.01 * loss3
                                prediction_loss_cnt += 1
                    else:
                        loss = 0.9 * loss1 + 0.01 * loss3

                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

                    loss_record1.append(loss.item())

                llprint('\rTrain--Epoch: %d, Step: %d/%d, L_p cnt: %d, L_neg cnt: %d' % (epoch, step, len(data_train), prediction_loss_cnt, neg_loss_cnt))
            # annealing
            T *= decay_weight

            ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, p_at_1, mrr, hit_at_5, precison_5, recall_5, f1_5,ja_at_5 = eval(model, data_eval, voc_size, epoch, ddi_adj)

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
            llprint('\tEpoch: %d, Loss: %.4f, One Epoch Time: %.2fm, Appro Left Time: %.2fh\n' % (epoch,
                                                                                                np.mean(loss_record1),
                                                                                                elapsed_time,
                                                                                                elapsed_time * (
                                                                                                            EPOCH - epoch - 1)/60))

            #torch.save(model.state_dict(), open( os.path.join('saved', model_name, 'Epoch_%d_p@1_%.4f_DDI_%.4f.model' % (epoch, p_at_1, ddi_rate)), 'wb'))
            print('')
            #if epoch != 0 and best_ja < avg_p:
            if best_ja < ja:
                best_epoch = epoch
                best_ja = ja
                torch.save(model.state_dict(), open( os.path.join('saved', model_name, 'Epoch_%d_p@1_%.4f_ja_%.4f.model' % (epoch, p_at_1, ja)), 'wb'))


        dill.dump(history, open(os.path.join('saved', model_name, 'history.pkl'), 'wb'))

        # test
        torch.save(model.state_dict(), open(
            os.path.join('saved', model_name, 'final.model'), 'wb'))

        print('best_epoch:', best_epoch)


if __name__ == '__main__':
    main()
    # with open('../data/ehr_adj_final.pkl','rb') as f1:
    #     data = pickle.load(f1)
    # print('hello')

