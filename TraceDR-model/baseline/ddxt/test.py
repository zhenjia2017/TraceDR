import time
import torch
import numpy as np
from network import Network
from dataset import load_dataset
from utils import mean, evaluate_ddx, evaluate_cls

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64
# vocab_size = 436
vocab_size = 21000
en_seq_len = 80
de_seq_len = 30
features = 128
heads = 4
layers = 6
# output_size = 54
# output_size = 5858
output_size = 17491
drop_rate = 0.1

print('Loading data & network ...')
_, test_loader = load_dataset(batch_size=batch_size, num_workers=0)

network = Network(vocab_size=vocab_size,
                  en_seq_len=en_seq_len,
                  de_seq_len=de_seq_len,
                  features=features,
                  heads=heads,
                  n_layer=layers,
                  output_size=output_size,
                  dropout_rate=drop_rate).to(device)

network.load_state_dict(torch.load('./weights_new/model_30.h5'))

print('Start testing ...')

# test
network.eval()
test_acc_ddx, test_acc_cls = [], []
tic = time.time()

np_true_ddx = []
np_pred_ddx = []

with torch.no_grad():
    for n, (en_in, de_in, de_out) in enumerate(test_loader):
        # Move data to the same device as the model
        en_in, de_in, de_out = en_in.to(device), de_in.to(device), de_out.to(device)
        # de_out = one_hot(de_out, output_size)

        # forward
        de_out_pred = network(en_input=en_in, de_input=de_in)

        # store
        np_true_ddx.append(de_out.detach().cpu().numpy())
        np_pred_ddx.append(torch.argmax(de_out_pred, dim=-1).detach().cpu().numpy())
        # np_true_cls.append(path.detach().cpu().numpy())
        # np_pred_cls.append(torch.argmax(path_pred, dim=-1).detach().cpu().numpy())

        # evaluate
        ddx_acc = evaluate_ddx(true=de_out, pred=de_out_pred)
        # cls_acc = evaluate_cls(true=path, pred=path_pred)
        test_acc_ddx.append(ddx_acc.item())
        # test_acc_cls.append(cls_acc.item())

test_acc_ddx = mean(test_acc_ddx) * 100
# test_acc_cls = mean(test_acc_cls) * 100
toc = time.time()

print(f'test ddx acc: {test_acc_ddx:.2f}%, eta: {toc - tic:.2}s')

np_true_ddx = np.concatenate(np_true_ddx, dtype=np.float32)
np_pred_ddx = np.concatenate(np_pred_ddx, dtype=np.float32)
# np_true_cls = np.concatenate(np_true_cls, dtype=np.float32)
# np_pred_cls = np.concatenate(np_pred_cls, dtype=np.float32)

print(np_true_ddx.shape)
print(np_pred_ddx.shape)
# print(np_true_cls.shape)
# print(np_pred_cls.shape)

# save file
np.save('results/true.npy', np_true_ddx)
np.save('results/pred.npy', np_pred_ddx)
# np.save('results/true_cls.npy', np_true_cls)
# np.save('results/pred_cls.npy', np_pred_cls)

print('All Done!')


