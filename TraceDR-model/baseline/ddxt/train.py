import time
import torch
from network import Network
from dataset import load_dataset
from utils import mean, evaluate_ddx, save_history

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Loading data & network ...')
train_loader, test_loader = load_dataset(batch_size=batch_size, num_workers=0)

network = Network(vocab_size=vocab_size,
                  en_seq_len=en_seq_len,
                  de_seq_len=de_seq_len,
                  features=features,
                  heads=heads,
                  n_layer=layers,
                  output_size=output_size,
                  dropout_rate=drop_rate).to(device)

# network.load_state_dict(torch.load('./weights/model.h5'))


epochs = 30
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
print('Start training ...')
history = []

for epoch in range(1, epochs + 1):
    train_loss, train_acc_ddx, train_acc_cls = [], [], []

    tic = time.time()

    # for en_in, de_in, de_out in train_loader:
    #     print(en_in.shape, de_in.shape, de_out.shape)

    # train
    network.train()
    for en_in, de_in, de_out in train_loader:

        en_in, de_in, de_out = en_in.to(device), de_in.to(device), de_out.to(device)
        optimizer.zero_grad()

        # forward + loss + backward + optimize
        de_out_pred = network(en_input=en_in, de_input=de_in)
        de_out_pred = de_out_pred.to(device)
        # print(de_out_pred.shape)
        loss = loss_function(de_out_pred.permute(0, 2, 1), de_out)
        # loss2 = loss_function(path_pred, path)
        # loss = loss1 + loss2

        loss.backward()
        optimizer.step()

        # evaluate
        ddx_acc = evaluate_ddx(true=de_out, pred=de_out_pred)
        # cls_acc = evaluate_cls(true=path, pred=path_pred)
        train_loss.append(loss.item())
        train_acc_ddx.append(ddx_acc.item())
        # train_acc_cls.append(cls_acc.item())

    train_loss = mean(train_loss)
    train_acc_ddx = mean(train_acc_ddx) * 100
    # train_acc_cls = mean(train_acc_cls) * 100

    # test
    network.eval()
    test_acc_ddx, test_acc_cls = [], []
    with torch.no_grad():
        for en_in, de_in, de_out in test_loader:
            en_in, de_in, de_out = en_in.to(device), de_in.to(device), de_out.to(device)
            # de_out = one_hot(de_out, output_size)

            # forward
            de_out_pred = network(en_input=en_in, de_input=de_in)
            de_out_pred = de_out_pred.to(device)
            # evaluate
            ddx_acc = evaluate_ddx(true=de_out, pred=de_out_pred)
            # cls_acc = evaluate_cls(true=path, pred=path_pred)
            test_acc_ddx.append(ddx_acc.item())
            # test_acc_cls.append(cls_acc.item())

    test_acc_ddx = mean(test_acc_ddx) * 100
    # test_acc_cls = mean(test_acc_cls) * 100
    toc = time.time()

    history.append(f'Epoch: {epoch}/{epochs}, train loss: {train_loss:>6.4f}, train ddx acc: {train_acc_ddx:.2f}%, '
                   f'test ddx acc: {test_acc_ddx:.2f}%, '
                   f'eta: {toc - tic:.2}s')
    print(history[-1])

    torch.save(network.state_dict(), f'weights_new/model_{epoch}.h5')
    scheduler.step()

save_history('weights_new/history.csv', history, mode='w')
print('All Done!')
