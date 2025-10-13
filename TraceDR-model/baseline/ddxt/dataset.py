import csv
import torch
import sys
from vocab import build_vocab
from preprocess import parse_patient
from torch.utils.data import Dataset, DataLoader
csv.field_size_limit(sys.maxsize)

class DDxDataset(Dataset):
    def __init__(self, filename):
        with open(filename, mode='r', encoding='utf-8') as f:
            self.loader = list(csv.DictReader(f))
        self.en_vocab, self.de_vocab = build_vocab()

    def __len__(self):
        return len(self.loader)

    def __getitem__(self, idx):
        en_input, de = parse_patient(self.loader[idx], en_max_len=80, de_max_len=30)
        en_input = list(map(lambda x: self.en_vocab.get(x, self.en_vocab['<unk>']), en_input.split(' ')))
        de = list(map(lambda x: self.de_vocab.get(x, self.de_vocab['<unk>']), de.split(' ')))
        de_input = de[0:-1]
        de_output = de[1:]
        # pathology = self.de_vocab.get(gt)
        # convert list to tensor
        en_input = torch.tensor(en_input)
        de_input = torch.tensor(de_input)
        de_output = torch.tensor(de_output)
        # pathology = torch.tensor(pathology)

        return en_input, de_input, de_output


def load_dataset(batch_size, num_workers):
    train_data = DDxDataset(filename='data/ddxt_train_data.csv')
    test_data = DDxDataset(filename='data/ddxt_test_data.csv')

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_loader = DataLoader(test_data,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)

    return train_loader, test_loader


if __name__ == '__main__':
    train_x, test_x = load_dataset(batch_size=256, num_workers=0)

    for en_in, de_in, de_out in train_x:
        print(en_in.shape)
        print(de_in.shape)
        print(de_out.shape)
        # print(path.shape)
        break

    for en_in, de_in, de_out in test_x:
        print(en_in.shape)
        print(de_in.shape)
        print(de_out.shape)
        # print(path.shape)
        break
