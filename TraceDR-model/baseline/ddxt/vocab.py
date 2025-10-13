import csv
import sys
from read_utils import *
import pickle
csv.field_size_limit(sys.maxsize)


def scan_duplicate(x):
    x_ = set(list(x.keys()))
    if len(x) == len(x_):
        print('No duplicates found.')
    else:
        print('List contains duplicates')


def build_vocab():
    specials = ['<pad>', '<bos>', '<eos>', '<sep>', '<unk>']
    group, symptom, diagnose, med_history, allergen, on_medicine = read_info()
    input_vocab = specials + read_age() + read_sex() + group + symptom + diagnose + med_history + allergen + on_medicine
    output_vocab = specials + read_drug()

    input_vocab = {key: value for value, key in enumerate(input_vocab)}
    output_vocab = {key: value for value, key in enumerate(output_vocab)}

    return input_vocab, output_vocab


def save_vocab():
    # 构建词汇表
    input_vocab, output_vocab = build_vocab()

    # 检查是否有重复的键
    scan_duplicate(input_vocab)
    scan_duplicate(output_vocab)

    # 保存字典到文件
    with open('results/input_vocab.pkl', 'wb') as f:
        pickle.dump(input_vocab, f)

    with open('results/output_vocab.pkl', 'wb') as f:
        pickle.dump(output_vocab, f)

    print('Vocabularies have been saved.')


if __name__ == '__main__':
    save_vocab()
    # 加载字典
    with open('results/input_vocab.pkl', 'rb') as f:
        in_vocab = pickle.load(f)

    with open('results/output_vocab.pkl', 'rb') as f:
        out_vocab = pickle.load(f)
    # in_vocab, out_vocab = build_vocab()

    # print(in_vocab)
    # print(out_vocab)
    print(len(in_vocab))
    print(len(out_vocab))
    print('')

    # 19867
    # 17491

    # test
    line = '<bos> age_15-29 <sep> F <sep> douleurxx_carac lancinante_/_choc_électrique <eos> <pad>'.split(' ')
    # line = '<bos> Bronchite RGO Possible_NSTEMI_/_STEMI Angine_instable Péricardite Anémie Angine_stable Syndrome_de_Boerhaave <eos> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>'.split(' ')
    s2i = list(map(lambda x: in_vocab.get(x, in_vocab['<unk>']), line))
    print(s2i)

    line = '<bos> Bronchite RGO Possible_NSTEMI_/_STEMI Angine_instable <eos> <pad> <pad>'.split(' ')
    s2i = list(map(lambda x: out_vocab.get(x, out_vocab['<unk>']), line))
    print(s2i)
