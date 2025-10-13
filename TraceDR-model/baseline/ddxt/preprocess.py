import ast


def parse_age(x):  
    if x <= 4:
        return 'age_1-4'
    elif x <= 14:
        return 'age_5-14'
    elif x <= 29:
        return 'age_15-29'
    elif x <= 44:
        return 'age_30-44'
    elif x <= 59:
        return 'age_45-59'
    elif x <= 74:
        return 'age_60-74'
    else:
        return 'age_above75'


def parse_ddx(x):
    # sort ddx based on most likely to the least likely
    x = sorted(x, key=lambda inputs: -inputs[1])
    # keep only the names and replace space with underscore
    x = [key.replace(' ', '_') for key, _ in x]
    return x


# def parse_evidences(x):
#     # separating categorical evidences
#
#
#
# def parse_pathology(x):
#     return x.replace(' ', '_')

def parse_info(x):
    x = x.split('、')
    # print(x)
    # x = [item.replace('_@_', ' ') for item in list]
    return x


# def pad_sequence(x, max_len):
#     n = max_len - len(x.split(' '))
#     x = x + ' ' + ' '.join(['<pad>'] * n)
#     return x

# def pad_sequence(x, max_len):
#     # 将输入字符串分割成单词列表
#     words = x.split(' ')
#     # 确保eos标记在截断的序列中
#     if len(words) > max_len:
#         # 如果序列长度超过最大长度，保留eos标记并截断前面的内容
#         words = words[-(max_len-1):]  # 保留max_len-1个单词，最后一个位置留给eos
#     # 添加填充
#     n = max_len - len(words)
#     words.extend(['<pad>'] * n)
#     # 将单词列表重新拼接成字符串
#     return ' '.join(words)


def pad_sequence(x, max_len):
    words = x.split(' ')
    if len(words) > max_len:
        words = words[:max_len-1]
        words.append('<eos>')
    n = max_len - len(words)
    words.extend(['<pad>'] * n)
    return ' '.join(words)


def parse_patient(x, en_max_len, de_max_len):
    age = int(x['age'])
    # ddx = ast.literal_eval(x['DIFFERENTIAL_DIAGNOSIS'])
    sex = x['gender']
    # pathology = x['PATHOLOGY']
    # evidences = ast.literal_eval(x['EVIDENCES'])
    # init_evidence = x['INITIAL_EVIDENCE']
    group = x['group']
    symptom = x['symptom']
    diagnose = x['diagnosis']
    med_history = x['antecedents']
    allergen = x['allergen']
    on_medicine = x['on_medicine']
    drug = x['answer']

    age = parse_age(age)
    # ddx = parse_ddx(ddx)
    # evidences = parse_evidences(evidences)
    # pathology = parse_pathology(pathology)
    symptom = parse_info(symptom)
    diagnose = parse_info(diagnose)
    med_history = parse_info(med_history)
    allergen = parse_info(allergen)
    drug = parse_info(drug)

    encoder_input = ' '.join(['<bos>', age, '<sep>', sex, '<sep>', group, '<sep>', *symptom, '<sep>', *diagnose, '<sep>', *med_history, '<sep>', *allergen, '<sep>', *on_medicine, '<eos>'])
    # lenOfEnO = len(encoder_input.split(' '))
    encoder_input = pad_sequence(encoder_input, max_len=en_max_len)

    decoder_input = ' '.join(['<bos>', *drug, '<eos>'])
    # lenOfDnO = len(decoder_input.split(' '))
    decoder_input = pad_sequence(decoder_input, max_len=de_max_len)



    return encoder_input, decoder_input


if __name__ == '__main__':
    import csv

    filename = 'data/new_test_data.csv'

    with open(filename, mode='r', encoding='utf-8') as f:
        loader = list(csv.DictReader(f))
    en_input, de_input = parse_patient(loader[3], en_max_len=80, de_max_len=20)
    print(en_input)
    print(de_input)
    # print(gt)
    # len_e, len_d = [], []
    # for item in loader:
        
    #     en_input, de_input, le, ld = parse_patient(item, en_max_len=80, de_max_len=20)
    #     len_e.append(le)
    #     len_d.append(ld)
    # # print(max(len_e))
    # print(sum(len_e)/len(len_e))


    # max_index = max(range(len(len_d)), key=lambda i: len_d[i])
    # max_value = len_d[max_index]

    # print("最大值:", max_value)
    # print("最大值索引:", max_index)

    # en_input, de_input, _, _ = parse_patient(loader[max_index], en_max_len=80, de_max_len=20)
    # print(de_input)
