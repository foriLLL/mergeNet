
from datasets import load_dataset
import torch
from config import max_len
from datasets import DatasetDict
import datasets
load_path = "./data/raw_data/nodoc_noAcceptOneSide_noNewLine.json"  # raw data


def res_indices(conflict):
    '''给 conflict 加上 label 和 lines

    Args:
        conflict: 冲突，需要有 ours, theirs, resolve

    Returns:
        label: resolve 对应在 lines 里的行号（没包括<eos>
        lines: ours+theirs 的所有行
    '''
    code_lines = []
    if conflict['ours']:
        code_lines += conflict['ours'][:]
    if conflict['theirs']:
        code_lines += conflict['theirs'][:]
    indices = []
    if conflict['resolve']:
        for line in conflict['resolve']:
            for i in range(len(code_lines)):
                if line == code_lines[i]:
                    indices.append(i+1)  # 改成i+1则行号下标从1开始，len+1为<eos>，0为超出
                    break
    return {'label': indices, 'lines': code_lines}


def padding(example, max_len):
    '''给一个冲突样本加入填充

    <pad> 对应 0
    <eos> 对应超过lines的行数的数

    Args:
        max_len: a+b+<eos>的总行数不应超过 max_len，resolution+<eos> 也不应该超过max_len
        example: 一个样本，需要保证加入一个<eos>不会超过最长max_len
    '''
    valid_len = len(example['label'])   # resolved 的行数
    pad_lines = example['lines'] + ['<eos>']
    pad_label = example['label'] + \
        [len(example['lines']) + 1]  # len + 1 对应 <eos>

    for _ in range(max_len - len(example['lines']) - 1):  # 填充到max_len
        pad_lines += ['<pad>']
    for _ in range(max_len - valid_len - 1):
        pad_label += [0]

    # padding resolve
    pad_resolve = example['resolve'] + ['<line_eos>']
    for _ in range(max_len - len(example['resolve']) - 1):
        # 注意 resolution 没加 eos   为什么没加，我先加上
        pad_resolve = pad_resolve + ['<line_pad>']

    return {
        'resolve': pad_resolve,
        'lines': pad_lines,
        'label': pad_label,
        'valid_len': valid_len + 1  # resolved 行数加上 <eos>
    }


# start processing
dataset = load_dataset("json", data_files=load_path)

print('compute resolution indices')

label_dataset = dataset.map(res_indices, remove_columns=[
                            'ours', 'theirs', 'base'])
if not isinstance(label_dataset, datasets.DatasetDict):
    raise TypeError("label_dataset must be an instance of DatasetDict")


print(len(label_dataset['train']))

print('filter large conflicts')
filter_dataset = label_dataset.filter(lambda data: len(
    data['lines']) < max_len - 1 and len(data['label']) < max_len - 1)  # max_len - 1 因为要给stop token留位置
print(len(filter_dataset['train']))

print('padding')
padding_dataset = filter_dataset.map(lambda x: padding(x, max_len))
print(len(padding_dataset['train']))

# def conflict2EditSeq()        todo

# padding_dataset.save_to_disk('./output/deepmerge_dataset')  # type: ignore

# print("tokenizing")
# tokenized_dataset = padding_dataset.map(tokenize_func, remove_columns=['lines'])
# tokenized_dataset = padding_dataset.map(tokenize_func)
# print(len(tokenized_dataset['train']))

# tokenized_dataset.save_to_disk(dataset_path)
