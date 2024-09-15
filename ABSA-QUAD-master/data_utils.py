# -*- coding: utf-8 -*-

# This script contains all data transformation and reading

import random
from torch.utils.data import Dataset
import jieba
senttag2word = {'POS': 'positive', 'NEG': 'negative', 'NEU': 'neutral'}
senttag2opinion = {'POS': 'great', 'NEG': 'bad', 'NEU': 'ok'}
sentword2opinion = {'positive': 'great', 'negative': 'bad', 'neutral': 'ok'}

aspect_cate_list = ['location general',
                    'food prices',
                    'food quality',
                    'food general',
                    'ambience general',
                    'service general',
                    'restaurant prices',
                    'drinks prices',
                    'restaurant miscellaneous',
                    'drinks quality',
                    'drinks style_options',
                    'restaurant general',
                    'food style_options']


def read_line_examples_from_file(data_path, silence):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    # 首先，明确文件的数据结构，是：   句子####标签
    # sents和labels见上边绿色的定义。sents就是句子列表，列表的每一项还是列表，是一个句子的分词列表
    # labels是标注列表，列表的每一个元素是四元组的列表（所以其实也是列表的列表）
    sents, labels = [], []
    with open(data_path, 'r', encoding='UTF-8') as fp:
        words, labels = [], []
        for line in fp:
            line = line.strip()
            if line != '':
                words, tuples = line.split('####')
                sents.append(words.split())
                labels.append(eval(tuples))
    if silence:
        print(f"Total examples = {len(sents)}")
    return sents, labels


def get_para_aste_targets(sents, labels):
    targets = []
    for i, label in enumerate(labels):
        all_tri_sentences = []
        for tri in label:
            # a is an aspect term
            if len(tri[0]) == 1:
                a = sents[i][tri[0][0]]
            else:
                start_idx, end_idx = tri[0][0], tri[0][-1]
                a = ' '.join(sents[i][start_idx:end_idx+1])

            # b is an opinion term
            if len(tri[1]) == 1:
                b = sents[i][tri[1][0]]
            else:
                start_idx, end_idx = tri[1][0], tri[1][-1]
                b = ' '.join(sents[i][start_idx:end_idx+1])

            # c is the sentiment polarity
            c = senttag2opinion[tri[2]]           # 'POS' -> 'good'

            one_tri = f"It is {c} because {a} is {b}"
            all_tri_sentences.append(one_tri)
        targets.append(' [SSEP] '.join(all_tri_sentences))
    return targets


def get_para_tasd_targets(sents, labels):

    targets = []
    for label in labels:
        all_tri_sentences = []
        for triplet in label:
            at, ac, sp = triplet

            man_ot = sentword2opinion[sp]   # 'positive' -> 'great'

            if at == 'NULL':
                at = 'it'
            one_tri = f"{ac} is {man_ot} because {at} is {man_ot}"
            all_tri_sentences.append(one_tri)

        target = ' [SSEP] '.join(all_tri_sentences)
        targets.append(target)
    return targets


def get_para_asqp_targets(sents, labels):
    """
    Obtain the target sentence under the paraphrase paradigm
    """
    targets = []
    for label in labels:  # 一个label有好几个四元组
        all_quad_sentences = []
        for quad in label:  # 一个quad是一个四元组
            at, ac, sp, ot = quad

            man_ot = sentword2opinion[sp]  # 'POS' -> 'good'    

            if at == 'NULL':  # for implicit aspect term
                at = 'it'

            one_quad_sentence = f"{ac} is {man_ot} because {at} is {ot}"
            all_quad_sentences.append(one_quad_sentence)  # 这里是所有四元组对应的句子组成的（有好几个句子）

        target = ' [SSEP] '.join(all_quad_sentences)
        targets.append(target)
    return targets


def get_transformed_io(data_path, data_dir):
    """
    The main function to transform input & target according to the task
    """
    # labels的元素是四元组列表（一个元素可能有好几个四元组）
    sents, labels = read_line_examples_from_file(data_path)  # 返回句子列表（分词过后的）和标注列表

    # the input is just the raw sentence
    inputs = [s.copy() for s in sents]  # input直接copy

    task = 'asqp'   # target，也是一个句子，是由四元组转换来的，具体方法在论文中有介绍，a is p because t is o，就这样类似的方法
    if task == 'aste':
        targets = get_para_aste_targets(sents, labels)
    elif task == 'tasd':
        targets = get_para_tasd_targets(sents, labels)
    elif task == 'asqp':
        targets = get_para_asqp_targets(sents, labels)
    else:
        raise NotImplementedError

    return inputs, targets


class ABSADataset(Dataset):
    def __init__(self, tokenizer, data_dir, data_type, max_len=128):
        # './data/rest16/train.txt'
        self.data_path = f'data/{data_dir}/{data_type}.txt' # 是在这里读取的数据，路径固定在了data文件夹下面，且根据类型读取。
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.data_dir = data_dir

        self.inputs = []
        self.targets = []

        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    # 获取数据集中索引为index的样本的输入编码（一个序列，序列中一个单词一个编码），输出编码，以及掩码（长度为128，有词的为1，没有的为0）
    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, 
                "target_ids": target_ids, "target_mask": target_mask}

    # 把输入输出转换为编码序列。首先看get_transformed_io()这个函数，会对输入和目标进行初步处理。
    def _build_examples(self):
        # 把inputs和targets经过转换，两个都是list，inputs是句子的list，元素是存放分词的list。
        # targets是经过四元组转换过后的句子的list（这里可能一个元素有多个句子，因为input中一个句子可能对应多个四元组，那么转换过后的target就会有多个句子，用SSEP分隔）
        inputs, targets = get_transformed_io(self.data_path, self.data_dir)

        for i in range(len(inputs)):
            # change input and target to two strings
            # input应该是分词形式存放的句子
            input = ' '.join(inputs[i])
            target = targets[i]

            tokenized_input = self.tokenizer.batch_encode_plus(
              [input], max_length=self.max_len, padding="max_length",
              truncation=True, return_tensors="pt"
            )
            tokenized_target = self.tokenizer.batch_encode_plus(
              [target], max_length=self.max_len, padding="max_length",
              truncation=True, return_tensors="pt"
            )

            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)
