import os
import sys
import json
import collections
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

import numpy as np
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def load_txt(path):
    with open(path, encoding='UTF-8', errors='ignore') as f:
        data = [i.strip() for i in f.readlines() if len(i) > 0]
    return data


def save_txt(data, path):
    with open(path, 'w', encoding='UTF-8') as f:
        f.write(data)


def load_json(path):
    with open(path, 'r', encoding='UTF_8') as f:
        return json.load(f)


def save_json(data, path, indent=0):
    with open(path, 'w', encoding='UTF-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def tokenize_by_bert(path):
    data = load_txt(path)

    res = []
    for seq in data:
        # tokens = tokenizer.tokenize(seq)
        # ids = tokenizer.convert_tokens_to_ids(tokens)
        ids = tokenizer.encode(seq)
        seq = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        res.append(seq)
    #save_txt("\n".join(res), "../data_daily/bert-tgt-test.txt")
    return res


def calc_diversity(hyp):
    # based on Yizhe Zhang's code
    tokens = [0.0, 0.0]
    types = [collections.defaultdict(int), collections.defaultdict(int)]
    for line in hyp:
        for n in range(2):
            for idx in range(len(line) - n):
                ngram = ' '.join(line[idx:idx + n + 1])
                types[n][ngram] = 1
                tokens[n] += 1
    div1 = len(types[0].keys()) / tokens[0]
    div2 = len(types[1].keys()) / tokens[1]
    return [div1, div2]


def calc_entropy(hyps, n_lines=None):
    # based on Yizhe Zhang's code
    etp_score = [0.0, 0.0, 0.0, 0.0]
    counter = [collections.defaultdict(int), collections.defaultdict(int), collections.defaultdict(int),
               collections.defaultdict(int)]
    for line in hyps:
        for n in range(4):
            for idx in range(len(line) - n):
                ngram = ' '.join(line[idx:idx + n + 1])
                counter[n][ngram] += 1

    for n in range(4):
        total = sum(counter[n].values())
        for v in counter[n].values():
            etp_score[n] += - v / total * (np.log(v) - np.log(total))

    return etp_score


def sta_freq(train_path):
    data = load_txt(train_path)
    vocab = collections.Counter()

    for line in data:
        seq = tokenizer.tokenize(line)
        vocab.update(seq)
    vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    return vocab
    # save_json(vocab, out_vocab_path)


def eval_freq(train_path, hyps, refs, data_type, freq_threshold, low_freq=True):
    # vocab_path = os.path.join(VOCAB_DIR, data_type + "_vocab.json")
    # vocab = load_json(vocab_path)
    vocab = load_json("E:/git/mine/AdaLab/result/train_tgt_vocab/ost_vocab.json")
    # vocab = sta_freq(train_path)
    freq_vocab = {token for token, freq in vocab if freq < freq_threshold} if low_freq \
        else {token for token, freq in vocab if freq > freq_threshold}
    hyps_cnt, hyps_freq_cnt = 0, 0
    refs_cnt, refs_freq_cnt = 0, 0
    for hyp, ref in zip(hyps, refs):
        hyp = tokenizer.tokenize(hyp)
        ref = tokenizer.tokenize(ref)
        hyps_cnt += len(hyp)
        refs_cnt += len(ref)
        hyps_freq_cnt += len([token for token in hyp if token in freq_vocab])
        refs_freq_cnt += len([token for token in ref if token in freq_vocab])

    return hyps_freq_cnt / hyps_cnt, refs_freq_cnt / refs_cnt


def eval(train_path, golden_str, infer_str, data_type, freq_threshold, report=False):
    res_dict = {}
    # [[word,...,word],...,[word,...,word]]
    infer = [line.strip().split() for line in infer_str]
    avg_len = sum([len(x) for x in infer]) / len(infer)
    res_dict["avg_len"] = round(avg_len, 2)
    if report:
        print("avg_len", round(avg_len, 2))

    # [[[word,...,word]],...,[[word,...,word]]]
    golden = [[line.strip().split()] for line in golden_str]
    avg_len_gold = sum([len(x[0]) for x in golden]) / len(golden)
    res_dict["avg_len_gold"] = round(avg_len_gold, 2)
    if report:
        print("avg_len_gold", round(avg_len_gold, 2))

    # eval freq
    temp_res = eval_freq(train_path, infer_str, golden_str,
                         data_type,
                         freq_threshold=freq_threshold,
                         low_freq=True)
    res_dict["resp_freq_ratio"], res_dict["ref_freq_ratio"] = [100 * x for x in temp_res]
    if report:
        print("infer freq ratio: ", round(res_dict["resp_freq_ratio"], 2))
        print("gold freq ratio: ", round(res_dict["ref_freq_ratio"], 2))

    # eval bleu
    nltk_bleu = []
    chencherry = SmoothingFunction()
    for i in range(4):
        weights = [1 / (i + 1)] * (i + 1)
        nltk_bleu.append(
            round(100 * corpus_bleu(
                golden, infer, weights=weights, smoothing_function=chencherry.method1), 2))
    res_dict["BLEU"] = nltk_bleu
    if report:
        print('BLEU', nltk_bleu)

    # eval dist
    distinct = [round(x * 100, 2) for x in calc_diversity(infer)]
    res_dict["dist"] = distinct
    if report:
        print('distinct', distinct)

    gold_distinct = [round(x * 100, 2) for x in calc_diversity([x[0] for x in golden])]
    res_dict["ref_dist"] = gold_distinct
    if report:
        print('human distinct', gold_distinct)

    # eval ent
    ent = calc_entropy(infer)
    res_dict["ent"] = [round(x, 2) for x in ent]
    if report:
        print("ent", [round(x, 2) for x in ent])

    return res_dict


if __name__ == '__main__':
    print("make sure BLEU score implementation copied from NLTK 3.4.3")

    train_path = os.path.join(sys.argv[1], "data_daily/tgt-train.txt")
    golden_path = os.path.join(sys.argv[1], "data_daily/tgt-test.txt")
    infer_path = sys.argv[2]

    golden_str = tokenize_by_bert(golden_path)
    infer_str = load_txt(infer_path)

    data_type = "daily"
    eval(train_path, golden_str, infer_str, data_type, freq_threshold=100, report=True)
