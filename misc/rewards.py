from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
import time
import misc.utils as utils
from collections import OrderedDict
import torch

import sys

sys.path.append("cider")
from pyciderevalcap.ciderD.ciderD import CiderD

sys.path.append("coco-caption")
from pycocoevalcap.bleu.bleu import Bleu

sys.path.append('bert_score')
from bert_scorer import BertScorer

CiderD_scorer = None
Bleu_scorer = None
Bert_scorer = None


# CiderD_scorer = CiderD(df='corpus')

def init_scorer(cached_tokens):
    global CiderD_scorer
    CiderD_scorer = CiderD_scorer or CiderD(df=cached_tokens)
    global Bleu_scorer
    Bleu_scorer = Bleu_scorer or Bleu(4)
    global Bert_scorer
    Bert_scorer = Bert_scorer or BertScorer(verbose=False, all_layers=False, lang='en')


def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()

def decode_sequence(ix_to_word, arr):
    out = ''
    for i in range(len(arr)):
        ix = arr[i]
        if ix > 0:
            out += ix_to_word[str(ix.item())] + ' '
        else:
            break
    return out.strip()

def get_self_critical_reward(greedy_res, data_gts, gen_result, opt):
    batch_size = gen_result.size(0)  # batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data_gts)
    ix_to_word = opt.ix_to_word

    res = OrderedDict()

    gen_result = gen_result.data.cpu().numpy()
    greedy_res = greedy_res.data.cpu().numpy()
    for i in range(batch_size):
        res[i] = [decode_sequence(ix_to_word, gen_result[i])]
    for i in range(batch_size):
        res[batch_size + i] = [decode_sequence(ix_to_word, greedy_res[i])]

    gts = OrderedDict()
    for i in range(len(data_gts)):
        gts[i] = [decode_sequence(ix_to_word, data_gts[i][j]) for j in range(len(data_gts[i]))]

    res_ = [{'image_id': i, 'caption': res[i]} for i in range(2 * batch_size)]
    res__ = {i: res[i] for i in range(2 * batch_size)}
    res_bert = [res[i][0] for i in range(2 * batch_size)]
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}
    gts_bert = [gts[i][random.randint(0, seq_per_img - 1)] for i in range(2 * batch_size)]

    if opt.cider_reward_weight > 0:
        _, cider_scores = CiderD_scorer.compute_score(gts, res_)
        print('Cider scores:', _)
    else:
        cider_scores = 0
    if opt.bleu_reward_weight > 0:
        _, bleu_scores = Bleu_scorer.compute_score(gts, res__)
        bleu_scores = np.array(bleu_scores[3])
        print('Bleu scores:', _[3])
    else:
        bleu_scores = 0
    if opt.bert_reward_weight > 0:
        P, R, F = Bert_scorer.score(res_bert, gts_bert)
        bert_scores = np.array(P)
        print('BERT scores:', P.mean().item())
    else:
        bert_scores = 0

    scores = opt.cider_reward_weight * cider_scores + \
             opt.bleu_reward_weight * bleu_scores + opt.bert_reward_weight * bert_scores

    scores = scores[:batch_size] - scores[batch_size:]

    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    return rewards
