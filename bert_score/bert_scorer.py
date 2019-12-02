import time
from collections import defaultdict

import torch
import torch.nn as nn

from transformers import AutoTokenizer

from bert_score.utils import (lang2model, model2layers, model_types,
                              cache_scibert, get_model, get_idf_dict,
                              bert_cos_score_idf)


class BertScorer(object):
    def __init__(self, model_type=None, num_layers=None, verbose=False,
                 idf=False, batch_size=64, nthreads=4, all_layers=False, lang=None,
                 return_hash=False):

        self.batch_size = batch_size
        self.nthreads = nthreads
        self.verbose = verbose
        self.idf = idf
        self.return_hash = return_hash
        self.all_layers = all_layers

        assert lang is not None or model_type is not None, \
            'Either lang or model_type should be specified'

        if model_type is None:
            lang = lang.lower()
            model_type = lang2model[lang]
        if num_layers is None:
            num_layers = model2layers[model_type]
        print(model_type)
        print("start initializing tokenizer")
        assert model_type in model_types
        if model_type.startswith('scibert'):
            tokenizer = AutoTokenizer.from_pretrained(cache_scibert(model_type))
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_type)

        print("start building model")
        model = get_model(model_type, num_layers, all_layers)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # device = 'cpu'
        model.to(device)

        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def score(self, cands, refs):
        if not self.idf:
            idf_dict = defaultdict(lambda: 1.)
            # set idf for [SEP] and [CLS] to 0
            idf_dict[self.tokenizer.sep_token_id] = 0
            idf_dict[self.tokenizer.cls_token_id] = 0
        else:
            if self.verbose:
                print('preparing IDF dict...')
            start = time.perf_counter()
            idf_dict = get_idf_dict(refs, self.tokenizer, nthreads=self.nthreads)
            if self.verbose:
                print('done in {:.2f} seconds'.format(time.perf_counter() - start))

        if self.verbose:
            print('calculating scores...')
        start = time.perf_counter()
        all_preds = bert_cos_score_idf(self.model, refs, cands, self.tokenizer, idf_dict,
                                       verbose=self.verbose, device=self.device,
                                       batch_size=self.batch_size, all_layers=self.all_layers)

        P = all_preds[..., 0].cpu()
        R = all_preds[..., 1].cpu()
        F1 = all_preds[..., 2].cpu()
        if self.verbose:
            time_diff = time.perf_counter() - start
            print(f'done in {time_diff:.2f} seconds, {len(refs) / time_diff:.2f} sentences/sec')

        return P, R, F1
