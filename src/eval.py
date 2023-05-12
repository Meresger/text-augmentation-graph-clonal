#!/usr/bin/env python3

import os, sys, json, time
from src.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction, NgramInst
from src.amr_graph import AMRGraph


def get_amr_ngrams(path, stat_save_path=None):
    data = []
    amr = AMRGraph(path)
    amr.revert_of_edges()
    ngrams = amr.extract_ngrams(3, multi_roots=True) # dict(list(tuple))
    data.append(NgramInst(ngram=ngrams, length=len(amr.edges)))
    if stat_save_path:
        print(len(amr), len(ngrams[1]), len(ngrams[2]), len(ngrams[3]), file=f)
    if stat_save_path:
        f.close()
    return data

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('python this-script ans-file ref-file')
        sys.exit(0)
    print('loading ...')
    hypothesis = get_amr_ngrams(sys.argv[1])
    references = [[x] for x in get_amr_ngrams(sys.argv[2])]
    smoofunc = getattr(SmoothingFunction(), 'method3')
    print('evaluating ...')
    st = time.time()
    if len(sys.argv) == 4:
        n = int(sys.argv[3])
        weights = (1.0/n, )*n
    else:
        weights = (0.34, 0.33, 0.34)
    print(corpus_bleu(references, hypothesis, weights=weights, smoothing_function=smoofunc, auto_reweigh=True))
    print('time:', time.time()-st, 'secs')
