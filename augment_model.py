import spacy
import numpy as np
import random
import re
import amrlib
import penman
import logging
import pandas as pd
import nltk
from collections import Counter
from nltk.corpus import wordnet
from gensim.models import KeyedVectors
import tensorflow_hub as hub
import random
from nltk.corpus import wordnet
from penman import transform
from penman.models import amr
import sys

# Download NLTK packages
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load Word2Vec word embeddings model
model_get_words = KeyedVectors.load_word2vec_format('../data/word2vec/GoogleNews-vectors-negative300.bin', binary=True)

# Load Universal Sentence Encoder model
model_4 = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

# Load AMR models
stog = amrlib.load_stog_model(device="cuda:0", disable_progress=True, model_dir="../data/model_stog/model_stog/")
gtos = amrlib.load_gtos_model(device="cuda:0", disable_progress=True, use_tense=False, model_dir="../data/model_gtos")

# Load Spacy language model
nlp = spacy.load("en_core_web_lg")

# Set up logging
logging.basicConfig(level=logging.CRITICAL)

# Define AMR roles and arguments
Roles = [
    ":accompanier", ":age", ":beneficiary", ":concession", ":condition", ":consist-of", ":degree", ":destination",
    ":direction", ":domain", ":duration", ":example", ":extent", ":citizen", ":frequency", ":instrument", ":li",
    ":location", ":country", ":manner", ":medium", ":mod", ":mode", ":city", ":name", ":ord", ":part", ":path",
    ":polarity", ":polite", ":poss", ":purpose", ":quant", ":range", ":scale", ":source", ":subevent", ":time",
    ":topic", ":unit", ":value", ":wiki", ":calendar", ":century", ":day", ":dayperiod", ":decade", ":era", ":month",
    ":quarter", ":season", ":timezone", ":weekday", ":year", ":year2", ":op1", ":op2", ":op3", ":op4", ":op5", ":op6",
    ":op7"
]
constant = [i.split(":")[1] for i in Roles]
ARGS = [":ARG0", ":ARG1", ":ARG2", ":ARG3", ":ARG4"]


class WordVectors:
    def __init__(self):
        self.model_get_words = model_get_words
        self.model_4 = model_4

    def embed(self, input):
        return self.model_4(input)

    def clean_word(self, s):
        # Remove non-alphabetic characters from the word
        s = ''.join(filter(lambda x: x.isalpha(), s))
        return s

    def get_distance_between_words(self, word1, word2):
        messages_ = [word1, word2]
        message_embeddings_ = self.embed(messages_)
        v = np.dot(message_embeddings_[0], message_embeddings_[1])
        return round(v, 1)

    def return_sim_words(self, word):
        try:
            # Get similar words from the Word2Vec model
            words = [i[0].lower() for i in self.model_get_words.most_similar(word, topn=25)]
            dist_word = [i for i in words if self.get_distance_between_words(i, word) >= 0.5]
            return self.clean_word(random.choice(dist_word))
        except (IndexError, KeyError) as e:
            print(f'Error: {e}')
            return word


class AMRModels:
    def __init__(self):
        self.stog = stog
        self.gtos = gtos

    def get_encode(self, graph):
        g = penman.decode(graph[0])
        g2 = penman.Graph(g.triples)
        return penman.encode(g2)

    def get_single_amr(self, x):
        try:
            graph = self.stog.parse_sents([x])
            amr = self.get_encode(graph)
            return amr
        except Exception as e:
            print(x)
            print(f'Error: {e}')

    def get_text(self, AMR):
        sents, _ = self.gtos.generate([AMR])
        return sents[0]


class AMRMutation:
    def __init__(self):
        self.Roles = Roles
        self.ARGS = ARGS
        self.WordVectors = WordVectors()

    def clean_string(self, s):
        # Remove non-alphabetic characters from the string
        s = ''.join(filter(lambda x: x.isalpha(), s))
        return s

    def change_instance_based_on_v(self, v, new_v, rr):
        for i in rr:
            new_in = i[0]
            new_target = i[2]
            if new_in == v:
                new_in = new_v
            elif new_target == v:
                new_target = new_v
            rr[rr.index(i)] = (new_in, i[1], new_target)

    def create_var(self, new_v, s):
        if new_v in s:
            v = new_v + '1'
            while v in s:
                v = v[:-1] + str(int(v[-1]) + 1)
            return v
        else:
            return new_v

    def alter_instances_2(self, triples, var, rr):
        try:
            clean_v = self.clean_string(triples[2])
            new_v = self.WordVectors.return_sim_words(clean_v)
            new_t = (self.create_var(new_v[0], var), ":instance", new_v + "-01")
            rr[rr.index(triples)] = new_t
            self.change_instance_based_on_v(triples[0], self.create_var(new_v[0], var), rr)
        except IndexError:
            pass

    def incoming_edge_count(self, node, pgraph):
        count = 0
        for edge in pgraph.edges():
            if edge[1] == node:
                count += 1
        return count

    def mutation_rel(self, AMR):
        g = penman.decode(AMR)
        edges = [(i) for i in g.triples if ':instance' not in i]
        for i in edges[:int(random.uniform(0, len(edges)))]:
            if i[1] not in self.Roles:
                g.triples[g.triples.index(i)] = (i[0], random.choice(self.ARGS), i[2])
        G = penman.Graph(g.triples)
        return penman.encode(G)

    def mutation_top(self, AMR):
        pgraph = penman.decode(AMR)
        candidate_tops = pgraph.variables()
        candidate_tops.remove(pgraph.top)
        candidate_tops = [v for v in candidate_tops if self.incoming_edge_count(v, pgraph) == 0]
        new_tops = [pgraph.top] + candidate_tops
        new_graphs = [penman.encode(pgraph, top=t) for t in new_tops]
        amrs = random.choice(new_graphs)
        t = penman.parse(amrs)
        t2 = transform.canonicalize_roles(t, model=amr.model)
        f = penman.format(t2)
        g = penman.decode(f)
        G = penman.Graph(g.triples)
        return penman.encode(G)

    def concept_mutation(self, AMR):
        g = penman.decode(AMR)
        In = [(i) for i in g.triples if ':instance' in i]
        select = random.choice(In)
        self.alter_instances_2(select, g.variables(), g.triples)
        G = penman.Graph(g.triples)
        return penman.encode(G)

    def alter_instances_inject(self, triples, var, rr, word):
        if triples[2] not in constant:
            try:
                clean_v = self.clean_string(triples[2])
                new_v = word
                new_t = (self.create_var(new_v[0], var), ":instance", new_v + "-01")
                rr[rr.index(triples)] = new_t
                self.change_instance_based_on_v(triples[0], self.create_var(new_v[0], var), rr)
            except IndexError:
                pass

    def inject_word(self, word, amr):
        g = penman.decode(amr)
        In = [(i) for i in g.triples if ':instance' in i]
        for i in In:
            v = self.WordVectors.get_distance_between_words(word, self.clean_string(i[2]))
            if v >= 0.5:
                self.alter_instances_inject(i, g.variables(), g.triples, word)
            else:
                pass
        G = penman.Graph(g.triples)
        return penman.encode(G)

    def inject_constant(self, word, amr):
        g = penman.decode(amr)
        In = [(i.target) for i in g.attributes()]
        for i in In:
            if i == '2016':
                for t in (g.triples):
                    if t[2] == i:
                        new_t = (t[0], t[1], '2022')
                        g.triples[g.triples.index(t)] = new_t
            v = self.WordVectors.get_distance_between_words(word, self.clean_string(i))
            if v >= 0.5:
                for t in (g.triples):
                    if i == t[2]:
                        new_t = (t[0], t[1], word)
                        g.triples[g.triples.index(t)] = new_t
            else:
                pass
        G = penman.Graph(g.triples)
        return penman.encode(G)

    def constant_mutation(self, amr):
        g = penman.decode(amr)
        In = [(i.target) for i in g.attributes()]
        for i in In:
            v = self.clean_string(i)
            for t in (g.triples):
                if i == t[2]:
                    new_t = (t[0], t[1], self.WordVectors.return_sim_words(v))
                    g.triples[g.triples.index(t)] = new_t
            else:
                pass
        G = penman.Graph(g.triples)
        return penman.encode(G)

from cleantext import clean
import re

class CleanText:
    def clean_text(self, x):
        """
        Cleans the given text using the cleantext library.

        Args:
            x (str): The text to be cleaned.

        Returns:
            str: The cleaned text.
        """
        return clean(x,
                     fix_unicode=True,
                     to_ascii=True,
                     lower=True,
                     no_line_breaks=True,
                     no_urls=True,
                     no_emails=True,
                     no_phone_numbers=True,
                     no_numbers=False,
                     no_digits=False,
                     no_currency_symbols=False,
                     no_punct=False,
                     replace_with_punct="",
                     replace_with_url="<URL>",
                     replace_with_email="<EMAIL>",
                     replace_with_phone_number="<PHONE>",
                     replace_with_number="<NUMBER>",
                     replace_with_digit="0",
                     replace_with_currency_symbol="<CUR>",
                     lang="en"
                     )

    def clean_urls(self, s):
        """
        Removes URLs from the given text using regular expressions.

        Args:
            s (str): The text to remove URLs from.

        Returns:
            str: The text with URLs removed.
        """
        s1 = re.sub('http://\S+|https://\S+', '', s)
        s2 = re.sub('http[s]?://\S+', '', s1)
        s3 = re.sub(r"http\S+", "", s2)
        return s3

    def clean_this(self, s):
        """
        Replaces "##" with a space in the given text.

        Args:
            s (str): The text to replace "##" with a space.

        Returns:
            str: The text with "##" replaced by a space.
        """
        return s.replace("##", " ")

    def clean_all_text(self, text):
        """
        Cleans the given text by applying multiple cleaning operations.

        Args:
            text (str): The text to be cleaned.

        Returns:
            str: The cleaned text.
        """
        text = self.clean_text(text)
        text = self.clean_urls(text)
        text = self.clean_this(text)
        return text

sys.path.append('/content/drive/MyDrive/src/')

from bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction, NgramInst
from amr_graph import AMRGraph

import os
import sys
import json
import time

def get_amr_ngrams(n):
    """
    Extracts n-grams from an AMR graph.

    Args:
        n (int): The number of tokens in each n-gram.

    Returns:
        list: A list of NgramInst objects containing the n-grams and graph length.
    """
    data = []
    amr = AMRGraph(n)
    amr.revert_of_edges()
    amr.extract_ngrams(3, multi_roots=True)
    ngrams = amr.extract_ngrams(3, multi_roots=True)
    data.append(NgramInst(ngram=ngrams, length=len(amr.edges)))
    return data

def compare(x, y):
    """
    Compares two AMR graphs using the BLEU score.

    Args:
        x (int): The number of tokens in n-grams for the hypothesis.
        y (int): The number of tokens in n-grams for the reference.

    Returns:
        float: The BLEU score.
    """
    hypothesis = get_amr_ngrams(x)
    references = [[x] for x in get_amr_ngrams(y)]
    smoofunc = getattr(SmoothingFunction(), 'method3')
    weights = (0.34, 0.33, 0.34)
    return corpus_bleu(references, hypothesis, weights=weights, smoothing_function=smoofunc, auto_reweigh=True)

