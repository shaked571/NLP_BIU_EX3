import pandas as pd
import re
import os
from collections import defaultdict, Counter
from itertools import combinations
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from stop_words import STOP_WORDS
import pickle as pk
from datetime import datetime


# data_path = r"C:\Dev\NLP_EX3\data\wikipedia.tinysample.trees.lemmatized"
data_path = r"C:\Dev\NLP_EX3\data\wikipedia.sample.trees.lemmatized"

import logging


class Vectorizer(object):
    MIN_OCCUR = 75
    # MIN_OCCUR = 1

    CONTEXT_LIMIT = 100
    # CONTEXT_LIMIT = 5



    def __init__(self, data_path, use_cach=True):
        self.logger = logging.getLogger(self.__class__.__name__)
        if not os.path.exists('.cache'):
            os.mkdir('.cache')
        if not os.path.exists('.logs'):
            os.mkdir('.logs')
        current_time = datetime.now().strftime("%H%M%S")
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                            handlers=[logging.FileHandler(os.path.join('.logs', f"nlp_ex3_{current_time}.log")),
                                      logging.StreamHandler()])


        f_name = re.split(r'[\\/]', data_path)[-1]
        self.cache_f_name = f"{f_name}.{self.__class__.__name__}"
        self.cached_path_dir = os.path.join('.cache', self.cache_f_name)
        if os.path.exists(self.cached_path_dir) and use_cach:
            self.logger.info("Loading from cache. ")
            self.logger.info("***Not verifying hyper parameters setup!***")
            self.data = pd.read_pickle(os.path.join(self.cached_path_dir, 'data.pk'))
            self.lemma_count = self.data.LEMMA.value_counts()
            with open(os.path.join(self.cached_path_dir, 'confusion_matrix.pk'), 'rb') as f:
                self.confusion_matrix = pk.load(f)
            self.logger.info("Finish loading from cache")

        else:
            self.logger.info("STARTING loading data...")
            self.data = self.read_data(data_path)
            self.logger.info("FINISH loading data!")
            self.lemma_count = self.data.LEMMA.value_counts()
            self.confusion_matrix = defaultdict(Counter)
            self.produce_matrices()
            self.logger.info("Dumping confusion matrix, data, lemma to cache")
            self.logger.info(f"Path to cahch: {self.cached_path_dir}")
            os.mkdir(self.cached_path_dir)
            self.data.to_pickle(os.path.join(self.cached_path_dir, 'data.pk'))
            with open(os.path.join(self.cached_path_dir, 'confusion_matrix.pk'), 'wb') as f:
                pk.dump(self.confusion_matrix, f)

        self.dict_vectors, self.vectors = self.vectorizer()
        self.word_vec_index = {word: idx for idx, word in enumerate(self.confusion_matrix.keys())}

    def read_data(self, data_path):
        df = pd.read_csv(data_path,
                         sep='\t',
                         names=['ID', 'FORM', 'LEMMA', 'CPOSTAG', 'POSTAG', 'FEATS', 'HEAD', 'DEPREL', 'PHEAD',
                                'PDEPREL'],
                         header=None, dtype=str,)
        df[["ID", "HEAD"]] = df[["ID", "HEAD"]].astype(int)
        df[['FORM', 'LEMMA', 'CPOSTAG', 'POSTAG', 'FEATS','DEPREL', 'PHEAD','PDEPREL']] = \
            df[['FORM', 'LEMMA', 'CPOSTAG', 'POSTAG', 'FEATS','DEPREL', 'PHEAD','PDEPREL']].astype(str)

        df['Sentence'] = (df['ID'].diff() < 0).cumsum()
        return df

    def filter_words(self, words: pd.Series, extra_words_tofilter=None):
        if extra_words_tofilter is None:
            extra_words_tofilter = set()
        # remove the extra words you wanna filter by removing it from the counter (and tham it fails in th isin)
        words_filter = self.lemma_count[~self.lemma_count.index.isin(extra_words_tofilter)]
        output = words[words.isin(words_filter[words_filter > self.MIN_OCCUR].index)]
        return output

    def vectorizer(self):
        v = DictVectorizer(sparse=True)
        # using the most_common we limit the context words to the 100 most common ones for each v (a.k.a a word)
        X = v.fit_transform([dict(v.most_common(self.CONTEXT_LIMIT)) for v in self.confusion_matrix.values()])

        return v, X

    def get_inverse_vec(self, word):
        return self.dict_vectors.inverse_transform(self.vectors[self.word_vec_index[word]])

    def get_word_vec(self, word):
            return self.vectors[self.word_vec_index[word]]

    def get_most_similar(self, word, top_n=10):
        w_v = self.vectors[self.word_vec_index[word]]
        sims = self.vectors.dot(w_v)
        most_similar_ids = sims.argspty()[-1:-top_n:-1]
        return most_similar_ids

    def produce_matrices(self):
        pass

    def dump_count_words(self):
        f_name = "counts_words.txt"
        op_d = self.lemma_count.head(50).to_dict()
        final_op = []
        for word, count in zip(op_d.keys(), op_d.values()):
            final_op.append(f"{word} {count}\n")
        with open(f_name, mode='w') as f:
            f.writelines(final_op)

    def dump_context(self):
        f_name = "counts_contexts_dep.txt"


class SentenceVector(Vectorizer):
    def __init__(self, data_path, use_cache=True):
        super().__init__(data_path, use_cache)

    def count_words_in_sent(self, sent):
        for w1, w2 in combinations(sent, 2):
            if self.lemma_count[w1] >= self.CONTEXT_LIMIT:
                self.confusion_matrix[w1][w2] += 1
            if self.lemma_count[w2] >= self.CONTEXT_LIMIT:
                self.confusion_matrix[w2][w1] += 1

    def produce_matrices(self):
        for _, sen in tqdm(self.data.groupby("Sentence")):
            filterd_sent = self.filter_words(sen.LEMMA)
            self.count_words_in_sent(filterd_sent)


class WindowVector(Vectorizer):
    def __init__(self, data_path, use_cache=True):
        super().__init__(data_path, use_cache)

    def produce_matrices(self):
        #TODO verify if not need to calculate function word vectors if
        # needed, we will iterate agagin but wwe will count only the STOP WORDS
        filtered_words = self.filter_words(self.data.LEMMA)
        for i, (w1back, w2back, pivot, w1front, w2front) in tqdm(enumerate(zip(filtered_words,
                                                                               filtered_words[1:],
                                                                               filtered_words[2:],
                                                                               filtered_words[3:],
                                                                               filtered_words[4:]))):

            if self.lemma_count[pivot] >= self.CONTEXT_LIMIT:
                    self.count_word(pivot, w1back)
                    self.count_word(pivot, w2back)
                    self.count_word(pivot, w1front)
                    self.count_word(pivot, w1front)

    def count_word(self, pivot, w):
        if w not in STOP_WORDS:
            self.confusion_matrix[pivot][w] += 1


class DependecyVector(Vectorizer):
    def __init__(self, data_path, use_cache=True):
        super().__init__(data_path, use_cache)

    PARENT_CON = "P"
    DAUGHTER_CON = "D"
    PREP_POS = "IN"

    def produce_matrices(self):
        for _, sen in tqdm(self.data.groupby("Sentence")):
            to_filter = self.filter_words(sen)
            filtered_sen = sen.loc[to_filter.LEMMA.dropna().index]
            for i, r in filtered_sen.iterrows():
                self.update_daughters(filtered_sen, r)
                self.update_parent(filtered_sen, r)

    def update_parent(self, filtered_sen, r):
        parent = filtered_sen[filtered_sen["ID"] == r["HEAD"]]
        if not parent.empty:
            parent = parent.iloc[0]
            if parent.POSTAG == self.PREP_POS:
                grand_parent = filtered_sen[filtered_sen["ID"] == parent["HEAD"]]
                if grand_parent.empty:
                    return
                feature = self.create_feature(f"{parent.DEPREL}_{parent.LEMMA}", self.PARENT_CON, grand_parent.iloc[0].LEMMA)
            else:
                feature = self.create_feature(r.DEPREL, self.PARENT_CON, parent.LEMMA)

            self.confusion_matrix[r["LEMMA"]][feature] += 1

    def update_daughters(self, filtered_sen, target_word):
        daughters_connection = filtered_sen[filtered_sen["HEAD"] == target_word["ID"]]
        for i, r_d in daughters_connection.iterrows():
            if r_d.POSTAG == self.PREP_POS:
                gran_daughters_connection = filtered_sen[filtered_sen["HEAD"] == r_d["ID"]]
                noun_gran_daughter = gran_daughters_connection[(gran_daughters_connection["POSTAG"] == "NN") | (gran_daughters_connection["POSTAG"] == "NNS")]
                if noun_gran_daughter.empty:
                    continue
                if len(noun_gran_daughter) > 1:
                    self.logger.error(f"DUDE u were wrong! \n Lemma prep: {r_d.LEMMA}")
                    raise Exception(f"DUDE u were wrong! \n Lemma prep: {r_d.LEMMA}")

                noun_lemma = noun_gran_daughter.iloc[0].LEMMA
                cur_feature = self.create_feature(f"{r_d.DEPREL}_{r_d.LEMMA}", self.DAUGHTER_CON, noun_lemma)
            else:
                cur_feature = self.create_feature(r_d.DEPREL, self.DAUGHTER_CON, r_d.LEMMA)
            self.confusion_matrix[target_word.LEMMA][cur_feature] += 1

    @staticmethod
    def create_feature(label, direction, connected_word):
        return f"{label}_{direction}_{connected_word}"





# vec = SentenceVector(data_path)
vec = WindowVector(data_path)
# vec = DependecyVector(data_path)
vec.dump_count_words()
vec.logger.info(vec.get_most_similar("boat"))
a=2