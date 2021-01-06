import argparse
import re
import os
import logging
import abc
from abc import ABC
from typing import Union

from collections import defaultdict, Counter
from itertools import combinations
import pandas as pd
import numpy as np
from tqdm import tqdm
from stop_words import STOP_WORDS
import pickle as pk
from datetime import datetime
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize

class Vectorizer(ABC):
    MIN_OCCUR = 75
    CONTEXT_LIMIT = 100


    def __init__(self, data_path, use_cache=True):
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
        if os.path.exists(self.cached_path_dir) and use_cache:
            self.logger.info("Loading from cache. ")
            self.logger.info(f"The cache path is: {self.cached_path_dir}")

            self.logger.info("Not verifying hyper parameters setup!")
            self.data = pd.read_pickle(os.path.join(self.cached_path_dir, 'data.pk'))
            lemma_count = self.data.LEMMA.value_counts()
            self.lemma_count = lemma_count[lemma_count > self.MIN_OCCUR]
            with open(os.path.join(self.cached_path_dir, 'confusion_matrix.pk'), 'rb') as f:
                self.confusion_matrix = pk.load(f)
            self.logger.info("Finish loading from cache")

        else:
            self.logger.info("Start to process.")
            self.logger.info(f"The cache path would be {self.cached_path_dir}")
            self.logger.info("STARTING loading data...")
            self.data = self.read_data(data_path)
            self.logger.info("FINISH loading data!")
            lemma_count = self.data.LEMMA.value_counts()
            self.lemma_count = lemma_count[lemma_count > self.MIN_OCCUR]
            self.confusion_matrix = defaultdict(Counter)
            self.produce_matrices()
            self.logger.info("Dumping confusion matrix, data, lemma to cache")
            self.logger.info(f"Path to cache: {self.cached_path_dir}")
            os.mkdir(self.cached_path_dir)
            self.data.to_pickle(os.path.join(self.cached_path_dir, 'data.pk'))
            with open(os.path.join(self.cached_path_dir, 'confusion_matrix.pk'), 'wb') as f:
                pk.dump(self.confusion_matrix, f)

        self.attribute_count, self.all_events = self.get_attribute_count()

        self.dict_vectors, self.vectors = self.vectorizer()
        self.word_vec_index = {word: idx for idx, word in enumerate(self.confusion_matrix.keys())}
        self.index_vec_word = {idx: word for word, idx in self.word_vec_index.items()}
        self.total_lemma = self.lemma_count.sum()


    @staticmethod
    def read_data(data_path):
        df = pd.read_csv(data_path, sep='\t',
                         names=['ID', 'FORM', 'LEMMA', 'CPOSTAG', 'POSTAG', 'FEATS', 'HEAD', 'DEPREL', 'PHEAD',
                                'PDEPREL'], header=None, dtype=str, )
        df[["ID", "HEAD"]] = df[["ID", "HEAD"]].astype(int)
        df[['FORM', 'LEMMA', 'CPOSTAG', 'POSTAG', 'FEATS', 'DEPREL', 'PHEAD', 'PDEPREL']] = df[
            ['FORM', 'LEMMA', 'CPOSTAG', 'POSTAG', 'FEATS', 'DEPREL', 'PHEAD', 'PDEPREL']].astype(str)

        df['Sentence'] = (df['ID'].diff() < 0).cumsum()
        return df

    def filter_words(self, words: Union[pd.DataFrame, pd.Series], extra_words_tofilter=None):
        if extra_words_tofilter is None:
            extra_words_tofilter = set()
        # remove the extra words you wanna filter by removing it from the counter (and than it fails in the 'isin')
        words_filter = self.lemma_count[~self.lemma_count.index.isin(extra_words_tofilter)]
        words = words[words['LEMMA'].isin(words_filter[words_filter > self.MIN_OCCUR].index)]
        return words

    def vectorizer(self):
        dict_vector = DictVectorizer(sparse=True, )
        # using the most_common we limit the context words to the 100 most common ones for each v (a.k.a a word)
        data = [(word, dict(attributes.most_common(self.CONTEXT_LIMIT))) for word, attributes in self.confusion_matrix.items()]
        pmi_data = []
        for word, attributes in data:
            word_pmi_context = {}
            for context in attributes:
                word_pmi_context[context] = self.pmi(word, context)
            pmi_data.append(word_pmi_context)

        vectors = normalize(dict_vector.fit_transform(pmi_data), norm='l2')
        return dict_vector, vectors

    def get_inverse_vec(self, word):
        return self.dict_vectors.inverse_transform(self.vectors[self.word_vec_index[word]])

    def get_word_vec(self, word):
        return self.vectors[self.word_vec_index[word]]

    def efficient_algo(self, w_v):
        res = []
        w_v_indices_dict = {v: i for i, v in enumerate(w_v.indices)}
        for vec_i, other in enumerate(self.vectors):
            score = 0
            mutual_indices = np.intersect1d(other.indices,  w_v.indices)
            other_indices_dict = {v: i for i, v in enumerate(other.indices)}

            if len(mutual_indices) == 0:
                continue
            for mutual_idx in mutual_indices:
                w_v_i = w_v_indices_dict[mutual_idx]
                other_i = other_indices_dict[mutual_idx]
                score += w_v.data[w_v_i] * other.data[other_i]
            res.append((vec_i, score))

        return res


    def get_most_similar(self, word, top_n=20):
        w_v = self.vectors[self.word_vec_index[word]]
        sims = self.efficient_algo(w_v)
        sims.sort(key=lambda x: x[1], reverse=True)
        most_similar_words2 = sims[:top_n + 1]
        res = [self.index_vec_word[idx] for idx, _ in most_similar_words2]
        res.pop(0)
        return res

    def produce_matrices(self):
        pass

    def dump_count_words(self):
        f_name = "counts_words.txt"
        op_d = self.lemma_count[~self.lemma_count.index.isin(STOP_WORDS)].nlargest(50).to_dict()
        final_op = []

        for word, count in op_d.items():
            final_op.append(f"{word} {count}\n")
        with open(f_name, mode='w') as f:
            f.writelines(final_op)


    def get_attribute_count(self):
        res = defaultdict(int)
        all_attribute = 0
        for att_count_dict in self.confusion_matrix.values():
            for attribute, att_count in att_count_dict.items():
                res[attribute] += att_count
                all_attribute += att_count
        return res, all_attribute


    def count(self, w):
        if w in self.lemma_count:
            return self.lemma_count[w]
        else:
            return 0

    @abc.abstractmethod
    def count_attribute(self, att):
        pass

    def common_prob(self, w1, w2):
        neighbors_sum = sum(self.confusion_matrix[w1].values())
        return self.confusion_matrix[w1][w2] / neighbors_sum


    def word_probe(self, w):
        return self.count(w) / self.total_lemma

    def attribute_probe(self, att):
        return self.count_attribute(att) / self.all_events

    def common_probe(self, word, att):
        return self.confusion_matrix[word][att] / (self.all_events * self.total_lemma)


    def pmi(self, w1, att):
        common_prob = self.confusion_matrix[w1][att]
        c_w1 = self.count(w1)
        c_w2 = self.count_attribute(att)

        return np.log2(common_prob * self.all_events / (c_w1 * c_w2))

    def get_best_pmi(self, w, top_n=20):
        scores = [(self.dict_vectors.feature_names_[att], self.pmi(w, self.dict_vectors.feature_names_[att]))
                  for att in self.get_word_vec(w).indices]

        scores.sort(key=lambda x: x[1], reverse=True)
        scores = scores[:top_n]
        return [att for att, score in scores]



class SentenceVector(Vectorizer):
    def __init__(self, data_path, use_cache=True):
        super().__init__(data_path, use_cache)

    def count_words_in_sent(self, sent):
        for w1, w2 in combinations(sent, 2):
            if self.lemma_count[w1] >= self.CONTEXT_LIMIT and w1 not in STOP_WORDS:
                self.confusion_matrix[w1][w2] += 1
            if self.lemma_count[w2] >= self.CONTEXT_LIMIT and w2 not in STOP_WORDS:
                self.confusion_matrix[w2][w1] += 1

    def produce_matrices(self):
        data = self.filter_words(self.data, extra_words_tofilter=STOP_WORDS)
        data_clean = data.dropna(subset=['LEMMA'])
        for _, sen in tqdm(data_clean.groupby("Sentence")):
            self.count_words_in_sent(sen.LEMMA)

    def count_attribute(self, att):
        return self.count(att)


class WindowVector(Vectorizer):
    def __init__(self, data_path, use_cache=True):
        super().__init__(data_path, use_cache)

    def produce_matrices(self):
        filtered_words = self.filter_words(self.data, extra_words_tofilter=STOP_WORDS)
        filtered_words = filtered_words.LEMMA
        for i, (w1back, w2back, pivot, w1front, w2front) in tqdm(enumerate(
                zip(filtered_words, filtered_words[1:], filtered_words[2:], filtered_words[3:], filtered_words[4:]))):

            if self.lemma_count[pivot] >= self.CONTEXT_LIMIT:
                self.add_count_to_confusion_mat(pivot, w1back)
                self.add_count_to_confusion_mat(pivot, w2back)
                self.add_count_to_confusion_mat(pivot, w1front)
                self.add_count_to_confusion_mat(pivot, w1front)

    def add_count_to_confusion_mat(self, pivot, w):
        if w not in STOP_WORDS:
            self.confusion_matrix[pivot][w] += 1

    def count_attribute(self, att):
        return self.count(att)


class DependencyVector(Vectorizer):
    def __init__(self, data_path, use_cache=True):
        super().__init__(data_path, use_cache)

    PARENT_CON = "P"
    DAUGHTER_CON = "D"
    PREP_POS = "IN"

    def count_attribute(self, att):
        if att in self.attribute_count:
            return self.attribute_count[att]
        else:
            return 0

    def produce_matrices(self):
        filter_data = self.filter_words(self.data, extra_words_tofilter=STOP_WORDS)
        for _, sen in tqdm(filter_data.groupby("Sentence")):
            for i, r in sen.iterrows():
                self.update_daughters(sen, r)
                self.update_parent(sen, r)

    def update_parent(self, filtered_sen, r):
        parent = filtered_sen[filtered_sen["ID"] == r["HEAD"]]
        if not parent.empty:
            parent = parent.iloc[0]
            if parent.POSTAG == self.PREP_POS:
                grand_parent = filtered_sen[filtered_sen["ID"] == parent["HEAD"]]
                if grand_parent.empty:
                    return
                feature = self.create_feature(f"{parent.DEPREL}_{parent.LEMMA}", self.PARENT_CON,
                                              grand_parent.iloc[0].LEMMA)
            else:
                feature = self.create_feature(r.DEPREL, self.PARENT_CON, parent.LEMMA)

            self.confusion_matrix[r["LEMMA"]][feature] += 1

    def update_daughters(self, filtered_sen, target_word):
        daughters_connection = filtered_sen[filtered_sen["HEAD"] == target_word["ID"]]
        for i, r_d in daughters_connection.iterrows():
            if r_d.POSTAG == self.PREP_POS:
                gran_daughters_connection = filtered_sen[filtered_sen["HEAD"] == r_d["ID"]]
                noun_gran_daughter = gran_daughters_connection[
                    (gran_daughters_connection["POSTAG"] == "NN") | (gran_daughters_connection["POSTAG"] == "NNS")]
                if noun_gran_daughter.empty:
                    continue
                for j in range(len(noun_gran_daughter)):
                    noun_lemma = noun_gran_daughter.iloc[j].LEMMA
                    cur_feature = self.create_feature(f"{r_d.DEPREL}_{r_d.LEMMA}", self.DAUGHTER_CON, noun_lemma)
                    self.confusion_matrix[target_word.LEMMA][cur_feature] += 1
            else:
                cur_feature = self.create_feature(r_d.DEPREL, self.DAUGHTER_CON, r_d.LEMMA)
                self.confusion_matrix[target_word.LEMMA][cur_feature] += 1

    @staticmethod
    def create_feature(label, direction, connected_word):
        return f"{label}_{direction}_{connected_word}"

    def dump_context(self):
        f_name = 'counts_contexts_dep.txt'
        op_d = sorted(self.attribute_count.items(), key=lambda x: x[1], reverse=True)[:50]
        final_op = []
        for att, count in op_d:
            final_op.append(f"{att} {count}\n")
        with open(f_name, mode='w') as f:
            f.writelines(final_op)



def main():
    parser = argparse.ArgumentParser(description='Vectorizer program')
    parser.add_argument('file')
    parser.add_argument('-v', type=int, help='Vector type - 1: Sentence , 2: Window , 3: Dependency  ')
    parser.add_argument('--all',  action="store_true", help='Run all vec - for plotting ')
    args = parser.parse_args()

    if args.all:

        vec_dep = DependencyVector(args.file)
        vec_dep.dump_count_words()
        vec_dep.dump_context()

        vec_win = WindowVector(args.file)
        vec_sen = SentenceVector(args.file)
        with open('submition_files/top20.txt', mode='w') as f:
            for cur_w in 'car bus hospital hotel gun bomb horse fox table bowl guitar piano'.split():
                win = vec_win.get_most_similar(cur_w)
                sen = vec_sen.get_most_similar(cur_w)
                dep = vec_dep.get_most_similar(cur_w)
                f.write(f'{cur_w}\n')
                for w, s, d in zip(win, sen, dep):
                    f.write(f"{w} {s} {d}\n")
                f.write('**********\n')
                word_res = [{f'window': w, f'sentence':s , 'dependency': d } for w, s, d in zip(win, sen, dep)]
                print("                      ", cur_w)
                print(pd.DataFrame(word_res))
                print()


        with open('submition_files/context_top20.txt', mode='w') as f:
                for cur_w in 'car bus hospital hotel gun bomb horse fox table bowl guitar piano'.split():
                    win = vec_win.get_best_pmi(cur_w)
                    sen = vec_sen.get_best_pmi(cur_w)
                    dep = vec_dep.get_best_pmi(cur_w)
                    f.write(f'{cur_w}\n')
                    for w, s, d in zip(win, sen, dep):
                        f.write(f"{w} {s} {d}\n")
                    f.write('**********\n')
                    word_res = [{f'window': w, f'sentence':s , 'dependency': d } for w, s, d in zip(win, sen, dep)]
                    print("                      ", cur_w)
                    print(pd.DataFrame(word_res))
                    print()
        return

    if args.v == 1:
        vec = SentenceVector(args.file)
    elif args.v == 2:
        vec = WindowVector(args.file)
    elif args.v == 3:
        vec = DependencyVector(args.file)
    else:
        ValueError("Support vec - {1,2,3} see -help")
        return


if __name__ == '__main__':
    main()
