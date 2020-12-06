import pandas as pd
from collections import defaultdict, Counter
from itertools import combinations
from sklearn.feature_extraction import DictVectorizer
from tqdm import tqdm
data_path = r"C:\Dev\NLP_EX3\data\wikipedia.tinysample.trees.lemmatized.txt"
# data = pd.read_csv(data_path, sep='\t',
#                 names=['ID', 'FORM', 'LEMMA', 'CPOSTAG', 'POSTAG', 'FEATS', 'HEAD', 'DEPREL', 'PHEAD', 'PDEPREL'],
#                 header=None)
# data["Sentence"] = (data['ID'].diff()<0).cumsum()
# print(data.head(20))
# print(data.Sentence)


class Vectorizer(object):
    MIN_OCCUR = 75
    CONTEXT_LIMIT = 100

    def __init__(self, data_path, ):
        self.data = self.read_data(data_path)
        self.cm_sent = defaultdict(Counter)
        self.cm_win = defaultdict(Counter)
        self.lemma_count = self.data.LEMMA.value_counts()

        self.produce_sent_vec_matrices()
        self.dict_sent_vecs, self.sent_vecs = self.sentence_vectorizer()
        self.word_sen_vec_index = {word: idx for idx, word in enumerate(self.cm_sent.keys())}

        self.produce_sin_vec_matrices()
        self.dict_window_vecs, self.window_vecs = self.window_vectorizer()
        self.word_win_vec_index = {word: idx for idx, word in enumerate(self.cm_win.keys())}



    def read_data(self, data_path):
        df = pd.read_csv(data_path,
                         sep='\t',
                         names=['ID', 'FORM', 'LEMMA', 'CPOSTAG', 'POSTAG', 'FEATS', 'HEAD', 'DEPREL', 'PHEAD',
                                'PDEPREL'],
                         header=None)
        df['Sentence'] = (df['ID'].diff() < 0).cumsum()
        return df

    def count_words_in_sent(self, sent):
        for w1, w2 in combinations(sent, 2):
            if self.lemma_count[w1] >= self.CONTEXT_LIMIT:
                self.cm_sent[w1][w2] += 1
            if self.lemma_count[w2] >= self.CONTEXT_LIMIT:
                self.cm_sent[w2][w1] += 1

    def produce_sent_vec_matrices(self):
        for name, sen in tqdm(self.data.groupby("Sentence")):
            filterd_sent = self.filter_rare_words(sen.LEMMA)
            self.count_words_in_sent(filterd_sent)

    def filter_rare_words(self, words):
        return [word for word in words if self.lemma_count[word] >= self.MIN_OCCUR]

    def sentence_vectorizer(self):
        v = DictVectorizer(sparse=True)
        X = v.fit_transform([dict(v.most_common(self.CONTEXT_LIMIT)) for v in self.cm_sent.values()])
        return v, X

    def get_word_vec(self, word, vec_type):
        if vec_type == "sen":
            return self.sent_vecs[self.word_sen_vec_index[word]]
        elif vec_type == "window":
            return self.window_vecs[self.word_win_vec_index[word]]

    def get_inverse_sent_vec(self, word):
            return self.dict_sent_vecs.inverse_transform(self.sent_vecs[self.word_sen_vec_index[word]])

    def window_vectorizer(self):
        v = DictVectorizer(sparse=True)
        X = v.fit_transform([dict(v.most_common(self.CONTEXT_LIMIT)) for v in self.cm_win.values()])
        return v, X

    def produce_sin_vec_matrices(self):
        filtered_words = self.filter_rare_words(self.data.LEMMA)
        for w1back, w2back, pivot, w1front, w2front in tqdm(zip(filtered_words,
                                                                filtered_words[1:],
                                                                filtered_words[2:],
                                                                filtered_words[3:],
                                                                filtered_words[4:])):
            if self.lemma_count[pivot] >= self.CONTEXT_LIMIT:
                    self.cm_win[pivot][w1back] += 1
                    self.cm_win[pivot][w2back] += 1
                    self.cm_win[pivot][w1front] += 1
                    self.cm_win[pivot][w1front] += 1



vec = Vectorizer(data_path)
a=2