import pandas as pd
from collections import defaultdict, Counter
from itertools import combinations
from sklearn.feature_extraction import DictVectorizer
# data_path = r"C:\Dev\NLP_EX3\data\wikipedia.tinysample.trees.lemmatized.txt"
# data = pd.read_csv(data_path, sep='\t',
#                 names=['ID', 'FORM', 'LEMMA', 'CPOSTAG', 'POSTAG', 'FEATS', 'HEAD', 'DEPREL', 'PHEAD', 'PDEPREL'],
#                 header=None)
#
#
# print(data.head(20))


class Vectorizer(object):
    MIN_OCCUR = 75
    CONTEXT_LIMIT = 100

    def __init__(self, data_path, ):
        self.data = self.read_data(data_path)
        self.cm_sent = defaultdict(Counter)
        self.cm_win = defaultdict(Counter)
        self.lemma_count = self.data.LEMMA.value_counts()
        self.produce_matrices()
        self.dict_vec, self.sent_vecs = self.sentence_vectorizer()
        self.word_index = {word: idx for idx, word in enumerate(self.cm_sent.keys())}

    def read_data(self, data_path):
        return pd.read_csv(data_path,
                           sep='\t',
                           names=['ID', 'FORM', 'LEMMA', 'CPOSTAG', 'POSTAG', 'FEATS', 'HEAD', 'DEPREL', 'PHEAD','PDEPREL'],
                           header=None)

    def count_words_in_sent(self, sent):
        for w1, w2 in combinations(sent, 2):
            if self.lemma_count[w1] >= self.CONTEXT_LIMIT:
                self.cm_sent[w1][w2] += 1
            if self.lemma_count[w2] >= self.CONTEXT_LIMIT:
                self.cm_sent[w2][w1] += 1

    def produce_matrices(self):
        cur_sent = []
        for i in range(len(self.data)):
            if data.iloc[i].ID == 1:
                self.count_words_in_sent(cur_sent)
                # count_words_in_win2(confusion_matrix_win, cur_sent, data.LEMMA.value_counts())

                cur_sent = []
            if self.lemma_count[self.data.iloc[i].LEMMA] >= self.MIN_OCCUR:
                cur_sent.append(self.data.iloc[i].LEMMA)

    def sentence_vectorizer(self):
        v = DictVectorizer(sparse=True)
        X = v.fit_transform([dict(v.most_common(self.CONTEXT_LIMIT)) for v in self.cm_sent.values()])
        return v, X

    def get_word_vec(self, word, vec_type):
        if vec_type == "sen":
            return self.sent_vecs[self.word_index[word]]


    def get_inverse_sent_vec(self, word):
            return self.dict_vec.inverse_transform(self.sent_vecs[self.word_index[word]])