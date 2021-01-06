import bz2
import numpy as np
import pandas as pd


class Word2Vec:
    def __init__(self, path_words, path_context):
        self.vectors_words, self.w2i_words = self.load_w2vec(path_words)
        self.vectors_contexts, self.w2i_context = self.load_w2vec(path_context)
        self.i2w_words = {v: k for k, v in self.w2i_words.items()}

    def load_w2vec(self, path):
        with bz2.BZ2File(path) as f:
            w2i = {}
            vectors = []
            for i, l in enumerate(f.readlines()):
                str_line = l.decode("utf8")
                str_vec = str_line.split()
                word = str_vec.pop(0)
                w2i[word] = i
                vectors.append(np.array([float(v) for v in str_vec]))
        return vectors, w2i

    def get_most_similar_w2v(self, cur_w, top_n=10):
        target_vector = self.vectors_words[self.w2i_words[cur_w]]
        sims = []
        for i, v in enumerate(self.vectors_words):
            sims.append((self.i2w_words[i], target_vector.dot(v)))
        sims.sort(key=lambda x: x[1], reverse=True)
        sims = sims[1:top_n + 1]
        return [att for att, score in sims]

    def get_most_similar_w2v_context(self, cur_w, top_n=10):
        target_vector = self.vectors_words[self.w2i_words[cur_w]]
        sims = []
        for i, v in enumerate(self.vectors_words):
            sims.append((self.i2w_words[i], target_vector.dot(v)))
        sims.sort(key=lambda x: x[1], reverse=True)
        sims = sims[1:top_n + 1]
        return [att for att, score in sims]

    def print_similar_words(self):
        print("Words")
        print()
        for cur_w in 'car bus hospital hotel gun bomb horse fox table bowl guitar piano'.split():
            bow = self.get_most_similar_w2v(cur_w, 20)
            dep = self.get_most_similar_w2v(cur_w,  20)
            print("            ", cur_w)
            print(pd.DataFrame([{f'bow5 {cur_w}': b, f'deps {cur_w}': d} for b, d in zip(bow, dep)]))

        print("Context")
        print()
        for cur_w in 'car bus hospital hotel gun bomb horse fox table bowl guitar piano'.split():
            bow = self.get_most_similar_w2v_context(cur_w)
            dep = self.get_most_similar_w2v_context(cur_w)
            print("            ", cur_w)
            print(pd.DataFrame([{f'bow5 {cur_w}': b, f'deps {cur_w}': d} for b, d in zip(bow, dep)]))

if __name__ == '__main__':
