import sys

import pandas as pd
import numpy as np

from gensim.models import KeyedVectors

sys.path.append('..')


# print(model.most_similar('teacher'))
# print(model.similarity('teacher', 'teaches'))

class MeanEmbeddingVectorizer(object):
    # source http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/

    def __init__(self):
        self.model = KeyedVectors.load_word2vec_format('../../resources/wiki-news-300d-1M.vec')
        # if a text is empty we should return a vector of zeros with the same dimensionality as all the other vectors
        self.dim = self.model.vector_size

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.model[token] for token in sentence.split() if token in self.model] or [np.zeros(self.dim)],
                    axis=0) for sentence in X
        ])


if __name__ == '__main__':
    v = MeanEmbeddingVectorizer()

    l = ['feds plosser taper pace may slow']
    df = pd.Series(l)

    v.transform(df)
