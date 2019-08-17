import multiprocessing
import sys

import gensim
import pandas as pd
from gensim.models import KeyedVectors

sys.path.append('..')

from preprocessing.preprocessing_en import preprocess

print('Preprocessing data')
df: pd.DataFrame = preprocess('../../resources/uci-news-aggregator.csv')

print('Creating model')
sentences = list(df['title'])
model = gensim.models.Word2Vec(
    sentences,
    size=100,
    window=10,
    min_count=1,
    workers=multiprocessing.cpu_count() - 1,
    iter=10,
)

word_vectors = model.wv

print('Saving model')
word_vectors.save('model.wv')
word_vectors = KeyedVectors.load('model.wv', mmap='r')
