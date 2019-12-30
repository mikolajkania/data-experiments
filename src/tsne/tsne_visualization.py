import sys
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

sys.path.append('..')

# tokens: 422 419 for min_df=5, max_df=0.25, i_iter=5; for bigger values it required more than 16GB of RAM
# svd_components / EVR
# 10   / 0.020065668406173024
# 20   / 0.034024560030019024
# 50   / 0.0653887673737613
# 100  / 0.10454540835468916
# 200  / 0.1633134157358559
# 500  / 0.2801729330821833
# 1000 / 0.4042387142333503
# 1500 / 0.4905397745045663
# 2000 / 0.5552525728609641
# 3000 / 0.6490454190312016
# 4000 / 0.9128349871927357
# 5000 / 0.9550601876255402

from preprocessing.preprocessing_en import preprocess


def load(name: str):
    with open(name + '.p', 'rb') as f:
        return pickle.load(f)


def save(object, name: str):
    pickle.dump(object, open(name + '.p', 'wb'))


if __name__ == '__main__':
    params = {
        'rows': 44000,
        'use_cache': False,
        'model_name': 'svd',
        'svd_components': 5000,
        'tsne_components': 2,
        'min_df': 5,
        'max_df': 0.25
    }

    df: pd.DataFrame = preprocess('../../resources/uci-news-aggregator.csv', rows=params['rows'])
    labels = ['b', 't', 'e', 'm']

    X = df['title']
    y = df['class']

    print('Params=' + str(params))

    print('Vectorizer')
    vectorizer = TfidfVectorizer(min_df=params['min_df'], max_df=params['max_df'])
    X = vectorizer.fit_transform(X)
    print(vectorizer.get_feature_names()[:10])

    print('TruncatedSVD')
    full_model_name = params['model_name'] + '_' + str(params['svd_components']) + '_' + str(params['rows'])
    if params['use_cache']:
        svd: TruncatedSVD = load(full_model_name)
    else:
        svd = TruncatedSVD(n_components=params['svd_components'])
        svd.fit(X)
        save(svd, full_model_name)
    X_svd = svd.transform(X)
    print('EVR sum=' + str(svd.explained_variance_ratio_.sum()))

    plt.plot(np.cumsum(svd.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()

    print('TSNE')
    tsne = TSNE(n_components=params['tsne_components'], verbose=1)
    X_2d = tsne.fit_transform(X_svd)

    print('Plotting 1')
    plt.figure(figsize=(6, 5))
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
    for i, l, label in zip(labels, colors, labels):
        plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=l, label=label)
    plt.legend()
    plt.show()

    print('Plotting 2')
    from yellowbrick.text import TSNEVisualizer

    tsne = TSNEVisualizer(decompose=None, decompose_by=params['svd_components'], labels=labels)
    tsne.fit(X_svd, y)
    tsne.poof()
