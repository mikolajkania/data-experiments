import sys
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

sys.path.append('..')

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
        'svd_components': 2000,
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

    print('Plotting svd vs true variance')
    plt.plot(svd.explained_variance_ratio_, label='svd explained variance')
    variances = np.var(X_svd, axis=0)
    total_variance = np.var(X.todense(), axis=0).sum()
    true_explained_variance_ratio = variances / total_variance
    plt.plot(true_explained_variance_ratio, label='true explained variance')
    plt.legend(loc='best')
    plt.xlabel('svd components')
    plt.ylabel('fraction of total variance')
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
