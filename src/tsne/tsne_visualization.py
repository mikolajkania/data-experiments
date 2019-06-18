import sys
import pickle

import matplotlib.pyplot as plt
import pandas as pd
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
    df: pd.DataFrame = preprocess('../../resources/uci-news-aggregator.csv')
    classes = ['b', 't', 'e', 'm']

    X = df['title']
    y = df['class']

    params = {
        'use_cache': False,
        'model_name': 'svd',
        'svd_components': 500,
        'tsne_components': 2,
        'min_df': 5,
        'max_df': 0.25
    }
    print('Params=' + str(params))

    print('Vectorizer')
    vectorizer = TfidfVectorizer(min_df=params['min_df'], max_df=params['max_df'])
    X = vectorizer.fit_transform(X)
    print(vectorizer.get_feature_names()[:10])

    print('TruncatedSVD')
    full_model_name = params['model_name'] + '_' + str(params['svd_components'])
    if params['use_cache']:
        svd: TruncatedSVD = load(full_model_name)
    else:
        svd = TruncatedSVD(n_components=params['svd_components'])
        svd.fit(X)
        save(svd, full_model_name)
    X_svd = svd.transform(X)
    print('EVR sum=' + str(svd.explained_variance_ratio_.sum()))

    print('TSNE')
    tsne = TSNE(n_components=params['tsne_components'], verbose=1)
    X_2d = tsne.fit_transform(X_svd)

    print('Plotting')
    plt.figure(figsize=(6, 5))
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
    for i, c, label in zip(classes, colors, classes):
        plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)
    plt.legend()
    plt.show()
