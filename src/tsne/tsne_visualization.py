import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

sys.path.append('..')

from preprocessing.preprocessing_en import preprocess

if __name__ == '__main__':
    df: pd.DataFrame = preprocess('../../resources/uci-news-aggregator.csv')
    classes = ['b', 't', 'e', 'm']

    X = df['title']
    y = df['class']

    print('Vectorizer')
    vectorizer = TfidfVectorizer(min_df=5, max_df=0.25)
    X = vectorizer.fit_transform(X)
    print(vectorizer.get_feature_names()[:10])

    print('TruncatedSVD')
    svd = TruncatedSVD(n_components=100)
    X_svd = svd.fit_transform(X)

    print('TSNE')
    tsne = TSNE(n_components=2, verbose=1)
    X_2d = tsne.fit_transform(X_svd)

    print('Plotting')
    plt.figure(figsize=(6, 5))
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
    for i, c, label in zip(classes, colors, classes):
        plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)
    plt.legend()
    plt.show()
