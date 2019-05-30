import sys
sys.path.append('..')

from preprocessing.preprocessing_en import preprocess

if __name__ == '__main__':
    preprocess('../../resources/uci-news-aggregator.csv')