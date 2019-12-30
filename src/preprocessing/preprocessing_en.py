import string
import pandas as pd
import numpy as np
import multiprocessing
from nltk.stem import PorterStemmer
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from datetime import datetime

# todo know issues: id=93873, cause of apostrophe it concatenates mulitple rows


stopwords = set(stopwords.words('english'))
tokenizer = WhitespaceTokenizer()
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def _stem_text(text: str):
    return ' '.join([stemmer.stem(t) for t in tokenizer.tokenize(text)])


def _lemmatize_text(text: str):
    return ' '.join([lemmatizer.lemmatize(t) for t in tokenizer.tokenize(text)])


def _remove_stopwords(text: str):
    return ' '.join(t for t in tokenizer.tokenize(text) if t not in stopwords)


def preprocess(file_path: str, rows=-1, stop=True, stemm=True):
    # loading
    start_func = datetime.now()
    print('Loading data')
    if rows == -1:
        df = pd.read_csv(file_path, error_bad_lines=False, header=0, usecols=['ID', 'TITLE', 'CATEGORY'])
    else:
        df = pd.read_csv(file_path, nrows=rows, error_bad_lines=False, header=0, usecols=['ID', 'TITLE', 'CATEGORY'])

    df.columns = ['id', 'title', 'class']
    print('Data loaded')

    print('Actual preprocessing start')
    start = datetime.now()
    df['title_original'] = df['title']

    # lowercasing
    df['title'] = df['title'].str.lower()

    # punctuation removal
    df['title'] = df['title'].str.translate(str.maketrans('', '', string.punctuation))
    print('Lowercase/punctuation=' + str(datetime.now() - start))

    # stopwords removal
    if stop:
        start = datetime.now()
        df['title'] = df['title'].apply(_remove_stopwords)
        print('Stopwords=' + str(datetime.now() - start))

    # stemming
    if stemm:
        start = datetime.now()
        # df['title_stem'] = df['title'].apply(_stem_text)
        df['title_lemma'] = df['title'].apply(_lemmatize_text)
        # the longer the operation, the more important it is to parallel it
        df = _parallelize(df, _stemm_df)
        print('Lemma/stemming=' + str(datetime.now() - start))

    print('Actual preprocessing end, took=' + str(datetime.now() - start_func))

    return df


def _stemm_df(df: pd.DataFrame):
    df['title_stem'] = df['title'].apply(_stem_text)
    return df


def _parallelize(df: pd.DataFrame, func):
    processing_cores = multiprocessing.cpu_count() - 1

    sub_dfs = np.array_split(df, processing_cores)

    pool = multiprocessing.Pool(processing_cores)
    df = pd.concat(pool.map(func, sub_dfs))

    # no more data will be added
    pool.close()
    # blocks until process terminates
    pool.join()
    return df


if __name__ == '__main__':
    preprocess('../../resources/uci-news-aggregator.csv')
