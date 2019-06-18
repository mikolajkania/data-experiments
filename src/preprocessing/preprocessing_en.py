import string
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


# todo know issues: id=93873, cause of apostrophe it concatenates mulitple rows


stopwords = set(stopwords.words('english'))


def preprocess(file_path: str, rows=-1):
    # loading
    print('Loading data')
    if rows == -1:
        df = pd.read_csv(file_path, error_bad_lines=False, header=0, usecols=['ID', 'TITLE', 'CATEGORY'])
    else:
        df = pd.read_csv(file_path, nrows=rows, error_bad_lines=False, header=0, usecols=['ID', 'TITLE', 'CATEGORY'])
    df.columns = ['id', 'title', 'class']
    print('Data loaded')

    print('Actual preprocessing start')
    df['title_original'] = df['title']

    # lowercasing
    df['title'] = df['title'].str.lower()

    # punctuation removal
    df['title'] = df['title'].str.translate(str.maketrans('', '', string.punctuation))

    # stopwords removal
    tokenizer = WhitespaceTokenizer()

    def remove_stopwords(text):
        return ' '.join(t for t in tokenizer.tokenize(text) if t not in stopwords)

    df['title'] = df['title'].apply(remove_stopwords)

    # stemming
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    def stem_text(text):
        return ' '.join([stemmer.stem(t) for t in tokenizer.tokenize(text)])

    def lemmatize_text(text):
        return ' '.join([lemmatizer.lemmatize(t) for t in tokenizer.tokenize(text)])

    df['title_stem'] = df['title'].apply(stem_text)
    df['title_lemma'] = df['title'].apply(lemmatize_text)

    print('Actual preprocessing end')

    return df


if __name__ == '__main__':
    preprocess('../../resources/uci-news-aggregator.csv')
