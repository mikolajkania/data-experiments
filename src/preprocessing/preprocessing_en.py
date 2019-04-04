import string
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# loading
print('Loading data')
df = pd.read_csv('../../resources/uci-news-aggregator.csv', error_bad_lines=False, header=0,
                 usecols=['TITLE', 'CATEGORY'])
df.columns = ['title', 'class']
df['title_original'] = df['title']

# lowercasing
df['title'] = df['title'].str.lower()

# punctuation removal
df['title'] = df['title'].str.translate(str.maketrans('', '', string.punctuation))

# stopwords removal
tokenizer = WhitespaceTokenizer()
stopwords = set(stopwords.words('english'))


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
