import datetime

import nltk
import numpy as np
import pandas as pd
from gensim.models import TfidfModel
from gensim.models import LdaMulticore
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from multiprocessing import cpu_count


class Lda:
    stemmer = SnowballStemmer('english')

    def __init__(self):
        nltk.download('wordnet')
        self.workers = cpu_count() - 1

    def stem_lemmatize(self, text):
        return self.stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

    def preprocess(self, text):
        result = []
        for token in simple_preprocess(text):
            if token not in STOPWORDS and len(token) > 3:
                result.append(self.stem_lemmatize(token))
        return result

    def main(self):
        print('Loading data')
        data = pd.read_csv('abcnews-date-text.csv', error_bad_lines=False)
        data_text = data[['headline_text']]
        data_text['index'] = data_text.index
        documents = data_text

        np.random.seed(2018)

        print('Preprocessing text')
        preprocessed_docs = documents['headline_text'].map(self.preprocess)

        print('Building bag of words corpus')
        dictionary = Dictionary(preprocessed_docs)   # list: token_id, token
        dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
        bow_corpus = [dictionary.doc2bow(doc) for doc in preprocessed_docs]     # list: token_id, token_count

        print(documents[documents['index'] == 4310].values[0][0])
        print(bow_corpus[4310])
        print(bow_corpus[:100])

        print('Building lda model from bag of words')
        lda_model_bow = LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, workers=self.workers)
        for idx, topic in lda_model_bow.print_topics(-1):
            print('Topic: {} \nWords: {}'.format(idx, topic))

        for index, score in sorted(lda_model_bow[bow_corpus[4310]], key=lambda tup: -1*tup[1]):
            print("\nScore: {}\t \nTopic: {}".format(score, lda_model_bow.print_topic(index, 10)))

        print('Building tfidf corpus from bag of words corpus')
        tfidf = TfidfModel(bow_corpus)
        tfidf_corpus = tfidf[bow_corpus]
        from pprint import pprint
        for doc in tfidf_corpus:
            pprint(doc)
            break

        print('Building lda model from tfidf')
        lda_model_tfidf = LdaMulticore(tfidf_corpus, num_topics=10, id2word=dictionary, workers=self.workers)
        for idx, topic in lda_model_tfidf.print_topics(-1):
            print('Topic: {} Word: {}'.format(idx, topic))

        for index, score in sorted(lda_model_tfidf[bow_corpus[4310]], key=lambda tup: -1*tup[1]):
            print("\nScore: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))

        print('Testing on unseen document')
        unseen_document = 'Facebook’s global lobbying against data privacy laws'
        bow_vector = dictionary.doc2bow(self.preprocess(unseen_document))

        print('Bow:')
        for index, score in sorted(lda_model_bow[bow_vector], key=lambda tup: -1*tup[1]):
            print("Score: {}\t Topic: {}".format(score, lda_model_bow.print_topic(index, 5)))

        print('TfIdf:')
        for index, score in sorted(lda_model_tfidf[bow_vector], key=lambda tup: -1*tup[1]):
            print("Score: {}\t Topic: {}".format(score, lda_model_tfidf.print_topic(index, 5)))


if __name__ == '__main__':
    start = datetime.datetime.now()
    print(start)

    lda = Lda()
    lda.main()

    end = datetime.datetime.now()
    print(end)
    print(end - start)
