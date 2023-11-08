import string
import re
import umap
import numpy as np
import pandas as pd

from scipy.spatial.distance import cosine

import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec

books = pd.read_csv('book-dataset/books_data.csv')

books.dropna(inplace=True)

books = books.head(100)
titles = dict(zip(books['description'], books['Title']))
nltk.download('stopwords')


def tokenize(text):
    return TreebankWordTokenizer().tokenize(text)


def lower(words):
    return [word.lower() for word in words]


def remove_stopwords(words):
    stopwords_list = stopwords.words('english')
    return [word for word in words if word not in stopwords_list]


def remove_punct(words):
    puncts = string.punctuation
    return [''.join(char for char in word if char not in puncts) for word in words]


def remove_mess(words):
    return [word for word in words if len(word) > 2]


def remove_digits(words):
    return [word for word in words if not re.match('\d+|\d+th', word)]


def remove_tags(text):
    text = re.sub('&\w+;', '', text)
    return text


def clean_text(text):
    text = remove_tags(text)
    tokens = tokenize(text)
    lower_words = lower(tokens)
    clean_words = remove_stopwords(lower_words)
    words = remove_punct(clean_words)
    words = remove_digits(words)
    return remove_mess(words)


books['clean_tokens'] = books['description'].apply(clean_text)

nltk.download('wordnet')

wn = WordNetLemmatizer()


def lemmatize(words):
    return [wn.lemmatize(word) for word in words]


books['lemmantized_tokens'] = books['clean_tokens'].apply(lemmatize)

model = Word2Vec(window=10, sg=1, hs=0,
                 negative=10,
                 alpha=0.03, min_alpha=0.0007,
                 seed=14)

model.build_vocab(books['lemmantized_tokens'], progress_per=200)

model.train(books['lemmantized_tokens'], total_examples=model.corpus_count,
            epochs=10, report_delay=1)


def get_description_vector(book_description):
    tokens = lemmatize(clean_text(book_description))
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    if vectors:
        descr_vector = np.mean(vectors, axis=0)
        return descr_vector
    else:
        return None

e = dict()


# Расчет косинусного расстояния с другими книгами
def get_embeddings():
    for other_book in books['description']:
        other_book_vector = get_description_vector(other_book)
        if other_book_vector is not None:
            e[other_book] = other_book_vector
    return e


def get_similar_books(description, e):
    description_vector = get_description_vector(description)
    similar_books = []
    for other_book in books['description']:
        other_book_vector = get_description_vector(other_book)
        if other_book_vector is not None:
            e[other_book] = other_book_vector
            similarity = 1 - cosine(description_vector, other_book_vector)
            similar_books.append((other_book, similarity))

    # Сортировка книг по убыванию схожести
    similar_books = sorted(similar_books, key=lambda x: x[1], reverse=True)
    result = list()
    for descr in similar_books[:10]:
        result.append(titles[descr[0]])
    return result
