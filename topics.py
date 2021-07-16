#!/usr/bin/env python
from bs4 import BeautifulSoup
import gensim
from gensim.test.utils import common_corpus, common_dictionary
from gensim.models import HdpModel
from gensim.corpora.dictionary import Dictionary
import numpy as np
import pandas as pd
import requests
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from nltk import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import collections
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


def get_text():
    url = "https://www.fool.com/investing/2021/07/12/8-lies-that-fueled-the-amc-pump-and-dump-scheme/"
    res = requests.get(url)
    html_page = res.content
    soup = BeautifulSoup(html_page, 'html.parser')
    result = soup.find_all("p")
    paragraph_text = ' '.join(p.text for p in result)
    paragraph_text = paragraph_text.strip()
    paragraph_text = paragraph_text.replace("\n","")
    sentences = paragraph_text.split(".")
    return(sentences)
get_text()

def topic_vectors():
    sentences = get_text()
    stoplist = set('for a of the and to in an but if how why then not how why'.split())
    data = pd.DataFrame()
    data["text"] = sentences

    lemmatizer = WordNetLemmatizer()
    data["lemmas"] = data["text"].apply(lambda x: [lemmatizer.lemmatize(y) for y in x])

    n_features = 9
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english', ngram_range=(1, 2))
    tf = tf_vectorizer.fit_transform(sentences)
    lda = LatentDirichletAllocation(n_components=4, random_state=1).fit(tf)
    x_lda = lda.fit_transform(tf)
    vocab = tf_vectorizer.get_feature_names()
    n_top_words =93
    topic_words = {}
    for topic, comp in enumerate(lda.components_):
        word_idx = np.argsort(comp)[::-1][:n_top_words]
        # store the words most relevant to the topic
        topic_words[topic] = [vocab[i] for i in word_idx]
        data["topics"] = [vocab[i] for i in word_idx]
    X_train = x_lda
    X_test = x_lda
    Y_target = data['topics']
    y_train = Y_target
    log = LogisticRegression(
        penalty="l2",
        C=10000,
        class_weight="balanced",
        solver="liblinear"
    )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    log.fit(X_scaled,y_train)
    log_score = log.score(X_test,Y_target)
    pred = log.predict(X_test)
    print("logistic regression: ",log_score)
    svm = SVC(kernel="rbf",C=10,gamma="auto")
    svm.fit(X_train,y_train)
    print("support vec: ",svm.score(X_test,Y_target))
    #print(svm.predict(X_test))
    rf = RandomForestClassifier()
    rf.fit(X_train,y_train)
    print("random forest: ", rf.score(X_test,Y_target))
    nb = MultinomialNB()
    nb.fit(X_train,y_train)
    print("naive bayes: ", nb.score(X_test,Y_target))
topic_vectors()
