import re

from nltk.tokenize import word_tokenize
from sklearn.externals import joblib
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

def preprocess_twitter(raw):
    #remove hashtag, link, and @
    cleanr = re.compile("http?:\/\/.*[\r\n]*")
    cleantext = re.sub(cleanr, '', raw)
    
    cleanr = re.compile("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)*")
    cleantext = re.sub(cleanr, '', cleantext)
    
    cleantext = re.sub(cleanr, '', cleantext)
    
    return cleantext

def tokenizer(w):
    words = word_tokenize(w)
    return words

def preprocess_pos_tag(X):
    preprocess1 = []
    preprocess2 = []

    for i in range(len(X)):
        preprocess1.append(preprocess_twitter(X[i]))
    
    for i in range(len(preprocess1)):
        preprocess2.append(tokenizer(preprocess1[i]))
    
    return preprocess2

def stemWord(words):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    sentence = ''
    for word in words:
        sentence += str(stemmer.stem(word)) + ' '
    return sentence

def preprocess_classifier(X):
    preprocess1 = []

    for i in range(len(X)):
        preprocess1.append(stemWord(X[i]))
    
    return preprocess1

def word_vectorizer(X):
    count_vect = load_count_vectorizer()
    X = count_vect.transform(X)
    return X 

def load_count_vectorizer():
    clf = joblib.load('count_vectorizer.joblib')
    return clf

def load_classifier():
    clf = joblib.load('classifier.joblib')
    return clf

def cek_aduan(text):
    tweets = (text,)

    pre_tweets = preprocess_pos_tag(tweets)
    prep_tweets = preprocess_classifier(pre_tweets)
    vector_tweets = word_vectorizer(prep_tweets)

    classifier = load_classifier()

    y = classifier.predict(vector_tweets)
    return y == 1
