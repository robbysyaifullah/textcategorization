#!/usr/bin/env python
# coding: utf-8

import codecs, json, csv
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pandas as pd
import string as string

import re

from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import random

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_score
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

with open('Indonesian_Manually_Tagged_Corpus.tsv', 'r', encoding='utf-8') as tsvfile :
    rows = tsvfile.read().split('\n')
print ('input data', rows[0:5])
pair = []
sentences = []

for each in rows:
    if each == '':
        sentences.append(pair)
        pair = [] 
    else:
        word, tag = each.split('\t')
        pair.append([word, tag])

print ('Preprocessing ', sentences[0])

def neighbor_word(sentence, i):
    if i == 0 :
        prev_word = ''
    else :
        prev_word = sentence[i-1][0]
    if i == len(sentence)-1 :
        next_word = ''
    else :
        next_word = sentence[i+1][0]
    return {'prev_word' : prev_word, 'next_word' : next_word}

def morphems(word):
    prefix_1 = word[0]
    prefix_2 = word[:2]
    prefix_3 = word[:3]
    prefix_4 = word[:4]
    suffix_1 = word[-1]
    suffix_2 = word[-2:]
    suffix_3 = word[-3:]

    return {'prefix_1' : prefix_1, 'prefix_2' : prefix_2, 'prefix_3' : prefix_3, 'prefix_4' : prefix_4, 'suffix_1' : suffix_1, 'suffix_2' : suffix_2, 'suffix_3' : suffix_3}

def has_hyphen(word):
    return {'has_hyphen' : '-' in word}

def is_digit(word):
    return {'is_digit' : word.isdigit()}

def word_case(word):
    is_capitalized = word[0].upper() == word[0]
    return {'is_capitalized' : is_capitalized}

def word_position(sentence, index):
    if (index == 0):
        pos = 0
        prev_pos = -1
        next_pos = 1
    elif (index == len(sentence)):
        pos = 2
        prev_pos = 1
        next_pos = -1
    else :
        if(index == 1):
            prev_pos = 0
            pos = 1
            next_pos = 2
        elif(index == len(sentence)-1):
            prev_pos = 1
            pos = 2
            next_pos = -1
        else :
            prev_pos = pos = next_pos = 1
    return {'prev_pos' : prev_pos, 'pos' : pos, 'next_pos' : next_pos}

def feature_extractor(sentences):
    X = []
    y = []
    for sentence in sentences:
        for i in range(len(sentence)):
            features = {}

            word = sentence[i][0]
            features.update({'value': word})
            
            features.update(neighbor_word(sentence, i))
            features.update(word_position(sentence, i))
            features.update(morphems(word))
            features.update(word_case(word))
            features.update(has_hyphen(word))
            features.update(is_digit(word))
            
            X.append(features)
            y.append(sentence[i][1])
    return X, y
X, y = feature_extractor(sentences)
print( 'feature ext', X[1])

import random
dict_vect = DictVectorizer(sparse=False)
size = 20000

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

y_train = y_train[0:size]
X_train = dict_vect.fit_transform(X_train[0:size])

y_test = y_test[0:size]
X_test = dict_vect.transform(X_test[0:size])

model = LogisticRegression()
model.fit(X_train, y_train)

prediction = model.predict(X_test)

print('F1 Score Logistic : ', f1_score(prediction, y_test, average='macro', labels=np.unique(prediction)))
print('Accuracy Score Logistic : ', accuracy_score(prediction, y_test))
print('Precision Score Logistic : ', precision_score(prediction, y_test, average='macro'))
print('Recall Score Logistic : ', recall_score(prediction, y_test, average='macro', labels=np.unique(prediction)))

from sklearn.externals import joblib
joblib.dump(dict_vect, 'model/dict_vectorizer.joblib')
joblib.dump(model, 'model/pos_tagger.joblib')
