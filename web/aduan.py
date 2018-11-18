import re

from nltk.tokenize import word_tokenize
from sklearn.externals import joblib
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

def load_count_vectorizer():
    clf = joblib.load('count_vectorizer.joblib')
    return clf

def load_classifier():
    clf = joblib.load('classifier.joblib')
    return clf

def load_pos_tag():
    clf = joblib.load('pos_tagger.joblib')
    return clf

def load_dict_vectorizer():
    clf = joblib.load('dict_vectorizer.joblib')
    return clf

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

def neighbor_word(sentence, i):
    if i == 0 :
        prev_word = ''
    else :
        prev_word = sentence[i-1]
    if i == len(sentence)-1 :
        next_word = ''
    else :
        next_word = sentence[i+1]
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
    for sentence in sentences:
        for i in range(len(sentence)):
            features = {}

            word = sentence[i]
            features.update({'value': word})
            
            features.update(neighbor_word(sentence, i))
            features.update(word_position(sentence, i))
            features.update(morphems(word))
            features.update(word_case(word))
            features.update(has_hyphen(word))
            features.update(is_digit(word))
            
            X.append(features)
    return X

def vectorize_features(features):
    dict_vect = load_dict_vectorizer()
    features_v = dict_vect.transform(features)
    return features_v

def pairing_tag(tweets, tags):
    j = 0
    pair_tweet = []
    for tweet in tweets:
        pair = []
        for i in range(len(tweet)):
            pair.append((tweet[i], tags[j]))
            j = j + 1
        pair_tweet.append(pair)
    return pair_tweet

def cek_aduan(text):
    tweet = (text,)

    pre_tweet = preprocess_pos_tag(tweet)
    prep_tweet = preprocess_classifier(pre_tweet)
    vector_tweet = word_vectorizer(prep_tweet)

    classifier = load_classifier()

    y = classifier.predict(vector_tweet)
    return y == 1

def deteksi_pos_tag(text):
    tweet = (text,)

    pre_tweet = preprocess_pos_tag(tweet)
    featured_tweet = feature_extractor(pre_tweet)
    vector_tweet = vectorize_features(featured_tweet)

    pos_tagger = load_pos_tag()

    tags = pos_tagger.predict(vector_tweet)
    paired_tweet = pairing_tag(pre_tweet, tags)

    Noun = []
    Verb = []
    Angka = []

    pairs = paired_tweet[0]

    for i in range(0, len(pairs)):
        if ((pairs[i][1]=='NN') |( pairs[i][1]=='NNP' )| (pairs[i][1]=='NND') ):
            Noun.append(pairs[i][0])
        elif (pairs[i][1]=='VB' ):
            Verb.append(pairs[i][0])
        elif ((pairs[i][1]=='CD' )| (pairs[i][1]=='OD') ):
            Angka.append(pairs[i][0] + ' ' + pairs[i+1][0])

    return Noun, Verb, Angka