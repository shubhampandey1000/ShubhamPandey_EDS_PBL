import pandas as pd
from collections import Counter
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk.stem import SnowballStemmer
from nltk.stem. porter import *
from gensim.utils import simple_preprocess

# prepare with additional stopwords
english_stops = stopwords.words('english')
english_stops.append("mr")
english_stops.append("mrs")
english_stops.append("miss")
english_stops.append("colonel")
english_stops.append("volume")
english_stops.append("chapter")
english_stops = set(english_stops)

# read in text files
novels = ['emma','mansfieldpark','northanger','persuasion','pridenp','sensensense']
inbooks = []
for novel in novels:
    f = open('./data/' + novel + '.txt', encoding = 'utf-8')
    inbooks.append(f.read())

emma = inbooks[0]
mansfield = inbooks[1]
northanger = inbooks[2]
persuasion = inbooks[3]
pridenp = inbooks[4]
sensensense = inbooks[5]


# EMMA PREPROCESSING - WORDS
# remove 30 lines, punctuation, chapter headers, lemmatize

emma.find('VOLUME')
emma.find('FINIS')
emma = emma[611:883631]

pattern = r"[A-Za-z]+"
emtok = nltk.regexp_tokenize(emma, pattern)
emwords = [w.lower() for w in emtok]

nostops = [w for w in emwords if w not in english_stops]

# lemmatizing

wnlemma = WordNetLemmatizer()
emmatized = [wnlemma.lemmatize(w) for w in nostops]
emmatized = [w for w in emmatized if w not in ['chapter']]

#sentences

emsent = sent_tokenize(emma)
# as lists of lists of words
emsplit_sentences = []
wnlemma = WordNetLemmatizer()
for sentence in emsent:
    tokens = nltk.regexp_tokenize(sentence,pattern)
    tokens = [w.lower() for w in tokens]
    tokens = [w for w in tokens if w not in english_stops]
    tokens = [wnlemma.lemmatize(w) for w in tokens]
    emsplit_sentences.append(tokens)





# MANSFIELD PARK PREPROCESSING - WORDS

mansfield.find('CHAPTER')
mansfield.find('THE END')
mansfield = mansfield[659:883910]

pattern = r"[A-Za-z]+"
manstok = nltk.regexp_tokenize(mansfield, pattern)
manswords = [w.lower() for w in manstok]

nostops = [w for w in manswords if w not in english_stops]

# lemmatizing
wnlemma = WordNetLemmatizer()
manslemmatized = [wnlemma.lemmatize(w) for w in nostops]
manslemmatized = [w for w in manslemmatized if w not in ['chapter']]

# sentences

mansent = sent_tokenize(mansfield)

# as lists of lists of words
mansplit_sentences = []
wnlemma = WordNetLemmatizer()
for sentence in mansent:
    tokens = nltk.regexp_tokenize(sentence,pattern)
    tokens = [w.lower() for w in tokens]
    tokens = [w for w in tokens if w not in english_stops]
    tokens = [wnlemma.lemmatize(w) for w in tokens]
    mansplit_sentences.append(tokens)


# NORTHANGER ABBEY PREPROCESSING - WORDS

northanger.find('CHAPTER')
northanger.rfind('Rambler')
northanger = northanger[1476:433575]

pattern = r"[A-Za-z]+"
northtok = nltk.regexp_tokenize(northanger, pattern)
northwords = [w.lower() for w in northtok]

nostops = [w for w in northwords if w not in english_stops]

# lemmatizing
wnlemma = WordNetLemmatizer()
northlemmatized = [wnlemma.lemmatize(w) for w in nostops]
northlemmatized = [w for w in northlemmatized if w not in ['chapter']]

# sentences

northsent = sent_tokenize(northanger)

# as lists of lists of words
northsplit_sentences = []
wnlemma = WordNetLemmatizer()
for sentence in northsent:
    tokens = nltk.regexp_tokenize(sentence,pattern)
    tokens = [w.lower() for w in tokens]
    tokens = [w for w in tokens if w not in english_stops]
    tokens = [wnlemma.lemmatize(w) for w in tokens]
    northsplit_sentences.append(tokens)


# PERSUASION PREPROCESSING - WORDS

persuasion.find('Chapter 1')
persuasion.rfind('Finis')
persuasion = persuasion[629:467438]

pattern = r"[A-Za-z]+"
perstok = nltk.regexp_tokenize(persuasion, pattern)
perswords = [w.lower() for w in perstok]

nostops = [w for w in perswords if w not in english_stops]

# lemmatizing
wnlemma = WordNetLemmatizer()
perslemmatized = [wnlemma.lemmatize(w) for w in nostops]
perslemmatized = [w for w in perslemmatized if w not in ['chapter']]

# sentences

persent = sent_tokenize(persuasion)

# as lists of lists of words
persplit_sentences = []
wnlemma = WordNetLemmatizer()
for sentence in persent:
    tokens = nltk.regexp_tokenize(sentence,pattern)
    tokens = [w.lower() for w in tokens]
    tokens = [w for w in tokens if w not in english_stops]
    tokens = [wnlemma.lemmatize(w) for w in tokens]
    persplit_sentences.append(tokens)

# PRIDE AND PREJUDICE PREPROCESSING - WORDS

pridenp.find('Chapter 1')
pridenp.rfind('uniting them')
pridenp = pridenp[665:685406]

pattern = r"[A-Za-z]+"
pridetok = nltk.regexp_tokenize(pridenp, pattern)
pridewords = [w.lower() for w in pridetok]

nostops = [w for w in pridewords if w not in english_stops]

# lemmatizing
wnlemma = WordNetLemmatizer()
pridelemmatized = [wnlemma.lemmatize(w) for w in nostops]
pridelemmatized = [w for w in pridelemmatized if w not in ['chapter']]

# sentences

pridesent = sent_tokenize(pridenp)

# as lists of lists of words
pridesplit_sentences = []
wnlemma = WordNetLemmatizer()
for sentence in pridesent:
    tokens = nltk.regexp_tokenize(sentence,pattern)
    tokens = [w.lower() for w in tokens]
    tokens = [w for w in tokens if w not in english_stops]
    tokens = [wnlemma.lemmatize(w) for w in tokens]
    pridesplit_sentences.append(tokens)


# SENSE AND SENSIBILITY PREPROCESSING - WORDS

sensensense.find('CHAPTER')
sensensense.rfind('THE END')
sensensense = sensensense[698:674328]

pattern = r"[A-Za-z]+"
sensetok = nltk.regexp_tokenize(sensensense, pattern)
sensewords = [w.lower() for w in sensetok]

nostops = [w for w in sensewords if w not in english_stops]

# lemmatizing
wnlemma = WordNetLemmatizer()
senselemmatized = [wnlemma.lemmatize(w) for w in nostops]
senselemmatized = [w for w in senselemmatized if w not in ['chapter']]

# sentences

sensent = sent_tokenize(sensensense)

# as lists of lists of words
sensplit_sentences = []
wnlemma = WordNetLemmatizer()
for sentence in sensent:
    tokens = nltk.regexp_tokenize(sentence,pattern)
    tokens = [w.lower() for w in tokens]
    tokens = [w for w in tokens if w not in english_stops]
    tokens = [wnlemma.lemmatize(w) for w in tokens]
    sensplit_sentences.append(tokens)



