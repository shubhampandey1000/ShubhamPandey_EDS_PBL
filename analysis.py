from gensim.corpora.dictionary import Dictionary
from collections import defaultdict
import itertools
import numpy as np
import pandas as pd
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.tfidfmodel import TfidfModel
from gensim.models import Word2Vec
from gensim.models import CoherenceModel
from gensim.models.phrases import Phrases, Phraser
from gensim.matutils import corpus2csc
np.random.seed(748)

novels = [emmatized, manslemmatized, northlemmatized, perslemmatized, pridelemmatized, senselemmatized]

# create corpus

dictionary = Dictionary(novels)
corpus = [dictionary.doc2bow(novel) for novel in novels]

# create default dict, sort
total_word_count = defaultdict(int)
for word_id, word_count in itertools.chain.from_iterable(corpus):
    total_word_count[word_id] += word_count

sorted_count = sorted(total_word_count.items(), key=lambda w: w[1], reverse=True)

#top 20 words in corpus

for word_id, word_count in sorted_count[:20]:
    print(dictionary.get(word_id), word_count)

#using tf-idf

tfidf = TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

doc = corpus[4]
pnpweights = tfidf[doc]

sorted_tfidf = sorted(pnpweights, key=lambda w: w[1], reverse=True)

for term_id, weight in sorted_tfidf[:20]:
    print(dictionary.get(term_id), weight)

# LDA analysis - BOW

austen_lda_bow = LdaMulticore(corpus, num_topics = 6, id2word = dictionary,
                                        passes = 10, workers = 2)

for idx, topic in austen_lda_bow.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

# LDA - tfidf

austen_lda_tfidf = LdaMulticore(corpus_tfidf, num_topics = 6, id2word = dictionary,
                                passes = 10, workers = 2)
coherence_tfidf = CoherenceModel(model=austen_lda_tfidf, dictionary=dictionary, corpus = corpus, coherence='u_mass')

for idx, topic in austen_lda_tfidf.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic)) #no useful weights

austen_lda_tfidf = LdaMulticore(corpus_tfidf, num_topics = 2, id2word = dictionary,
                                passes = 10, workers = 2)
for idx, topic in austen_lda_tfidf.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

# find ideal topic number with LDA tfidf

def lda_coherence(dictionary, corpus, limit, start=2, step=3):
    #compute coherence score of models with many topic counts
    #returns list of LDA models & coherence values
    coherence_values = []
    model_list = []

    for num_topics in range(start, limit, step):
        model = LdaMulticore(corpus, num_topics = num_topics, id2word = dictionary)
        model_list.append(model)
        coherence_mod = CoherenceModel(model = model, corpus=corpus, dictionary=dictionary, coherence='u_mass')
        coherence_values.append(coherence_mod.get_coherence())

    return model_list, coherence_values

model_list, coherence_values = lda_coherence(dictionary, corpus, limit=30, step=2)

limit=30;start=2;step=2
x = range(start, limit, step)
plt.plot(x,coherence_values)
plt.xlabel("No. topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"),loc='best')
plt.show()
# lowest topic number, 2, is best. Novels indistinguishable from one another??




# word2vec to T-SNE
# create sentence corpus via flatten

novelsent = [emsplit_sentences, mansplit_sentences, northsplit_sentences, persplit_sentences, pridesplit_sentences,
             sensplit_sentences]

sentences = [sentence for novel in novelsent for sentence in novel]

austen_w2v = Word2Vec(sentences, size=100, window=5, min_count=10, workers=4, sg=0)

austen_w2v_20 = Word2Vec(sentences, size=100, window=5, min_count=20, workers=4, sg=0)

austen_w2v.wv.most_similar("poor")
austen_w2v.similar_by_word("poor")

austen_w2v.save("./output/austen_w2v.model")

# word corpus to matrix (co-occurrence), then export for Gephi network graph

mat_corp = corpus2csc(corpus)
mat_corpdf = pd.DataFrame(mat_corp.toarray())

mat_corpdf.to_csv("./output/mat_corp.csv")

# tsne to matrix for network graph

tsne_labels = []
tsne_tokens = []

for word in austen_w2v_20.wv.vocab:
    tsne_tokens.append(austen_w2v_20[word])
    tsne_labels.append(word)

tsne_model = TSNE(perplexity=40, n_components=2, init='pca', random_state=748)
new_values = tsne_model.fit_transform(tsne_tokens)


# bigram and trigram models from original sentences

originals = [persent, mansent, northsent, persent, pridesent, sensent]



bigrams = Phrases(originals, min_count=5)
biphraser = Phraser(bigrams)
tokens_ = biphraser[sentences]
list(tokens_[:5])

bicounter = Counter()
for key in bigrams.vocab.keys():
    if type(key) == 'bytes':
        if len(key.decode('utf8').split("_")) > 1:
            bicounter[key] += bigrams.vocab[key]
        else:
            if len(key.split("_")) > 1:
                bicounter[key] += bigrams.vocab[key]

bicounter.most_common(20)

trigrams = TrigramCollocationFinder.from_words(corpus)
