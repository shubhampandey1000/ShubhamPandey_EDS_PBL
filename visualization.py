from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from bokeh.plotting import figure, show, output_file, save
from bokeh.models import value, LabelSet, ColumnDataSource
output_file("./output/austen_tsne.html", title = "austen tsne")

vocab = list(austen_w2v.wv.vocab)
X = austen_w2v[vocab]

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)

tsne_df = pd.DataFrame(X_tsne, index = vocab, columns = ['x','y'])

fig = sns.scatterplot(tsne_df['x'],tsne_df['y'])
for word, pos in tsne_df.iterrows():
    fig.annotate(word, pos)

plt.show()

plt.ylim(40,80)
plt.xlim(-25,50)

#function to make tsne model of specific words
def tsne_closest(model, word):
    arr = np.empty((0,100), dtype='f')
    word_labels = [word]

    #similar words
    close_words = model.wv.most_similar(word, topn=20)

    #vector for close words
    arr = np.append(arr, np.array([model[word]]),axis=0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)

    #tsne coords
    tsne = TSNE(n_components=2, random_state=748)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]

    #display
    sns.scatterplot(x_coords, y_coords)

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy = (x,y), xytext = (0,0), textcoords='offset points')
    plt.plot(x_coords[0],y_coords[0],'ro')
    plt.show()

tsne_closest(austen_w2v,"lady")

# function to make tsne of all words
def tsne_all(model):
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', random_state=748)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
    zip_labs = zip(labels,x,y)
    list_df = pd.DataFrame(list(zip_labs),columns=['label','x','y'])

    sns.scatterplot(x,y)

    main_chars = ['darcy','elizabeth','bennet','jane','bingley','emma','woodhouse','harriet','knightley','wentworth','fanny','elinor','marianne','edmund', 'catherine', 'james', 'tilney', 'anne', 'wentworth']

    for label, x, y in zip(labels,x,y):
        plt.annotate(label, xy = (x,y), xytext = (0,0), textcoords='offset points')
        if label in main_chars:
            plt.plot(x,y,'ro')
    plt.show()

    return list_df

tsne_df = tsne_all(austen_w2v_20)

# TSNE to bokeh

main_chars = ['darcy','elizabeth','bennet','jane','bingley','emma','woodhouse','harriet','knightley','wentworth','fanny','elinor','marianne','edmund', 'catherine', 'james', 'tilney', 'anne', 'wentworth']

tsne_short = tsne_df[tsne_df.label.isin(main_chars)]

source = ColumnDataSource(dict(
    x = tsne_df['x'],
    y = tsne_df['y'],
    words = tsne_df['label']
))

short_source = ColumnDataSource(dict(
    x = tsne_short['x'],
    y = tsne_short['y'],
    words = tsne_short['label']
))

title = "TSNE graph of words in Austen's major novels"

p = figure(plot_width = 1500, plot_height = 800, title = title, tools = "pan,wheel_zoom,box_zoom,reset,previewsave",
           x_axis_type = None, y_axis_type = None, min_border=1)

p.scatter(x='x',y='y', source=source)
p.scatter(x='x',y='y', source=short_source, color='red', size=5)

labels = LabelSet(x='x',y='y',text='words',level='glyph',
              x_offset=5, y_offset=5, source=source, render_mode='canvas', text_font_size='8pt', text_alpha=0.7)

p.add_layout(labels)

show(p)


