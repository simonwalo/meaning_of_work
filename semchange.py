##### SEMANTIC CHANGE OF TERM ######

# define function to visualize semantic change (PCA mit keyword als passive projektionen)

# import packages
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from adjustText import adjust_text




def semchange(models, keyword, rangelow = 1810, rangehigh = 2000, rangestep = 60, export = False):

    # get list of all similar words from different periods

    sim_words = []

    for year, model in models.items():
        if year in range(rangelow, rangehigh, rangestep):
            if model[keyword].all() == models[1840]['biology'].all():
                sys.stderr.write("Term not found in all indicated time periods")
                sys.exit(1)
            else:
                tempsim = model.most_similar(keyword, topn=7)
                for term, vector in tempsim:
                    sim_words.append(term)

    sim_words = list(set(sim_words))

    # get vectors of similar words in most recent embedding (1990)
    sim_vectors1990 = np.array([models[1990][w] for w in sim_words])

    # get vectors of keyword in all periods

    keyword_vectors = np.zeros(shape=(0,300))

    for year, model in models.items():
        if year in range(rangelow, rangehigh, rangestep):
            temp_keyword_vector = np.array([model[keyword]])
            keyword_vectors = np.append(keyword_vectors, temp_keyword_vector, axis=0)

    # add keyword vectors from all periods to vectors of similar words 1990

    allvectors = np.append(sim_vectors1990, keyword_vectors, axis=0)

    # "train" PCA model with only similar words
    pca = PCA(n_components=2)
    pca.fit(sim_vectors1990)
    two_dim = pca.transform(allvectors)

    # get labels
    labels = sim_words
    for year, model in models.items():
        if year in range(rangelow, rangehigh, rangestep):
            labels.append(keyword + str(year))

    #plot results
    plt.scatter(two_dim[:, 0], two_dim[:, 1])

    texts = [plt.text(x=two_dim[i, 0], y=two_dim[i, 1], s=labels[i]) for i in range(len(sim_words))]
    adjust_text(texts)

    #plot arrow between keywords

    for i in range(-2, -(len(keyword_vectors)+1), -1):
        plt.arrow(two_dim[i,0], two_dim[i,1],
                  two_dim[i+1, 0] - two_dim[i,0], two_dim[i+1, 1] - two_dim[i,1],
                  head_width=0.03, length_includes_head=True)

    if export == True:
        plt.savefig('./output/' + keyword + '_' + str(rangelow) + '-' + str(rangehigh-10) + '.png')
    plt.show()
    plt.close()