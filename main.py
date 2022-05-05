#%% import packages

from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle


#%% load all data in C text format

embeddings1800 = KeyedVectors.load_word2vec_format("./data/english_all_sgns_gensim_1800.txt", binary=False, no_header=True)
embeddings1810 = KeyedVectors.load_word2vec_format("./data/english_all_sgns_gensim_1810.txt", binary=False, no_header=True)
embeddings1820 = KeyedVectors.load_word2vec_format("./data/english_all_sgns_gensim_1820.txt", binary=False, no_header=True)
embeddings1830 = KeyedVectors.load_word2vec_format("./data/english_all_sgns_gensim_1830.txt", binary=False, no_header=True)
embeddings1840 = KeyedVectors.load_word2vec_format("./data/english_all_sgns_gensim_1840.txt", binary=False, no_header=True)
embeddings1850 = KeyedVectors.load_word2vec_format("./data/english_all_sgns_gensim_1850.txt", binary=False, no_header=True)
embeddings1860 = KeyedVectors.load_word2vec_format("./data/english_all_sgns_gensim_1860.txt", binary=False, no_header=True)
embeddings1870 = KeyedVectors.load_word2vec_format("./data/english_all_sgns_gensim_1870.txt", binary=False, no_header=True)
embeddings1880 = KeyedVectors.load_word2vec_format("./data/english_all_sgns_gensim_1880.txt", binary=False, no_header=True)
embeddings1890 = KeyedVectors.load_word2vec_format("./data/english_all_sgns_gensim_1890.txt", binary=False, no_header=True)
embeddings1900 = KeyedVectors.load_word2vec_format("./data/english_all_sgns_gensim_1900.txt", binary=False, no_header=True)
embeddings1910 = KeyedVectors.load_word2vec_format("./data/english_all_sgns_gensim_1910.txt", binary=False, no_header=True)
embeddings1920 = KeyedVectors.load_word2vec_format("./data/english_all_sgns_gensim_1920.txt", binary=False, no_header=True)
embeddings1930 = KeyedVectors.load_word2vec_format("./data/english_all_sgns_gensim_1930.txt", binary=False, no_header=True)
embeddings1940 = KeyedVectors.load_word2vec_format("./data/english_all_sgns_gensim_1940.txt", binary=False, no_header=True)
embeddings1950 = KeyedVectors.load_word2vec_format("./data/english_all_sgns_gensim_1950.txt", binary=False, no_header=True)
embeddings1960 = KeyedVectors.load_word2vec_format("./data/english_all_sgns_gensim_1960.txt", binary=False, no_header=True)
embeddings1970 = KeyedVectors.load_word2vec_format("./data/english_all_sgns_gensim_1970.txt", binary=False, no_header=True)
embeddings1980 = KeyedVectors.load_word2vec_format("./data/english_all_sgns_gensim_1980.txt", binary=False, no_header=True)
embeddings1990 = KeyedVectors.load_word2vec_format("./data/english_all_sgns_gensim_1990.txt", binary=False, no_header=True)



#%% save embeddings in python format (loads faster)

pickle.dump(embeddings1800, open("./data/embeddings1800.pickle", 'wb'))
pickle.dump(embeddings1810, open("./data/embeddings1810.pickle", 'wb'))
pickle.dump(embeddings1820, open("./data/embeddings1820.pickle", 'wb'))
pickle.dump(embeddings1830, open("./data/embeddings1830.pickle", 'wb'))
pickle.dump(embeddings1840, open("./data/embeddings1840.pickle", 'wb'))
pickle.dump(embeddings1850, open("./data/embeddings1850.pickle", 'wb'))
pickle.dump(embeddings1860, open("./data/embeddings1860.pickle", 'wb'))
pickle.dump(embeddings1870, open("./data/embeddings1870.pickle", 'wb'))
pickle.dump(embeddings1880, open("./data/embeddings1880.pickle", 'wb'))
pickle.dump(embeddings1890, open("./data/embeddings1890.pickle", 'wb'))
pickle.dump(embeddings1900, open("./data/embeddings1900.pickle", 'wb'))
pickle.dump(embeddings1910, open("./data/embeddings1910.pickle", 'wb'))
pickle.dump(embeddings1920, open("./data/embeddings1920.pickle", 'wb'))
pickle.dump(embeddings1930, open("./data/embeddings1930.pickle", 'wb'))
pickle.dump(embeddings1940, open("./data/embeddings1940.pickle", 'wb'))
pickle.dump(embeddings1950, open("./data/embeddings1950.pickle", 'wb'))
pickle.dump(embeddings1960, open("./data/embeddings1960.pickle", 'wb'))
pickle.dump(embeddings1970, open("./data/embeddings1970.pickle", 'wb'))
pickle.dump(embeddings1980, open("./data/embeddings1980.pickle", 'wb'))
pickle.dump(embeddings1990, open("./data/embeddings1990.pickle", 'wb'))


#%% load pickled files

embeddings1800 = pickle.load(open("./data/embeddings1800.pickle", 'rb'))
embeddings1810 = pickle.load(open("./data/embeddings1810.pickle", 'rb'))
embeddings1820 = pickle.load(open("./data/embeddings1820.pickle", 'rb'))
embeddings1830 = pickle.load(open("./data/embeddings1830.pickle", 'rb'))
embeddings1840 = pickle.load(open("./data/embeddings1840.pickle", 'rb'))
embeddings1850 = pickle.load(open("./data/embeddings1850.pickle", 'rb'))
embeddings1860 = pickle.load(open("./data/embeddings1860.pickle", 'rb'))
embeddings1870 = pickle.load(open("./data/embeddings1870.pickle", 'rb'))
embeddings1880 = pickle.load(open("./data/embeddings1880.pickle", 'rb'))
embeddings1890 = pickle.load(open("./data/embeddings1890.pickle", 'rb'))
embeddings1900 = pickle.load(open("./data/embeddings1900.pickle", 'rb'))
embeddings1910 = pickle.load(open("./data/embeddings1910.pickle", 'rb'))
embeddings1920 = pickle.load(open("./data/embeddings1920.pickle", 'rb'))
embeddings1930 = pickle.load(open("./data/embeddings1930.pickle", 'rb'))
embeddings1940 = pickle.load(open("./data/embeddings1940.pickle", 'rb'))
embeddings1950 = pickle.load(open("./data/embeddings1950.pickle", 'rb'))
embeddings1960 = pickle.load(open("./data/embeddings1960.pickle", 'rb'))
embeddings1970 = pickle.load(open("./data/embeddings1970.pickle", 'rb'))
embeddings1980 = pickle.load(open("./data/embeddings1980.pickle", 'rb'))
embeddings1990 = pickle.load(open("./data/embeddings1990.pickle", 'rb'))


#%% models dictionary

models_all = {
    embeddings1800: 1800,
    embeddings1810: 1810,
    embeddings1820: 1820,
    embeddings1830: 1830,
    embeddings1840: 1840,
    embeddings1850: 1850,
    embeddings1860: 1860,
    embeddings1870: 1870,
    embeddings1880: 1880,
    embeddings1890: 1890,
    embeddings1900: 1900,
    embeddings1910: 1910,
    embeddings1920: 1920,
    embeddings1930: 1930,
    embeddings1940: 1940,
    embeddings1950: 1950,
    embeddings1960: 1960,
    embeddings1970: 1970,
    embeddings1980: 1980,
    embeddings1990: 1990
}





#%% define functions


# define similarity function for one dimension (Cosine_distance = 1 - cosine_similarity)

def sim_onedim(dim, rangelow = 1800, rangehigh = 2000, rangestep = 10):

    d = []

    for model, year in models_all.items():
        if year in range(rangelow, rangehigh, rangestep):
            d.append(
                {
                    "year": year,
                    dim: model.n_similarity(keywords['work'], keywords[dim])
                }
            )

    data = pd.DataFrame(d)

    # lineplot
    plt.plot(data['year'], data[dim])

    # the trendline
    z = np.polyfit(data['year'], data[dim], 1)
    p = np.poly1d(z)
    plt.plot(data['year'], p(data['year']), "r--")

    # show plot
    plt.title(dim)
    plt.show()
    plt.close()



# define similarity function for two dimensions (Cosine_distance = 1 - cosine_similarity)

def sim_twodim(dim1, dim2, rangelow = 1800, rangehigh = 2000, rangestep = 10):

    d = []

    for model, year in models_all.items():
        if year in range(rangelow, rangehigh, rangestep):
            d.append(
                {
                    "year": year,
                    dim1: model.n_similarity(keywords['work'], keywords[dim1]),
                    dim2: model.n_similarity(keywords['work'], keywords[dim2])
                }
            )

    data = pd.DataFrame(d)
    data['diff'] = data[dim1] - data[dim2]

    #lineplot
    plt.plot(data['year'], data['diff'])

    #the trendline
    z = np.polyfit(data['year'], data['diff'], 1)
    p = np.poly1d(z)
    plt.plot(data['year'],p(data['year']),"r--")

    #show plot
    plt.title(dim1 + "-" + dim2)
    plt.show()
    plt.close()


# define function to visualize semantic change (PCA)

def similarplot(keyword, rangelow = 1800, rangehigh = 2000, rangestep = 10):

    # get list of all similar words from different periods

    sim_words = []

    for model, year in models_all.items():
        if year in range(rangelow, rangehigh, rangestep):
            tempsim = model.most_similar(keyword, topn=7)
            for term, vector in tempsim:
                sim_words.append(term)

    sim_words = list(set(sim_words))

    # get vectors of similar words in most recent embedding (1990)
    sim_vectors1990 = np.array([embeddings1990[w] for w in sim_words])

    # get vectors of keyword in all periods and add them to vectors of similar words

    allvectors = sim_vectors1990

    for model, year in models_all.items():
        if year in range(rangelow, rangehigh, rangestep):
            keyword_vectors = np.array([model[keyword]])
            allvectors = np.append(allvectors, keyword_vectors, axis=0)

    # reduce dimensions of vectors
    pca = PCA(n_components=2)
    two_dim = pca.fit_transform(allvectors)

    # get labels
    labels = sim_words
    for model, year in models_all.items():
        if year in range(rangelow, rangehigh, rangestep):
            labels.append(keyword + str(year))

    #plot results
    plt.scatter(two_dim[:, 0], two_dim[:, 1])

    for i in range(len(sim_words)):
        plt.text(x=two_dim[i, 0], y=two_dim[i, 1], s=labels[i])

    plt.show()
    plt.close()




# define function to visualize semantic change (T-SNE)

def similarplot2(keyword, rangelow = 1800, rangehigh = 2000, rangestep = 10):

    # get list of all similar words from different periods

    sim_words = []

    for model, year in models_all.items():
        if year in range(rangelow, rangehigh, rangestep):
            tempsim = model.most_similar(keyword, topn=7)
            for term, vector in tempsim:
                sim_words.append(term)

    sim_words = list(set(sim_words))

    # get vectors of similar words in most recent embedding (1990)
    sim_vectors1990 = np.array([embeddings1990[w] for w in sim_words])

    # get vectors of keyword in all periods and add them to vectors of similar words

    allvectors = sim_vectors1990

    for model, year in models_all.items():
        if year in range(rangelow, rangehigh, rangestep):
            keyword_vectors = np.array([model[keyword]])
            allvectors = np.append(allvectors, keyword_vectors, axis=0)

    # reduce dimensions of vectors
    tsne = TSNE(n_components=2, learning_rate="auto", init="pca", perplexity=5)
    two_dim = tsne.fit_transform(allvectors)

    # get labels
    labels = sim_words
    for model, year in models_all.items():
        if year in range(rangelow, rangehigh, rangestep):
            labels.append(keyword + str(year))

    #plot results
    plt.scatter(two_dim[:, 0], two_dim[:, 1])

    for i in range(len(sim_words)):
        plt.text(x=two_dim[i, 0], y=two_dim[i, 1], s=labels[i])

    plt.show()
    plt.close()





# define function to visualize semantic change (PCA mit keyword als passive projektionen)

def similarplot3(keyword, rangelow = 1800, rangehigh = 2000, rangestep = 10, export = False):

    # get list of all similar words from different periods

    sim_words = []

    for model, year in models_all.items():
        if year in range(rangelow, rangehigh, rangestep):
            tempsim = model.most_similar(keyword, topn=7)
            for term, vector in tempsim:
                sim_words.append(term)

    sim_words = list(set(sim_words))

    # get vectors of similar words in most recent embedding (1990)
    sim_vectors1990 = np.array([embeddings1990[w] for w in sim_words])

    # get vectors of keyword in all periods and add them to vectors of similar words

    allvectors = sim_vectors1990

    for model, year in models_all.items():
        if year in range(rangelow, rangehigh, rangestep):
            keyword_vectors = np.array([model[keyword]])
            allvectors = np.append(allvectors, keyword_vectors, axis=0)

    # "train" PCA model with only similar words
    pca = PCA(n_components=2)
    pca_model = pca.fit(sim_vectors1990)
    two_dim = pca.transform(allvectors)

    # get labels
    labels = sim_words
    for model, year in models_all.items():
        if year in range(rangelow, rangehigh, rangestep):
            labels.append(keyword + str(year))

    #plot results
    plt.scatter(two_dim[:, 0], two_dim[:, 1])

    for i in range(len(sim_words)):
        plt.text(x=two_dim[i, 0], y=two_dim[i, 1], s=labels[i])

    if export == True:
        plt.savefig('./output/' + keyword + '_' + str(rangelow) + '-' + str(rangehigh-10) + '.png')
    plt.show()
    plt.close()



#%%  most similar terms

for x, y in models_all.items():
    print(y)
    print(x.most_similar("work"))

# --> work has a different meaning before 1850

#%% visualize word embeddings over time

similarplot("work", 1810, 2000, 30) # PCA
similarplot2("work", 1810, 2000, 30) # T-SNE
similarplot3("work", 1810, 2000, 60, export=False) # PCA mit keyword als passiv


#%% keywords

keywords = {
    "work":
        ["work", "works", "worked", "working",
         "job", "jobs",
         "career", "careers",
         "profession", "professions", "professional",
         "occupation", "occupations",
         "employment", "employed",
         "labor", "labors"],

    "male":
        ["male", "man", "boy", "brother", "he", "him", "his", "son"],
    "female":
        ["female", "woman", "girl", "sister", "she", "her", "hers", "daughter"],

    "religion":
        ["redemption", "salvation"],

    "mat":
        ["earn", "earns", "earning", "earnings",
         "wage", "wages", "salary", "income", "remuneration", "secure", "pay"],
    'postmat':
        ["interesting", "boring", "fulfilling", "meaningful", "meaningless"],

    'rich': ["wealth", "wealthy", "rich", "affluence", "affluent"],
    'poor': ["poor", "poverty", "impoverished", "destitute", "needy"],
    'affluence': ["wealth", "wealthy", "rich", "affluence", "affluent",
                  "poor", "poverty", "impoverished", "destitute", "needy"],

    'hard': ["hard", "effort", "strive", "push", "struggle"],

    'politics': ["party", "politics", "movement", "election"],

    'moral': ['good', 'evil', 'moral', 'immoral', 'good', 'bad', 'honest', 'dishonest',
              'virtuous', 'sinful', 'virtue', 'vice'],

    'vocation': ["vocation", "calling", "meaning", "purpose"],

    'success': ["success", "succeed", "failure", "fail"],

    'housework': ["housework", "household"],

    'emotion': ["emotion", "emotional"],

    'relations': ["relationship"],

    'status': ["prestigious", "honorable", "esteemed", "influential", "reputable", "distinguished",
               "eminent", "illustrious", "renowned", "acclaimed"],

    'patriot': ["duty", "country", "patriot", "fatherland", "home"],

    'commodity': ["market", "exchange", "trade", "hire", "rent"]
}


#%% association of work with different dimension

sim_onedim('religion', 1850)

sim_onedim('male', 1850)
sim_onedim('female', 1850)
sim_twodim('male', 'female', 1850)

sim_onedim('mat')
sim_onedim('postmat', 1850)
sim_twodim('mat', 'postmat', 1850)

sim_onedim('rich')
sim_onedim('poor')
sim_twodim('rich', 'poor')

sim_onedim('affluence', 1850)

sim_onedim('hard', 1850)

sim_onedim('politics', 1850)

sim_onedim('moral', 1850)

sim_onedim('vocation', 1850)

sim_onedim('success', 1850)

sim_onedim('housework', 1850)

sim_onedim('emotion', 1850)

sim_onedim('relations', 1850)

sim_onedim('status', 1850)

sim_onedim('patriot', 1850)

sim_onedim('commodity', 1850)










