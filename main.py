#%% import packages
import sys

from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle
from adjustText import adjust_text
from scipy.interpolate import interp1d


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

def sim_onedim(dim, rangelow = 1850, rangehigh = 2000, rangestep = 10):

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

    # the trendline
    x = data['year'].tolist()
    y = data[dim].tolist()

    fun = interp1d(x, y, kind='cubic')

    xnew = np.linspace(rangelow, (rangehigh-10), 100)

    plt.plot(xnew, fun(xnew), '-', x, y, 'o')

    # show plot
    plt.title(dim)
    plt.show()
    plt.close()



# define similarity function for one term (Cosine_distance = 1 - cosine_similarity)

def sim_oneterm(term, rangelow = 1850, rangehigh = 2000, rangestep = 10):

    d = []

    for model, year in models_all.items():
        if year in range(rangelow, rangehigh, rangestep):
            d.append(
                {
                    "year": year,
                    term: model.n_similarity(keywords['work'], [term])
                }
            )

    data = pd.DataFrame(d)

    # the trendline
    x = data['year'].tolist()
    y = data[term].tolist()

    fun = interp1d(x, y, kind='cubic')

    xnew = np.linspace(rangelow, (rangehigh-10), 100)

    plt.plot(xnew, fun(xnew), '-', x, y, 'o')

    # show plot
    plt.title(term)
    plt.show()
    plt.close()




# define similarity function for two dimensions (Cosine_distance = 1 - cosine_similarity)

def sim_twodim(dim1, dim2, diff = False, rangelow = 1850, rangehigh = 2000, rangestep = 10):

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

    # the trendline

    if diff==True:
        x = data['year'].tolist()
        y = data['diff'].tolist()

        fun = interp1d(x, y, kind='cubic')
        xnew = np.linspace(rangelow, (rangehigh-10), 100)

        plt.plot(xnew, fun(xnew), "-b")

        plt.title(dim1 + "-" + dim2)
        plt.show()
        plt.close()

    else:
        x = data['year'].tolist()
        y1 = data[dim1].tolist()
        y2 = data[dim2].tolist()

        fun1 = interp1d(x, y1, kind='cubic')
        fun2 = interp1d(x, y2, kind='cubic')

        xnew = np.linspace(rangelow, (rangehigh-10), 100)

        plt.plot(xnew, fun1(xnew), "-b", label=dim1)
        plt.plot(x, y1, 'bo')
        plt.plot(xnew, fun2(xnew), "-r", label=dim2)
        plt.plot(x, y2, 'ro')

        plt.legend(loc="best")

        #show plot
        plt.title(dim1 + "&" + dim2)
        plt.show()
        plt.close()









def sim_threedim(dim1, dim2, dim3, trend = 3, rangelow = 1850, rangehigh = 2000, rangestep = 10):

    d = []

    for model, year in models_all.items():
        if year in range(rangelow, rangehigh, rangestep):
            d.append(
                {
                    "year": year,
                    dim1: model.n_similarity(keywords['work'], keywords[dim1]),
                    dim2: model.n_similarity(keywords['work'], keywords[dim2]),
                    dim3: model.n_similarity(keywords['work'], keywords[dim3])
                }
            )

    data = pd.DataFrame(d)

    # the trendline

    if trend == 3:

        x = data['year'].tolist()
        y1 = data[dim1].tolist()
        y2 = data[dim2].tolist()
        y3 = data[dim3].tolist()

        fun1 = interp1d(x, y1, kind='cubic')
        fun2 = interp1d(x, y2, kind='cubic')
        fun3 = interp1d(x, y3, kind='cubic')


        xnew = np.linspace(rangelow, (rangehigh - 10), 100)

        plt.plot(xnew, fun1(xnew), "-b", label=dim1)
        plt.plot(x, y1, 'bo')
        plt.plot(xnew, fun2(xnew), "-r", label=dim2)
        plt.plot(x, y2, 'ro')
        plt.plot(xnew, fun3(xnew), "-g", label=dim3)
        plt.plot(x, y3, 'go')


    elif trend == 2:

        z1 = np.polyfit(data['year'], data[dim1], 2)
        z2 = np.polyfit(data['year'], data[dim2], 2)
        z3 = np.polyfit(data['year'], data[dim3], 2)

        p1 = np.poly1d(z1)
        p2 = np.poly1d(z2)
        p3 = np.poly1d(z3)

        plt.plot(data['year'], p1(data['year']), "r--", label=dim1)
        plt.plot(data['year'], p2(data['year']), "b--", label=dim2)
        plt.plot(data['year'], p3(data['year']), "g--", label=dim3)

    elif trend == 1:

        z1 = np.polyfit(data['year'], data[dim1], 1)
        z2 = np.polyfit(data['year'], data[dim2], 1)
        z3 = np.polyfit(data['year'], data[dim3], 1)

        p1 = np.poly1d(z1)
        p2 = np.poly1d(z2)
        p3 = np.poly1d(z3)

        plt.plot(data['year'], p1(data['year']), "r--", label=dim1)
        plt.plot(data['year'], p2(data['year']), "b--", label=dim2)
        plt.plot(data['year'], p3(data['year']), "g--", label=dim3)


    #show plot
    plt.legend(loc="best")
    plt.title(dim1 + "&" + dim2 + "&" + dim3)
    plt.show()
    plt.close()






# define similarity function for occupations with gender (Cosine_distance = 1 - cosine_similarity)

def sim_occs(*occs):

    diffdata = pd.DataFrame(models_all.values())
    diffdata.rename(columns={0: 'year'}, inplace=True)

    for occ in occs:
        d = []
        for model, year in models_all.items():
            if year in range(1800, 2000, 10):
                d.append(
                    {
                        "year": year,
                        'male': model.n_similarity(keywords['male'], [occ]),
                        'female': model.n_similarity(keywords['female'], [occ])
                    }
                )
        data = pd.DataFrame(d)
        data['diff'] = data['male'] - data['female']
        diffdata[occ] = data['diff']

    # lineplot
    for occ in occs:
        plt.plot(diffdata['year'], diffdata[occ], label=occ)

    # show plot
    plt.title('male-female connotation by occs')
    plt.legend(loc="best")
    plt.axhline(0)
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
            if model[keyword].all() == embeddings1840['biology'].all():
                sys.stderr.write("Term not found in all indicated time periods")
                sys.exit(1)
            else:
                tempsim = model.most_similar(keyword, topn=7)
                for term, vector in tempsim:
                    sim_words.append(term)

    sim_words = list(set(sim_words))

    # get vectors of similar words in most recent embedding (1990)
    sim_vectors1990 = np.array([embeddings1990[w] for w in sim_words])

    # get vectors of keyword in all periods

    keyword_vectors = np.zeros(shape=(0,300))

    for model, year in models_all.items():
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
    for model, year in models_all.items():
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



#%%  most similar terms

for x, y in models_all.items():
    print(y)
    print(x.most_similar("office"))

# --> work has a different meaning before 1850

#%% visualize word embeddings over time

similarplot("gay", 1810, 2000, 60) # PCA
similarplot2("work", 1810, 2000, 60) # T-SNE
similarplot3("work", 1810, 2000, 60, export=False) # PCA mit keyword als passiv







#%% association of work with different dimension

# set up dictionary and define "work"

keywords = {}

keywords['work'] = [
    "work", "works", "worked", "working","job", "jobs",
    "career", "careers",
    "profession", "professions", "professional",
    "occupation", "occupations",
    "employment", "employed",
    "labor", "labors"
]


# smith: toil (einzelne Begriffe anzeigen?)

keywords['toil'] = [
    "hard", "struggle", "toil", "trouble", "suffer", "endure", "arduous", "strenuous"
]
sim_onedim('toil', 1850)

keywords['hard'] = ['hard']
keywords['struggle'] = ['struggle']
keywords['toil'] = ['toil']
keywords['trouble'] = ['trouble']
keywords['suffer'] = ['suffer']
keywords['endure'] = ['endure']
keywords['arduous'] = ['arduous']
keywords['strenuous'] = ['strenuous']

sim_threedim('hard', 'struggle', 'toil', trend=2)
sim_threedim('trouble', 'suffer', 'endure', trend=2)
sim_threedim('arduous', 'strenuous', 'toil', trend=2)



keywords['fun'] = ["fun", "enjoy", "pleasant"]

keywords['leisure'] = ["leisure", "ease", "rest"]
sim_onedim('leisure')

sim_twodim('toil', 'fun', 1850)

keywords['emotion'] = [
    "pleasant", "interesting", "boring", "fulfilling", "meaningful", "meaningless",
    "hard", "struggle", "toil", "trouble", "suffer", "endure", "arduous", "strenuous"
]
sim_onedim('emotion', 1850)

keywords['commodity'] = [
    "market", "exchange", "trade", "hire", "rent"
]
sim_onedim('commodity', 1850) # nicht sehr spannend

sim_oneterm('duty') # auch teil von "patriot"

sim_oneterm('pleasant')











# marx: alienation (extrinsic vs. intrinsic)

keywords['mat'] = [
                      "earn", "earns", "earning", "earnings",
                      "wage", "wages", "salary", "income", "remuneration", "secure", "pay"
]
keywords['postmat'] = ["interesting", "boring", "fulfilling", "meaningful", "meaningless"]

sim_onedim('mat', 1850)
sim_onedim('postmat', 1850)
sim_twodim('mat', 'postmat')

keywords['useful'] = ["useful", "society"]
sim_onedim('useful', 1850)


keywords['status'] = [
    "prestigious", "honorable", "esteemed", "influential", "reputable", "distinguished",
    "eminent", "illustrious", "renowned", "acclaimed"
]
sim_onedim('status', 1850)

keywords['social'] = ["colleague", "friend", "people"]
sim_onedim('social', 1850)











# weber: wealth & religion

keywords['rich'] = ["wealth", "wealthy", "rich", "affluence", "affluent"]
keywords['poor'] = ["poor", "poverty", "impoverished", "destitute", "needy"]
sim_twodim('rich', 'poor')

keywords['affluence'] = keywords['rich'] + keywords['poor']
sim_onedim('affluence', 1850) # --> Piketty!

keywords['success'] = ["success", "succeed", "failure", "fail"]
sim_onedim('success', 1850)

keywords['religion'] = ["redemption", "salvation"]
sim_onedim('religion', 1850)

keywords['vocation'] = ["vocation", "calling", "meaning", "purpose"]
sim_onedim('vocation', 1850)

keywords['moral'] = [
    'good', 'evil', 'moral', 'immoral', 'good', 'bad', 'honest', 'dishonest',
    'virtuous', 'sinful', 'virtue', 'vice'
]
sim_onedim('moral', 1850) # --> Piketty!



# Weber: was läuft bei WK?

keywords['patriot'] = ["duty", "country", "patriot", "fatherland", "home"]
sim_onedim('patriot', 1850)




# VALIDATION

# faktisch

# connotation von Arbeit mit mann/frau

keywords['male'] = ["male", "man", "boy", "brother", "he", "him", "his", "son"]
keywords['female'] = ["female", "woman", "girl", "sister", "she", "her", "hers", "daughter"]

sim_onedim('male', 1850)
sim_onedim('female', 1850)
sim_twodim('male', 'female', diff=False)
sim_twodim('male', 'female', diff=True)

# typische arbeitsgeräte für verschiedene epochen

keywords['plow'] = ['plow']
keywords['telephone'] = ['telephone']
keywords['computer'] = ['computer']

sim_onedim('plow')
sim_onedim('telephone')
sim_onedim('computer')

sim_threedim('plow', 'telephone', 'computer', trend=2)


# historisches wachstum von sektoren

keywords['sector1'] = ["agriculture", "farming", "logging", "fishing", "forestry", "mining"]

keywords['sector2'] = ["manufacturing", "textile", "car", "handicraft"]

keywords['sector3'] = ["service", "social", "information", "advice", "access"]

sim_threedim('sector1', 'sector2', 'sector3', trend=2)

# typisch weibliche/männliche Berufe

sim_occs('mechanic', 'carpenter', 'engineer', 'nurse', "dancer", "housekeeper")

# SEMANTIC DRIFT

# housework --> work

keywords['housework'] = ["housework", "household"]
sim_onedim('housework', 1850)

# beziehungsarbeit

keywords['relations'] = ["relationship"]
sim_onedim('relations', 1850)

# DISKURS: Arbeiterbewegung

keywords['politics'] = ["party", "politics", "movement", "election"]
sim_onedim('politics', 1850)



























#%% test area




# stacked area chart for sectors

d = []

for model, year in models_all.items():
    if year in range(1850, 2000, 10):
        d.append(
            {
                "year": year,
                "sector1": model.n_similarity(keywords['work'], keywords["sector1"]),
                "sector2": model.n_similarity(keywords['work'], keywords["sector2"]),
                "sector3": model.n_similarity(keywords['work'], keywords["sector3"])
            }
        )

data = pd.DataFrame(d)

data['total'] = data['sector1'] + data['sector2'] + data['sector3']

data['sector1'] = data['sector1'].divide(data['total'])
data['sector2'] = data['sector2'].divide(data['total'])
data['sector3'] = data['sector3'].divide(data['total'])

plt.stackplot(data["year"], data["sector1"], data["sector2"], data["sector3"], labels=['sector1','sector2','sector3'])
plt.legend(loc='upper left')
plt.margins(0,0)
plt.title('100 % stacked area chart')
plt.show()















