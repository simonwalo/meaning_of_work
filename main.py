#%% import packages
import sys
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
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



# define similarity function for n dimensions

def simdim(*dims, trend = 3, diff = False, rangelow = 1850, rangehigh = 2000, rangestep = 10):

    data = pd.DataFrame()
    data['year'] = range(rangelow, rangehigh, rangestep)

    for dim in dims:
        d = []
        for model, year in models_all.items():
            if year in range(rangelow, rangehigh, rangestep):
                d.append(model.n_similarity(keywords['work'], keywords[dim]))
        data[dim] = d

    if len(dims)==2:
        data['diff'] = data.iloc[:, 1]-data.iloc[:, 2]

    # the trendline

    if trend == 3:

        x = data['year'].tolist()
        xnew = np.linspace(rangelow, (rangehigh - 10), 100)

        n = len(dims)
        colors = iter(cm.rainbow(np.linspace(0, 1, n)))

        if diff == True:
            y = data['diff'].tolist()
            fun = interp1d(x, y, kind='cubic')
            plt.plot(xnew, fun(xnew), "-", label='diff')
            plt.plot(x, y, 'o')

        elif diff == False:
            for dim in dims:
                y = data[dim].tolist()
                fun = interp1d(x, y, kind='cubic')
                color=next(colors)
                plt.plot(xnew, fun(xnew), "-", color=color, label=dim)
                plt.plot(x, y, 'o', color=color)


    elif trend == 2:

        n = len(dims)
        colors = iter(cm.rainbow(np.linspace(0, 1, n)))

        if diff == True:
            z = np.polyfit(data['year'], data['diff'], 2)
            p = np.poly1d(z)
            plt.plot(data['year'], p(data['year']), label='diff')

        elif diff == False:
            for dim in dims:
                z = np.polyfit(data['year'], data[dim], 2)
                p = np.poly1d(z)
                color = next(colors)
                plt.plot(data['year'], p(data['year']), color=color, label=dim)


    elif trend == 1:

        n = len(dims)
        colors = iter(cm.rainbow(np.linspace(0, 1, n)))

        if diff == True:
            z = np.polyfit(data['year'], data['diff'], 1)
            p = np.poly1d(z)
            plt.plot(data['year'], p(data['year']), label='diff')

        elif diff == False:
            for dim in dims:
                z = np.polyfit(data['year'], data[dim], 1)
                p = np.poly1d(z)
                color = next(colors)
                plt.plot(data['year'], p(data['year']), color=color, label=dim)

    #show plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title("Association of dimension(s) with work")
    plt.tight_layout()
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








##### SEMANTIC CHANGE OF TERM ######

# define function to visualize semantic change (PCA mit keyword als passive projektionen)


def semchange(keyword, rangelow = 1800, rangehigh = 2000, rangestep = 10, export = False):

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
    print(x.most_similar("work"))

# --> work has a different meaning before 1850

#%% visualize word embeddings over time

semchange("gay", 1810, 2000, 60, export=False) # PCA mit keyword als passiv







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
    "hard", "struggle", "toil", "trouble", "suffer", "endure", "arduous", "strenuous", "grind"
]
simdim('toil')

keywords['leisure'] = ["leisure", "ease", "rest", "recreation", "relaxation", "freedom"]
simdim('leisure')

simdim('toil', 'leisure')

keywords['hard'] = ['hard']
keywords['struggle'] = ['struggle']
keywords['toil'] = ['toil']
keywords['trouble'] = ['trouble']
keywords['suffer'] = ['suffer']
keywords['endure'] = ['endure']
keywords['arduous'] = ['arduous']
keywords['strenuous'] = ['strenuous']

simdim('hard', 'struggle', 'toil', 'trouble', 'suffer', 'endure', 'arduous', 'strenuous', trend=3)






keywords['fun'] = ["fun", "enjoy", "pleasant"]
simdim('fun')



keywords['emotion'] = [
    "pleasant", "interesting", "boring", "fulfilling", "meaningful", "meaningless",
    "hard", "struggle", "toil", "trouble", "suffer", "endure", "arduous", "strenuous"
]
simdim('emotion', 1850)

keywords['commodity'] = [
    "market", "exchange", "trade", "hire", "rent"
]
simdim('commodity', 1850) # nicht sehr spannend

sim_oneterm('duty') # auch teil von "patriot"

sim_oneterm('pleasant')











# marx: alienation (extrinsic vs. intrinsic)

keywords['mat'] = [
                      "earn", "earns", "earning", "earnings",
                      "wage", "wages", "salary", "income", "remuneration", "secure", "pay"
]
keywords['postmat'] = ["interesting", "boring", "fulfilling", "meaningful", "meaningless", "useful", "useless"]

simdim('mat', 1850)
simdim('postmat', 1850)
simdim('mat', 'postmat')

keywords['useful'] = ["useful", "society"]
simdim('useful', 1850)


keywords['status'] = [
    "prestigious", "honorable", "esteemed", "influential", "reputable", "distinguished",
    "eminent", "illustrious", "renowned", "acclaimed"
]
simdim('status', 1850)

keywords['social'] = ["colleague", "colleague", "friend", "friends", "people"]
simdim('social', 1850)

simdim('mat', 'postmat', 'status', 'social')













# weber: wealth & religion

keywords['rich'] = ["wealth", "wealthy", "rich", "affluence", "affluent"]
keywords['poor'] = ["poor", "poverty", "impoverished", "destitute", "needy"]
simdim('rich', 'poor')

keywords['affluence'] = keywords['rich'] + keywords['poor']
simdim('affluence', 1850) # --> Piketty!

keywords['success'] = ["success", "succeed", "failure", "fail"]
simdim('success', 1850)

keywords['religion'] = ["redemption", "salvation"]
simdim('religion', 1850)

keywords['vocation'] = ["vocation", "calling", "meaning", "purpose"]
simdim('vocation', 1850)

keywords['moral'] = [
    'good', 'evil', 'moral', 'immoral', 'good', 'bad', 'honest', 'dishonest',
    'virtuous', 'sinful', 'virtue', 'vice'
]
simdim('moral', 1850) # --> Piketty!



# Weber: was läuft bei WK?

keywords['patriot'] = ["duty", "country", "patriot", "fatherland", "home"]
simdim('patriot', 1850)




# VALIDATION

# faktisch

# connotation von Arbeit mit mann/frau

keywords['male'] = ["male", "man", "boy", "brother", "he", "him", "his", "son"]
keywords['female'] = ["female", "woman", "girl", "sister", "she", "her", "hers", "daughter"]

simdim('male', 1850)
simdim('female', 1850)
simdim('male', 'female', diff=False)
simdim('male', 'female', diff=True)

# typische arbeitsgeräte für verschiedene epochen

keywords['plow'] = ['plow']
keywords['telephone'] = ['telephone']
keywords['computer'] = ['computer']

simdim('plow')
simdim('telephone')
simdim('computer')

simdim('plow', 'telephone', 'computer', trend=2)


# historisches wachstum von sektoren

keywords['sector1'] = ["agriculture", "farming", "logging", "fishing", "forestry", "mining"]

keywords['sector2'] = ["manufacturing", "textile", "car", "handicraft"]

keywords['sector3'] = ["service", "social", "information", "advice", "access"]

simdim('sector1', 'sector2', 'sector3', trend=2)

# typisch weibliche/männliche Berufe

keywords['male'] = ["male", "man", "boy", "brother", "he", "him", "his", "son"]
keywords['female'] = ["female", "woman", "girl", "sister", "she", "her", "hers", "daughter"]

sim_occs('mechanic', 'carpenter', 'engineer', 'nurse', "dancer", "housekeeper")

# SEMANTIC DRIFT

# housework --> work

keywords['housework'] = ["housework", "household"]
simdim('housework', 1850)

# beziehungsarbeit

keywords['relations'] = ["relationship"]
simdim('relations', 1850)

# DISKURS: Arbeiterbewegung

keywords['politics'] = ["party", "politics", "movement", "election"]
simdim('politics', 1850)



























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

















