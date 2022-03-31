#%% import packages

import gensim
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



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



#%% word lists & preps

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

models = {
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


# define similarity function for one dimension (Cosine_distance = 1 - cosine_similarity)

def sim_onedim(dim):
    d = []

    for x, y in models.items():
        d.append(
            {
                "year": y,
                dim: x.n_similarity(keywords['work'], keywords[dim])
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

def sim_twodim(dim1, dim2):
    d = []

    for x, y in models.items():
        d.append(
            {
                "year": y,
                dim1: x.n_similarity(keywords['work'], keywords[dim1]),
                dim2: x.n_similarity(keywords['work'], keywords[dim2])
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

    'vocation': ["vocation", "calling"],

    'success': ["success", "succeed", "failure", "fail"],

    'housework': ["housework", "household"],

    'emotion': ["emotion", "emotional"],

    'relations': ["relationship"],

    'status': ["prestigious", "honorable", "esteemed", "influential", "reputable", "distinguished",
               "eminent", "illustrious", "renowned", "acclaimed"]
}



#%%  most similar terms

for x, y in models_all.items():
    print(y)
    print(x.most_similar("work"))

# --> work has a different meaning before 1850


#%% results

sim_onedim('religion')

sim_onedim('male')
sim_onedim('female')
sim_twodim('male', 'female')

sim_onedim('mat')
sim_onedim('postmat')
sim_twodim('mat', 'postmat')

sim_onedim('rich')
sim_onedim('poor')
sim_twodim('rich', 'poor')

sim_onedim('affluence')

sim_onedim('hard')

sim_onedim('politics')

sim_onedim('moral')

sim_onedim('vocation')

sim_onedim('success')

sim_onedim('housework')

sim_onedim('emotion')

sim_onedim('relations')

sim_onedim('status')