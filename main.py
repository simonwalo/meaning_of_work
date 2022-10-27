#%% import packages

import sim_occs
import semchange
import simdim
import sim_oneterm
from gensim.models import KeyedVectors


#%% load data

models_all = {
    1800: KeyedVectors.load('./data/vectors1800.kv'),
    1810: KeyedVectors.load('./data/vectors1810.kv'),
    1820: KeyedVectors.load('./data/vectors1820.kv'),
    1830: KeyedVectors.load('./data/vectors1830.kv'),
    1840: KeyedVectors.load('./data/vectors1840.kv'),
    1850: KeyedVectors.load('./data/vectors1850.kv'),
    1860: KeyedVectors.load('./data/vectors1860.kv'),
    1870: KeyedVectors.load('./data/vectors1870.kv'),
    1880: KeyedVectors.load('./data/vectors1880.kv'),
    1890: KeyedVectors.load('./data/vectors1890.kv'),
    1900: KeyedVectors.load('./data/vectors1900.kv'),
    1910: KeyedVectors.load('./data/vectors1910.kv'),
    1920: KeyedVectors.load('./data/vectors1920.kv'),
    1930: KeyedVectors.load('./data/vectors1930.kv'),
    1940: KeyedVectors.load('./data/vectors1940.kv'),
    1950: KeyedVectors.load('./data/vectors1950.kv'),
    1960: KeyedVectors.load('./data/vectors1960.kv'),
    1970: KeyedVectors.load('./data/vectors1970.kv'),
    1980: KeyedVectors.load('./data/vectors1980.kv'),
    1990: KeyedVectors.load('./data/vectors1990.kv')
}


#%%  most similar terms

for x, y in models_all.items():
    print(x)
    print(y.most_similar("work"))

# --> work has a different meaning before 1850

#%% visualize word embeddings over time (PCA mit keyword als passiv)

semchange.semchange(models_all, "religious", rangelow=1810, rangehigh=2000, rangestep=60, export=False)


#%% association of work with different dimension

# set up dictionary and define "work"

keywords = dict()

keywords['work'] = [
    "work", "works", "worked", "working", "job", "jobs",
    "career", "careers",
    "profession", "professions", "professional",
    "occupation", "occupations",
    "employment", "employed",
    "labor", "labors", 'labour', 'labours'
]

for i in keywords['work']:
    for year, model in models_all.items():
        if model[i].all() == models_all[1840]['biology'].all():
            if year >= 1850:
                print(str(year) + ": " + i)


keywords['work'] = [
    "work", "works", "worked", "working", "job", "jobs",
    "career",
    "profession", "professions", "professional",
    "occupation", "occupations",
    "employment", "employed"
]

# check similarity of words
for i in keywords['work']:
    print(models_all[1850].n_similarity(['labors'], [i]))


#%% smith: toil (einzelne Begriffe anzeigen?)

keywords['toil'] = [
    "hard", "struggle", "toil", "trouble", "suffer", "endure", "arduous", "strenuous", "grind"
]

# check similarity of words
for i in keywords['toil']:
    print(models_all[1910].n_similarity(['exertion'], [i]))

# so far best CIs:     "toil", "toils", "toiling", "trouble", "exertion", "struggle", "drudgery", "pains", "labor", "labour", "travail", "fatigue"

keywords['toil2'] = [
    "toil", "toils", "trouble", 'troubles', "exertion", "struggle", "struggles", "drudgery",
    "pains", "labor", "labour", "travail", "fatigue"
]

keywords['toil3'] = [
    "hard", "arduous", "strenuous", "exhausting", "burdensome", "painful", "tough", "onerous"
]

keywords['toil4'] = [
    "toil", "trouble", "exertion", "struggle", "drudgery"
]

for i in keywords['toil2']:
    for year, model in models_all.items():
        if model[i].all() == models_all[1840]['biology'].all():
            if year >= 1850:
                print(str(year) + ": " + i)

simdim.simdim(models_all, keywords, 'work', 'toil', rangelow=1850, rangehigh=2000, rangestep=10)
simdim.simdim(models_all, keywords, 'work', 'toil2')

keywords['pleasure'] = [
    "fun", "pleasure", "enjoy", "pleasant", "happy", "like", "love", "delight", "happiness",
    "satisfaction", "bliss"
]

for i in keywords['pleasure']:
    for year, model in models_all.items():
        if model[i].all() == models_all[1840]['biology'].all():
            if year >= 1850:
                print(str(year) + ": " + i)

simdim.simdim(models_all, keywords, 'work', 'toil2', 'pleasure', ci=90)
simdim.simdim(models_all, keywords, 'work', 'toil3', 'pleasure', ci=95)
simdim.simdim(models_all, keywords, 'work', 'toil4', 'pleasure', ci=95)


keywords['leisure'] = [
    "leisure", "ease", "rest", "recreation", "relaxation", "freedom"
]

simdim.simdim(models_all, keywords, 'work', 'leisure')

simdim.simdim(models_all, keywords, 'work', 'toil', 'leisure')


keywords['hard'] = ['hard']
keywords['struggle'] = ['struggle']
keywords['toil'] = ['toil']
keywords['trouble'] = ['trouble']
keywords['suffer'] = ['suffer']
keywords['endure'] = ['endure']
keywords['arduous'] = ['arduous']
keywords['strenuous'] = ['strenuous']

simdim.simdim(models_all, keywords, 'work', 'hard', 'struggle', 'toil', 'trouble',
              'suffer', 'endure', 'arduous', 'strenuous')




keywords['stress'] = ["stress", "exhausting", "tired"]
simdim.simdim(models_all, keywords, 'work', 'stress')
simdim.simdim(models_all, keywords, 'work', 'toil', 'stress')

keywords['emotion'] = [
    "pleasant", "interesting", "boring", "fulfilling", "meaningful", "meaningless",
    "hard", "struggle", "toil", "trouble", "suffer", "endure", "arduous", "strenuous"
]
simdim.simdim(models_all, keywords, 'work', 'emotion')

keywords['commodity'] = [
    "market", "exchange", "trade", "hire", "rent"
]
simdim.simdim(models_all, keywords, 'work', 'commodity')  # nicht sehr spannend


sim_oneterm.sim_oneterm(models_all, keywords, 'work', 'duty')  # auch teil von "patriot"

sim_oneterm.sim_oneterm(models_all, keywords, 'work', 'pleasant')


#%% marx: alienation (extrinsic vs. intrinsic)

keywords['mat'] = [
                    "earn", "earns", "earning", "earnings",
                    "wage", "wages", "salary", "income", "remuneration", "pay",
                    "secure", "security", "insecure", "insecurity"
]

for i in keywords['mat']:
    for year, model in models_all.items():
        if model[i].all() == models_all[1840]['biology'].all():
            if year >= 1850:
                print(str(year) + ": " + i)

keywords['mat'] = [
                    "earn", "earning", "earnings",
                    "wage", "wages", "salary", "income", "remuneration", "pay",
                    "secure", "security", "insecure", "insecurity"
]

keywords['secure'] = [
                      "secure", "security", "insecure", "insecurity"
]


keywords['postmat'] = ["interesting", "boring", "fulfilling", "meaningful", "meaningless", "useful", "useless",
                       "expression", "creative"]

for i in keywords['postmat']:
    for year, model in models_all.items():
        if model[i].all() == models_all[1840]['biology'].all():
            if year >= 1850:
                print(str(year) + ": " + i)

keywords['postmat'] = ["interesting", "boring", "fulfilling", "useful", "useless",
                       "expression", "creative"]

simdim.simdim(models_all, keywords, 'work', 'mat')
simdim.simdim(models_all, keywords, 'work', 'postmat')
simdim.simdim(models_all, keywords, 'work', 'secure')

simdim.simdim(models_all, keywords, 'work', 'mat', 'postmat', ci=90)

keywords['useful'] = ["useful", "society"]
simdim.simdim(models_all, keywords, 'work', 'useful')


keywords['status'] = [
    "prestigious", "honorable", "esteemed", "influential", "reputable", "distinguished",
    "eminent", "illustrious", "renowned", "acclaimed"
]
simdim.simdim(models_all, keywords, 'work', 'mat', 'postmat', 'status')

keywords['social'] = ["colleague", "colleague", "friend", "friends", "people"]
simdim.simdim(models_all, keywords, 'work', 'social')

simdim.simdim(models_all, keywords, 'work', 'mat', 'postmat', 'status', 'social')


#%% weber: wealth, morality & religion

keywords['rich'] = ["wealth", "wealthy", "rich", "affluence", "affluent"]
keywords['poor'] = ["poor", "poverty", "impoverished", "destitute", "needy"]

keywords['affluence'] = keywords['rich'] + keywords['poor']

for i in keywords['affluence']:
    for year, model in models_all.items():
        if model[i].all() == models_all[1840]['biology'].all():
            if year >= 1850:
                print(str(year) + ": " + i)

simdim.simdim(models_all, keywords, 'work', 'affluence')  # --> Piketty!

keywords['religion'] = [
    "redemption", "salvation", "god", "religion", "holy", "calling", "faith", "pious",
    "spiritual", "sacred", "divine", "belief", "worship"
]

for i in keywords['religion']:
    for year, model in models_all.items():
        if model[i].all() == models_all[1840]['biology'].all():
            if year >= 1850:
                print(str(year) + ": " + i)

# check similarity of words
for i in keywords['religion']:
    print(models_all[1910].n_similarity(['calling'], [i]))

simdim.simdim(models_all, keywords, 'work', 'religion')

keywords['moral'] = [
    'good', 'evil', 'moral', 'immoral', 'good', 'bad', 'honest', 'dishonest',
    'virtuous', 'sinful', 'virtue', 'vice',
    "decent", "noble", "honour", "integrity", "worth", "dignity"
]

keywords['moral2'] = [
    'good', 'moral', 'good', 'honest', 'virtuous', 'virtue'
]


for i in keywords['moral']:
    for year, model in models_all.items():
        if model[i].all() == models_all[1840]['biology'].all():
            if year >= 1850:
                print(str(year) + ": " + i)

simdim.simdim(models_all, keywords, 'work', 'moral')  # --> Piketty!

simdim.simdim(models_all, keywords, 'work', 'religion', 'moral', 'affluence', ci=90)
simdim.simdim(models_all, keywords, 'work', 'religion', 'affluence', ci=90)
simdim.simdim(models_all, keywords, 'work', 'religion', 'moral', ci=90)
simdim.simdim(models_all, keywords, 'work', 'religion', 'moral2', ci=90)
simdim.simdim(models_all, keywords, 'work', 'moral', 'affluence', ci=90)




keywords['success'] = ["success", "succeed", "failure", "fail"]
simdim.simdim(models_all, keywords, 'work', 'success')





keywords['vocation'] = ["vocation", "calling", "meaning", "purpose"]
simdim.simdim(models_all, keywords, 'work', 'vocation')




simdim.simdim(models_all, keywords, 'work', 'moral', 'affluence', 'religion')  # --> Piketty!


# work hard?

# Weber: was läuft bei WK?

keywords['patriot'] = ["duty", "country", "patriot", "fatherland", "home"]
simdim.simdim(models_all, keywords, 'work', 'patriot')









#%% VALIDATION

# faktisch

# connotation von Arbeit mit mann/frau

keywords['male'] = ["male", "man", "boy", "brother", "he", "him", "his", "son"]
keywords['male'] = [
    'he', 'son', 'his', 'him', 'father', 'man', 'boy', 'himself',
    'male', 'brother', 'sons', 'fathers', 'men', 'boys', 'males',
    'brothers', 'uncle', 'uncles', 'nephew', 'nephews'
]

keywords['female'] = ["female", "woman", "girl", "sister", "she", "her", "hers", "daughter"]
keywords['female'] = [
    'she', 'daughter', 'hers', 'her', 'mother', 'woman', 'girl', 'herself', 'female',
    'sister', 'daughters', 'mothers', 'women', 'girls', 'sisters', 'aunt',
    'aunts', 'niece', 'nieces'
]

simdim.simdim(models_all, keywords, 'work', 'male')
simdim.simdim(models_all, keywords, 'work', 'female')
simdim.simdim(models_all, keywords, 'work', 'male', 'female',  rangelow=1850, rangehigh=2000, rangestep=10)
simdim.simdim(models_all, keywords, 'work', 'male', 'female')


# typische arbeitsgeräte für verschiedene epochen

keywords['plow'] = ['plow']
keywords['telephone'] = ['telephone']
keywords['computer'] = ['computer']

simdim.simdim(models_all, keywords, 'work', 'plow')
simdim.simdim(models_all, keywords, 'work', 'telephone')
simdim.simdim(models_all, keywords, 'work', 'computer')

simdim.simdim(models_all, keywords, 'work', 'plow', 'telephone', 'computer', trend=3)


# historisches wachstum von sektoren

keywords['sector1'] = ["agriculture", "farming", "logging", "fishing", "forestry", "mining"]

keywords['sector2'] = ["manufacturing", "textile", "car", "handicraft"]

keywords['sector3'] = ["service", "social", "information", "advice", "access"]

simdim.simdim(models_all, keywords, 'work', 'sector1', 'sector2', 'sector3', trend=3)

# typisch weibliche/männliche Berufe

keywords['male'] = ["male", "man", "boy", "brother", "he", "him", "his", "son"]
keywords['female'] = ["female", "woman", "girl", "sister", "she", "her", "hers", "daughter"]


sim_occs.sim_occs(models_all, keywords, 'mechanic', 'carpenter', 'engineer', 'nurse', "dancer", "housekeeper")


# SEMANTIC DRIFT

# housework --> work

keywords['housework'] = ["housework", "household"]
simdim.simdim(models_all, keywords, 'work', 'housework')


# beziehungsarbeit

keywords['relations'] = ["relationship"]
simdim.simdim(models_all, keywords, 'work', 'relations')


# DISKURS: Arbeiterbewegung

keywords['politics'] = ["party", "politics", "movement", "election"]
simdim.simdim(models_all, keywords, 'work', 'politics')




#%% Test distances between different POS

# random lists of adjectives/nouns/verbs:
# https://www.randomlists.com/random-adjectives?dup=false&qty=100

# import word lists (1000 each)

import pandas as pd
df = pd.read_csv(r'./data/random word lists.CSV', sep=';')

# clean word lists: delete if not in embeddings

adjectives = []
for i in df.adjectives:
    if i in models_all[1810]:
        adjectives.append(i)

nouns = []
for i in df.nouns:
    if i in models_all[1810]:
        nouns.append(i)

verbs = []
for i in df.verbs:
    if i in models_all[1810]:
        verbs.append(i)

for year, model in models_all.items():
    if year >= 1850:
        print(year, model.n_similarity(keywords['work'], adjectives))

for year, model in models_all.items():
    if year >= 1850:
        print(year, model.n_similarity(keywords['work'], nouns))

for year, model in models_all.items():
    if year >= 1850:
        print(year, model.n_similarity(keywords['work'], verbs))

for year, model in models_all.items():
    if year >= 1850:
        print(year, model.n_similarity(nouns, verbs))