#%% import packages

import semchange
import simdim
import simdim2
import listsim
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


#%%  most similar terms by decade

for x, y in models_all.items():
    print(x)
    print(y.most_similar("work"))

# --> work has a different meaning before 1850

#%% visualize semantic change over time (PCA with keyword as passive projection)

semchange.semchange(models_all, "religious", rangelow=1810, rangehigh=2000, rangestep=60, export=False)


#%% set up dictionary and define "work"

keywords = dict()

keywords['work'] = [
    "work", "works", "worked", "working", "job", "jobs",
    "career", "careers",
    "profession", "professions", "professional",
    "occupation", "occupations",
    "employment", "employed",
    "labor", "labors", 'labour', 'labours'
]

# check if all terms exist in all embeddings
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
checksim = listsim.listsim(models_all, keywords, 'work')
checksim.to_clipboard() # insert & check in excel


#%% Smith: toil & trouble

keywords['Toil & Trouble'] = [
    "toil", "toils", "trouble", 'troubles', "exertion", "struggle", "struggles", "drudgery",
    "pains", "fatigue", "effort", "hard", "arduous", "strenuous"
]

# check similarity of words
checksim = listsim.listsim(models_all, keywords, 'Toil & Trouble')
checksim.to_clipboard()

# check if all terms exist in all embeddings
for i in keywords['Toil & Trouble']:
    for year, model in models_all.items():
        if model[i].all() == models_all[1840]['biology'].all():
            if year >= 1850:
                print(str(year) + ": " + i)



keywords['Pleasure'] = [
    "fun", "pleasure", "enjoy", "pleasant", "happy", "like", "love", "delight", "happiness",
    "satisfaction", "bliss"
]

# check if all terms exist in all embeddings
for i in keywords['Pleasure']:
    for year, model in models_all.items():
        if model[i].all() == models_all[1840]['biology'].all():
            if year >= 1850:
                print(str(year) + ": " + i)

simdim.simdim(models_all, keywords, 'work', 'Toil & Trouble', 'Pleasure', ci=90)



#%% Marx: alienation (extrinsic vs. intrinsic)

keywords['Extrinsic'] = [
                    "earn", "earning", "earnings",
                    "wage", "wages", "salary", "income", "remuneration", "pay",
                    "secure", "security", "insecure", "insecurity"
]

# check if all terms exist in all embeddings
for i in keywords['Extrinsic']:
    for year, model in models_all.items():
        if model[i].all() == models_all[1840]['biology'].all():
            if year >= 1850:
                print(str(year) + ": " + i)

# check similarity of words
checksim = listsim.listsim(models_all, keywords, 'Extrinsic')
checksim.to_clipboard()

keywords['Intrinsic'] = [
    "interesting", "boring", "fulfilling", "useful", "useless",
    "expression", "creative", "express", "satisfying", "stimulating", "expressive", "important"
]

# check similarity of words
checksim = listsim.listsim(models_all, keywords, 'Intrinsic')
checksim.to_clipboard()

# check if all terms exist in all embeddings
for i in keywords['Intrinsic']:
    for year, model in models_all.items():
        if model[i].all() == models_all[1840]['biology'].all():
            if year >= 1850:
                print(str(year) + ": " + i)


simdim.simdim(models_all, keywords, 'work', 'Extrinsic', 'Intrinsic', ci=90)


#%% Weber: wealth, morality & religion

keywords['rich'] = ["wealth", "wealthy", "rich", "affluence", "affluent"]
keywords['poor'] = ["poor", "poverty", "impoverished", "destitute", "needy"]

keywords['Affluence'] = keywords['rich'] + keywords['poor']

# check if all terms exist in all embeddings
for i in keywords['Affluence']:
    for year, model in models_all.items():
        if model[i].all() == models_all[1840]['biology'].all():
            if year >= 1850:
                print(str(year) + ": " + i)


keywords['Religion'] = [
    "redemption", "salvation", "god", "religion", "holy", "calling", "faith", "pious",
    "spiritual", "sacred", "divine", "belief", "worship"
]

# check if all terms exist in all embeddings
for i in keywords['Religion']:
    for year, model in models_all.items():
        if model[i].all() == models_all[1840]['biology'].all():
            if year >= 1850:
                print(str(year) + ": " + i)

# check similarity of words
checksim = listsim.listsim(models_all, keywords, 'Religion')
checksim.to_clipboard()

keywords['Morality'] = [
    'good', 'evil', 'moral', 'immoral', 'good', 'bad', 'honest', 'dishonest',
    'virtuous', 'sinful', 'virtue', 'vice',
    "decent", "noble", "honour", "integrity", "worth", "dignity"
]

# check if all terms exist in all embeddings
for i in keywords['Morality']:
    for year, model in models_all.items():
        if model[i].all() == models_all[1840]['biology'].all():
            if year >= 1850:
                print(str(year) + ": " + i)

# check similarity of words
checksim = listsim.listsim(models_all, keywords, 'Morality')
checksim.to_clipboard()

simdim.simdim(models_all, keywords, 'work', 'Religion', 'Affluence', ci=90)
simdim.simdim(models_all, keywords, 'work', 'Religion', 'Morality', ci=90)
simdim.simdim(models_all, keywords, 'work', 'Morality', 'Affluence', ci=90)






#%% test alternative way to measure distances (simdim2)

simdim.simdim(models_all, keywords, 'work', 'Toil & Trouble', 'Pleasure', ci=90)
simdim2.simdim2(models_all, keywords, 'work', 'Toil & Trouble', 'Pleasure', ci=90)
