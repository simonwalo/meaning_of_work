#%% import packages

import pickle


#%% load data

models_all = {
    1800: pickle.load(open("./data/embeddings1800.pickle", 'rb')),
    1810: pickle.load(open("./data/embeddings1810.pickle", 'rb')),
    1820: pickle.load(open("./data/embeddings1820.pickle", 'rb')),
    1830: pickle.load(open("./data/embeddings1830.pickle", 'rb')),
    1840: pickle.load(open("./data/embeddings1840.pickle", 'rb')),
    1850: pickle.load(open("./data/embeddings1850.pickle", 'rb')),
    1860: pickle.load(open("./data/embeddings1860.pickle", 'rb')),
    1870: pickle.load(open("./data/embeddings1870.pickle", 'rb')),
    1880: pickle.load(open("./data/embeddings1880.pickle", 'rb')),
    1890: pickle.load(open("./data/embeddings1890.pickle", 'rb')),
    1900: pickle.load(open("./data/embeddings1900.pickle", 'rb')),
    1910: pickle.load(open("./data/embeddings1910.pickle", 'rb')),
    1920: pickle.load(open("./data/embeddings1920.pickle", 'rb')),
    1930: pickle.load(open("./data/embeddings1930.pickle", 'rb')),
    1940: pickle.load(open("./data/embeddings1940.pickle", 'rb')),
    1950: pickle.load(open("./data/embeddings1950.pickle", 'rb')),
    1960: pickle.load(open("./data/embeddings1960.pickle", 'rb')),
    1970: pickle.load(open("./data/embeddings1970.pickle", 'rb')),
    1980: pickle.load(open("./data/embeddings1980.pickle", 'rb')),
    1990: pickle.load(open("./data/embeddings1990.pickle", 'rb')),
}


#%%  most similar terms

for x, y in models_all.items():
    print(x)
    print(y.most_similar("work"))

# --> work has a different meaning before 1850

#%% visualize word embeddings over time (PCA mit keyword als passiv)

import semchange

semchange.semchange(models_all, "work", rangelow=1810, rangehigh=2000, rangestep=60, export=False)




#%% association of work with different dimension

import simdim

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

simdim.simdim(models_all, keywords, 'work', 'toil', trend=3, diff=False, rangelow=1850, rangehigh=2000, rangestep=10)
simdim.simdim(models_all, keywords, 'work', 'toil')


keywords['leisure'] = ["leisure", "ease", "rest", "recreation", "relaxation", "freedom"]
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

simdim.simdim(models_all, keywords, 'work', 'hard', 'struggle', 'toil', 'trouble', 'suffer', 'endure', 'arduous', 'strenuous')





keywords['fun'] = ["fun", "enjoy", "pleasant"]
simdim.simdim(models_all, keywords, 'work', 'fun')



keywords['emotion'] = [
    "pleasant", "interesting", "boring", "fulfilling", "meaningful", "meaningless",
    "hard", "struggle", "toil", "trouble", "suffer", "endure", "arduous", "strenuous"
]
simdim('emotion', 1850)
simdim.simdim(models_all, keywords, 'work', 'emotion')

keywords['commodity'] = [
    "market", "exchange", "trade", "hire", "rent"
]
simdim.simdim(models_all, keywords, 'work', 'commodity') # nicht sehr spannend


import sim_oneterm

sim_oneterm.sim_oneterm(models_all, keywords, 'work', 'duty') # auch teil von "patriot"

sim_oneterm.sim_oneterm(models_all, keywords, 'work', 'pleasant')










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


# work hard?



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

import sim_occs
sim_occs.sim_occs(models_all, keywords, 'mechanic', 'carpenter', 'engineer', 'nurse', "dancer", "housekeeper")




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











































