import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from scipy.interpolate import interp1d

keywords = dict()

keywords['work'] = [
    "work", "works", "worked", "working", "job", "jobs",
    "career",
    "profession", "professions", "professional",
    "occupation", "occupations",
    "employment", "employed",
    "labor", "labors"
]

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



dims = ['male', 'female']


medians = pd.DataFrame()
lower_cis = pd.DataFrame()
upper_cis = pd.DataFrame()

for dim in dims:

    data = pd.DataFrame(index=range(20),columns=range(1000))

    for i in range(1000):

        sample1= keywords['work'] #random.choices(keywords['work'], k=len(keywords['work']))
        sample2= random.choices(keywords[dim], k=len(keywords[dim]))

        d = []
        for year, model in models_all.items():
            d.append(model.n_similarity(sample1, sample2))
        d = np.asarray(d)
        data[i] = d

    # get medians
    temp = []
    for i in range(20):
        sample = data.iloc[i]
        temp.append(sample.median())
    medians[dim] = temp

    # get 95% intervals
    alpha = 100 - 95

    # get lowers CIs
    temp = []
    for i in range(20):
        sample = data.iloc[i]
        temp.append(np.percentile(sample, alpha/2))
    lower_cis[dim] = temp

    # get upper CIs
    temp = []
    for i in range(20):
        sample = data.iloc[i]
        temp.append(np.percentile(sample, 100-alpha/2))
    upper_cis[dim] = temp


# create plot

x = range(1800, 2000, 10)
xnew = np.linspace(1800, (2000 - 10), 100)

n = len(dims)
colors = iter(cm.rainbow(np.linspace(0, 1, n)))

for dim in dims:
    y = medians[dim].tolist()
    fun = interp1d(x, y, kind='cubic')
    color=next(colors)
    plt.plot(xnew, fun(xnew), "-", color=color, label=dim)
    plt.plot(x, y, 'o', color=color)
    plt.fill_between(x, lower_cis[dim].tolist(), upper_cis[dim]. tolist(), alpha=0.2)

plt.show()