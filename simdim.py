
# debug: title of graph should inlude key name

# define similarity function for n dimensions

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import random



def simdim(models, keywords, key, *dims, rangelow=1850, rangehigh=2000, rangestep=10):

    medians = pd.DataFrame()
    lower_cis = pd.DataFrame()
    upper_cis = pd.DataFrame()

    for dim in dims:
        data = pd.DataFrame(index=range(len(range(rangelow, rangehigh, rangestep))), columns=range(1000))

        for i in range(1000):

            sample1 = keywords['work']  # random.choices(keywords['work'], k=len(keywords['work']))
            sample2 = random.choices(keywords[dim], k=len(keywords[dim]))

            d = []
            for year, model in models.items():
                if year in range(rangelow, rangehigh, rangestep):
                    d.append(model.n_similarity(sample1, sample2))
            d = np.asarray(d)
            data[i] = d

        # get medians
        temp = []
        for i in range(len(range(rangelow, rangehigh, rangestep))):
            sample = data.iloc[i]
            temp.append(sample.median())
        medians[dim] = temp

        # get 95% intervals
        alpha = 100 - 95

        # get lowers CIs
        temp = []
        for i in range(len(range(rangelow, rangehigh, rangestep))):
            sample = data.iloc[i]
            temp.append(np.percentile(sample, alpha / 2))
        lower_cis[dim] = temp

        # get upper CIs
        temp = []
        for i in range(len(range(rangelow, rangehigh, rangestep))):
            sample = data.iloc[i]
            temp.append(np.percentile(sample, 100 - alpha / 2))
        upper_cis[dim] = temp


    # the trendline

    x = range(rangelow, rangehigh, rangestep)
    xnew = np.linspace(rangelow, (rangehigh - 10), 100)

    n = len(dims)
    colors = iter(cm.rainbow(np.linspace(0, 1, n)))

    for dim in dims:
        y = medians[dim].tolist()
        fun = interp1d(x, y, kind='cubic')
        color = next(colors)
        plt.plot(xnew, fun(xnew), "-", color=color, label=dim)
        plt.plot(x, y, 'o', color=color)
        plt.fill_between(x, lower_cis[dim].tolist(), upper_cis[dim].tolist(), alpha=0.2)

    #show plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(f'Association of dimension(s) with {key=}')
    plt.tight_layout()
    plt.show()
    plt.close()










