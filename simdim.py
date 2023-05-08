
# debug: title of graph should inlude key name

# define similarity function for n dimensions

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import random
import matplotlib.lines as mlines




def simdim(models, keywords, key, *dims, rangelow=1850, rangehigh=2000, rangestep=10, ci=95, bootstrap=1000):

    medians = pd.DataFrame()
    lower_cis = pd.DataFrame()
    upper_cis = pd.DataFrame()

    for dim in dims:
        data = pd.DataFrame(index=range(len(range(rangelow, rangehigh, rangestep))), columns=range(bootstrap))

        for i in range(bootstrap):

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
        alpha = 100 - ci

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
    markslist = ['o', 's']
    marks = iter(markslist)

    for dim in dims:
        y = medians[dim].tolist()
        fun = interp1d(x, y, kind='cubic')

        low = lower_cis[dim].tolist()
        fun_low = interp1d(x, low, kind='cubic')

        high = upper_cis[dim].tolist()
        fun_high = interp1d(x, high, kind='cubic')

        plt.plot(xnew, fun(xnew), "-", x, y, next(marks), color='black')
        plt.fill_between(xnew, fun_low(xnew), fun_high(xnew), alpha=0.2, color='grey')


    # add legend and labels
    legend1 = mlines.Line2D([], [], color='black', marker='o', label=dims[0])
    legend2 = mlines.Line2D([], [], color='black', marker='s', label=dims[1])
    plt.legend(handles=[legend1, legend2], loc='center left', bbox_to_anchor=(1, 0.5))

    plt.xlabel("Year")
    plt.ylabel("Cosine Similarity")
    plt.xticks(range(1850, 2000, 20))
    plt.tight_layout()

    # show plot
    plt.show()
    plt.close()










