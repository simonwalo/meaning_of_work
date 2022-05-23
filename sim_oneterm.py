# define similarity function for one term (Cosine_distance = 1 - cosine_similarity)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d


def sim_oneterm(models, keywords, key, term, rangelow = 1850, rangehigh = 2000, rangestep = 10):

    d = []

    for year, model in models.items():
        if year in range(rangelow, rangehigh, rangestep):
            d.append(
                {
                    "year": year,
                    term: model.n_similarity(keywords[key], [term])
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