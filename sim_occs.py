
# define similarity function for occupations with gender (Cosine_distance = 1 - cosine_similarity)

#%% import packages

import matplotlib.pyplot as plt
import pandas as pd


def sim_occs(models, keywords, *occs):

    diffdata = pd.DataFrame(models.keys())
    diffdata.rename(columns={0: 'year'}, inplace=True)

    for occ in occs:
        d = []
        for year, model in models.items():
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