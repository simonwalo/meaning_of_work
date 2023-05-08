import gensim
import pandas as pd

# check similarity of words within list

def listsim(models, keywords, listkey, year=1990):
    data = pd.DataFrame(index=keywords[listkey], columns=keywords[listkey])
    col = 0
    for word1 in keywords[listkey]:
        d = []
        for word2 in keywords[listkey]:
            d.append(models[year].n_similarity([word1], [word2]))
        data.iloc[:, col] = d
        col = col + 1
    return data

