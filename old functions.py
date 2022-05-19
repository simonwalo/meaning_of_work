
# define similarity function for one dimension (Cosine_distance = 1 - cosine_similarity)

def sim_onedim(dim, rangelow = 1850, rangehigh = 2000, rangestep = 10):

    d = []

    for model, year in models_all.items():
        if year in range(rangelow, rangehigh, rangestep):
            d.append(
                {
                    "year": year,
                    dim: model.n_similarity(keywords['work'], keywords[dim])
                }
            )

    data = pd.DataFrame(d)

    # the trendline
    x = data['year'].tolist()
    y = data[dim].tolist()

    fun = interp1d(x, y, kind='cubic')

    xnew = np.linspace(rangelow, (rangehigh-10), 100)

    plt.plot(xnew, fun(xnew), '-', x, y, 'o')

    # show plot
    plt.title(dim)
    plt.show()
    plt.close()



# define similarity function for two dimensions (Cosine_distance = 1 - cosine_similarity)

def sim_twodim(dim1, dim2, diff = False, rangelow = 1850, rangehigh = 2000, rangestep = 10):

    d = []

    for model, year in models_all.items():
        if year in range(rangelow, rangehigh, rangestep):
            d.append(
                {
                    "year": year,
                    dim1: model.n_similarity(keywords['work'], keywords[dim1]),
                    dim2: model.n_similarity(keywords['work'], keywords[dim2])
                }
            )

    data = pd.DataFrame(d)
    data['diff'] = data[dim1] - data[dim2]

    # the trendline

    if diff==True:
        x = data['year'].tolist()
        y = data['diff'].tolist()

        fun = interp1d(x, y, kind='cubic')
        xnew = np.linspace(rangelow, (rangehigh-10), 100)

        plt.plot(xnew, fun(xnew), "-b")

        plt.title(dim1 + "-" + dim2)
        plt.show()
        plt.close()

    else:
        x = data['year'].tolist()
        y1 = data[dim1].tolist()
        y2 = data[dim2].tolist()

        fun1 = interp1d(x, y1, kind='cubic')
        fun2 = interp1d(x, y2, kind='cubic')

        xnew = np.linspace(rangelow, (rangehigh-10), 100)

        plt.plot(xnew, fun1(xnew), "-b", label=dim1)
        plt.plot(x, y1, 'bo')
        plt.plot(xnew, fun2(xnew), "-r", label=dim2)
        plt.plot(x, y2, 'ro')

        plt.legend(loc="best")

        #show plot
        plt.title(dim1 + "&" + dim2)
        plt.show()
        plt.close()







# define similarity function for three dimensions (Cosine_distance = 1 - cosine_similarity)


def sim_threedim(dim1, dim2, dim3, trend = 3, rangelow = 1850, rangehigh = 2000, rangestep = 10):

    d = []

    for model, year in models_all.items():
        if year in range(rangelow, rangehigh, rangestep):
            d.append(
                {
                    "year": year,
                    dim1: model.n_similarity(keywords['work'], keywords[dim1]),
                    dim2: model.n_similarity(keywords['work'], keywords[dim2]),
                    dim3: model.n_similarity(keywords['work'], keywords[dim3])
                }
            )

    data = pd.DataFrame(d)

    # the trendline

    if trend == 3:

        x = data['year'].tolist()
        y1 = data[dim1].tolist()
        y2 = data[dim2].tolist()
        y3 = data[dim3].tolist()

        fun1 = interp1d(x, y1, kind='cubic')
        fun2 = interp1d(x, y2, kind='cubic')
        fun3 = interp1d(x, y3, kind='cubic')


        xnew = np.linspace(rangelow, (rangehigh - 10), 100)

        plt.plot(xnew, fun1(xnew), "-b", label=dim1)
        plt.plot(x, y1, 'bo')
        plt.plot(xnew, fun2(xnew), "-r", label=dim2)
        plt.plot(x, y2, 'ro')
        plt.plot(xnew, fun3(xnew), "-g", label=dim3)
        plt.plot(x, y3, 'go')


    elif trend == 2:

        z1 = np.polyfit(data['year'], data[dim1], 2)
        z2 = np.polyfit(data['year'], data[dim2], 2)
        z3 = np.polyfit(data['year'], data[dim3], 2)

        p1 = np.poly1d(z1)
        p2 = np.poly1d(z2)
        p3 = np.poly1d(z3)

        plt.plot(data['year'], p1(data['year']), "r--", label=dim1)
        plt.plot(data['year'], p2(data['year']), "b--", label=dim2)
        plt.plot(data['year'], p3(data['year']), "g--", label=dim3)

    elif trend == 1:

        z1 = np.polyfit(data['year'], data[dim1], 1)
        z2 = np.polyfit(data['year'], data[dim2], 1)
        z3 = np.polyfit(data['year'], data[dim3], 1)

        p1 = np.poly1d(z1)
        p2 = np.poly1d(z2)
        p3 = np.poly1d(z3)

        plt.plot(data['year'], p1(data['year']), "r--", label=dim1)
        plt.plot(data['year'], p2(data['year']), "b--", label=dim2)
        plt.plot(data['year'], p3(data['year']), "g--", label=dim3)


    #show plot
    plt.legend(loc="best")
    plt.title(dim1 + "&" + dim2 + "&" + dim3)
    plt.show()
    plt.close()















##### Semantic Change of terms



# define function to visualize semantic change (PCA)

def similarplot1(keyword, rangelow = 1800, rangehigh = 2000, rangestep = 10):

    # get list of all similar words from different periods

    sim_words = []

    for model, year in models_all.items():
        if year in range(rangelow, rangehigh, rangestep):
            tempsim = model.most_similar(keyword, topn=7)
            for term, vector in tempsim:
                sim_words.append(term)

    sim_words = list(set(sim_words))

    # get vectors of similar words in most recent embedding (1990)
    sim_vectors1990 = np.array([embeddings1990[w] for w in sim_words])

    # get vectors of keyword in all periods and add them to vectors of similar words

    allvectors = sim_vectors1990

    for model, year in models_all.items():
        if year in range(rangelow, rangehigh, rangestep):
            keyword_vectors = np.array([model[keyword]])
            allvectors = np.append(allvectors, keyword_vectors, axis=0)

    # reduce dimensions of vectors
    pca = PCA(n_components=2)
    two_dim = pca.fit_transform(allvectors)

    # get labels
    labels = sim_words
    for model, year in models_all.items():
        if year in range(rangelow, rangehigh, rangestep):
            labels.append(keyword + str(year))

    #plot results
    plt.scatter(two_dim[:, 0], two_dim[:, 1])

    for i in range(len(sim_words)):
        plt.text(x=two_dim[i, 0], y=two_dim[i, 1], s=labels[i])

    plt.show()
    plt.close()




# define function to visualize semantic change (T-SNE)

def similarplot2(keyword, rangelow = 1800, rangehigh = 2000, rangestep = 10):

    # get list of all similar words from different periods

    sim_words = []

    for model, year in models_all.items():
        if year in range(rangelow, rangehigh, rangestep):
            tempsim = model.most_similar(keyword, topn=7)
            for term, vector in tempsim:
                sim_words.append(term)

    sim_words = list(set(sim_words))

    # get vectors of similar words in most recent embedding (1990)
    sim_vectors1990 = np.array([embeddings1990[w] for w in sim_words])

    # get vectors of keyword in all periods and add them to vectors of similar words

    allvectors = sim_vectors1990

    for model, year in models_all.items():
        if year in range(rangelow, rangehigh, rangestep):
            keyword_vectors = np.array([model[keyword]])
            allvectors = np.append(allvectors, keyword_vectors, axis=0)

    # reduce dimensions of vectors
    tsne = TSNE(n_components=2, learning_rate="auto", init="pca", perplexity=5)
    two_dim = tsne.fit_transform(allvectors)

    # get labels
    labels = sim_words
    for model, year in models_all.items():
        if year in range(rangelow, rangehigh, rangestep):
            labels.append(keyword + str(year))

    #plot results
    plt.scatter(two_dim[:, 0], two_dim[:, 1])

    for i in range(len(sim_words)):
        plt.text(x=two_dim[i, 0], y=two_dim[i, 1], s=labels[i])

    plt.show()
    plt.close()
