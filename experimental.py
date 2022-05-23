# stacked area chart for sectors

d = []

for model, year in models_all.items():
    if year in range(1850, 2000, 10):
        d.append(
            {
                "year": year,
                "sector1": model.n_similarity(keywords['work'], keywords["sector1"]),
                "sector2": model.n_similarity(keywords['work'], keywords["sector2"]),
                "sector3": model.n_similarity(keywords['work'], keywords["sector3"])
            }
        )

data = pd.DataFrame(d)

data['total'] = data['sector1'] + data['sector2'] + data['sector3']

data['sector1'] = data['sector1'].divide(data['total'])
data['sector2'] = data['sector2'].divide(data['total'])
data['sector3'] = data['sector3'].divide(data['total'])

plt.stackplot(data["year"], data["sector1"], data["sector2"], data["sector3"], labels=['sector1','sector2','sector3'])
plt.legend(loc='upper left')
plt.margins(0,0)
plt.title('100 % stacked area chart')
plt.show()
