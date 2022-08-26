import pandas as pd
import numpy as np
import random


data = pd.DataFrame()

for i in range(1000):

    sample1= random.choices(keywords['work'], k=len(keywords['work']))
    sample2= random.choices(keywords['female'], k=len(keywords['female']))

    d = []
    for year, model in models_all.items():
        d.append(model.n_similarity(sample1, sample2))
    data[i] = d

# get median
sample1800 = data.iloc[0]

median = np.percentile(sample1800, 50)

# get 95% interval
alpha = 100-95
lower_ci = np.percentile(sample1800, alpha/2)
upper_ci = np.percentile(sample1800, 100-alpha/2)

print(f"Model accuracy is reported on the test set. 1000 bootstrapped samples " 
      f"were used to calculate 95% confidence intervals.\n"
      f"Median accuracy is {median:.2f} with a 95% a confidence "
      f"interval of [{lower_ci:.2f},{upper_ci:.2f}].")