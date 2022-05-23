#%% import packages
import sys
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle
from adjustText import adjust_text
from scipy.interpolate import interp1d

#%% load all data in C text format

embeddings1800 = KeyedVectors.load_word2vec_format("./data/english_all_sgns_gensim_1800.txt", binary=False, no_header=True)
embeddings1810 = KeyedVectors.load_word2vec_format("./data/english_all_sgns_gensim_1810.txt", binary=False, no_header=True)
embeddings1820 = KeyedVectors.load_word2vec_format("./data/english_all_sgns_gensim_1820.txt", binary=False, no_header=True)
embeddings1830 = KeyedVectors.load_word2vec_format("./data/english_all_sgns_gensim_1830.txt", binary=False, no_header=True)
embeddings1840 = KeyedVectors.load_word2vec_format("./data/english_all_sgns_gensim_1840.txt", binary=False, no_header=True)
embeddings1850 = KeyedVectors.load_word2vec_format("./data/english_all_sgns_gensim_1850.txt", binary=False, no_header=True)
embeddings1860 = KeyedVectors.load_word2vec_format("./data/english_all_sgns_gensim_1860.txt", binary=False, no_header=True)
embeddings1870 = KeyedVectors.load_word2vec_format("./data/english_all_sgns_gensim_1870.txt", binary=False, no_header=True)
embeddings1880 = KeyedVectors.load_word2vec_format("./data/english_all_sgns_gensim_1880.txt", binary=False, no_header=True)
embeddings1890 = KeyedVectors.load_word2vec_format("./data/english_all_sgns_gensim_1890.txt", binary=False, no_header=True)
embeddings1900 = KeyedVectors.load_word2vec_format("./data/english_all_sgns_gensim_1900.txt", binary=False, no_header=True)
embeddings1910 = KeyedVectors.load_word2vec_format("./data/english_all_sgns_gensim_1910.txt", binary=False, no_header=True)
embeddings1920 = KeyedVectors.load_word2vec_format("./data/english_all_sgns_gensim_1920.txt", binary=False, no_header=True)
embeddings1930 = KeyedVectors.load_word2vec_format("./data/english_all_sgns_gensim_1930.txt", binary=False, no_header=True)
embeddings1940 = KeyedVectors.load_word2vec_format("./data/english_all_sgns_gensim_1940.txt", binary=False, no_header=True)
embeddings1950 = KeyedVectors.load_word2vec_format("./data/english_all_sgns_gensim_1950.txt", binary=False, no_header=True)
embeddings1960 = KeyedVectors.load_word2vec_format("./data/english_all_sgns_gensim_1960.txt", binary=False, no_header=True)
embeddings1970 = KeyedVectors.load_word2vec_format("./data/english_all_sgns_gensim_1970.txt", binary=False, no_header=True)
embeddings1980 = KeyedVectors.load_word2vec_format("./data/english_all_sgns_gensim_1980.txt", binary=False, no_header=True)
embeddings1990 = KeyedVectors.load_word2vec_format("./data/english_all_sgns_gensim_1990.txt", binary=False, no_header=True)



#%% save embeddings in python format (loads faster)

pickle.dump(embeddings1800, open("./data/embeddings1800.pickle", 'wb'))
pickle.dump(embeddings1810, open("./data/embeddings1810.pickle", 'wb'))
pickle.dump(embeddings1820, open("./data/embeddings1820.pickle", 'wb'))
pickle.dump(embeddings1830, open("./data/embeddings1830.pickle", 'wb'))
pickle.dump(embeddings1840, open("./data/embeddings1840.pickle", 'wb'))
pickle.dump(embeddings1850, open("./data/embeddings1850.pickle", 'wb'))
pickle.dump(embeddings1860, open("./data/embeddings1860.pickle", 'wb'))
pickle.dump(embeddings1870, open("./data/embeddings1870.pickle", 'wb'))
pickle.dump(embeddings1880, open("./data/embeddings1880.pickle", 'wb'))
pickle.dump(embeddings1890, open("./data/embeddings1890.pickle", 'wb'))
pickle.dump(embeddings1900, open("./data/embeddings1900.pickle", 'wb'))
pickle.dump(embeddings1910, open("./data/embeddings1910.pickle", 'wb'))
pickle.dump(embeddings1920, open("./data/embeddings1920.pickle", 'wb'))
pickle.dump(embeddings1930, open("./data/embeddings1930.pickle", 'wb'))
pickle.dump(embeddings1940, open("./data/embeddings1940.pickle", 'wb'))
pickle.dump(embeddings1950, open("./data/embeddings1950.pickle", 'wb'))
pickle.dump(embeddings1960, open("./data/embeddings1960.pickle", 'wb'))
pickle.dump(embeddings1970, open("./data/embeddings1970.pickle", 'wb'))
pickle.dump(embeddings1980, open("./data/embeddings1980.pickle", 'wb'))
pickle.dump(embeddings1990, open("./data/embeddings1990.pickle", 'wb'))