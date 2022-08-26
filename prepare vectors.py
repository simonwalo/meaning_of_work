from gensim.models import KeyedVectors

coha1810 = KeyedVectors.load_word2vec_format('./data/COHA/coha-word_sgns_gensim_1810.txt', binary=False, no_header=True)
coha1820 = KeyedVectors.load_word2vec_format('./data/COHA/coha-word_sgns_gensim_1820.txt', binary=False, no_header=True)
coha1830 = KeyedVectors.load_word2vec_format('./data/COHA/coha-word_sgns_gensim_1830.txt', binary=False, no_header=True)
coha1840 = KeyedVectors.load_word2vec_format('./data/COHA/coha-word_sgns_gensim_1840.txt', binary=False, no_header=True)
coha1850 = KeyedVectors.load_word2vec_format('./data/COHA/coha-word_sgns_gensim_1850.txt', binary=False, no_header=True)
coha1860 = KeyedVectors.load_word2vec_format('./data/COHA/coha-word_sgns_gensim_1860.txt', binary=False, no_header=True)
coha1870 = KeyedVectors.load_word2vec_format('./data/COHA/coha-word_sgns_gensim_1870.txt', binary=False, no_header=True)
coha1880 = KeyedVectors.load_word2vec_format('./data/COHA/coha-word_sgns_gensim_1880.txt', binary=False, no_header=True)
coha1890 = KeyedVectors.load_word2vec_format('./data/COHA/coha-word_sgns_gensim_1890.txt', binary=False, no_header=True)
coha1900 = KeyedVectors.load_word2vec_format('./data/COHA/coha-word_sgns_gensim_1900.txt', binary=False, no_header=True)
coha1910 = KeyedVectors.load_word2vec_format('./data/COHA/coha-word_sgns_gensim_1910.txt', binary=False, no_header=True)
coha1920 = KeyedVectors.load_word2vec_format('./data/COHA/coha-word_sgns_gensim_1920.txt', binary=False, no_header=True)
coha1930 = KeyedVectors.load_word2vec_format('./data/COHA/coha-word_sgns_gensim_1930.txt', binary=False, no_header=True)
coha1940 = KeyedVectors.load_word2vec_format('./data/COHA/coha-word_sgns_gensim_1940.txt', binary=False, no_header=True)
coha1950 = KeyedVectors.load_word2vec_format('./data/COHA/coha-word_sgns_gensim_1950.txt', binary=False, no_header=True)
coha1960 = KeyedVectors.load_word2vec_format('./data/COHA/coha-word_sgns_gensim_1960.txt', binary=False, no_header=True)
coha1970 = KeyedVectors.load_word2vec_format('./data/COHA/coha-word_sgns_gensim_1970.txt', binary=False, no_header=True)
coha1980 = KeyedVectors.load_word2vec_format('./data/COHA/coha-word_sgns_gensim_1980.txt', binary=False, no_header=True)
coha1990 = KeyedVectors.load_word2vec_format('./data/COHA/coha-word_sgns_gensim_1990.txt', binary=False, no_header=True)
coha2000 = KeyedVectors.load_word2vec_format('./data/COHA/coha-word_sgns_gensim_2000.txt', binary=False, no_header=True)

with open('data/COHA/coha1810.pickle', 'wb') as f:
    pickle.dump(coha1810, f)
with open('data/COHA/coha1820.pickle', 'wb') as f:
    pickle.dump(coha1820, f)
with open('data/COHA/coha1830.pickle', 'wb') as f:
    pickle.dump(coha1830, f)
with open('data/COHA/coha1840.pickle', 'wb') as f:
    pickle.dump(coha1840, f)
with open('data/COHA/coha1850.pickle', 'wb') as f:
    pickle.dump(coha1850, f)
with open('data/COHA/coha1860.pickle', 'wb') as f:
    pickle.dump(coha1860, f)
with open('data/COHA/coha1870.pickle', 'wb') as f:
    pickle.dump(coha1870, f)
with open('data/COHA/coha1880.pickle', 'wb') as f:
    pickle.dump(coha1880, f)
with open('data/COHA/coha1890.pickle', 'wb') as f:
    pickle.dump(coha1890, f)
with open('data/COHA/coha1900.pickle', 'wb') as f:
    pickle.dump(coha1900, f)
with open('data/COHA/coha1910.pickle', 'wb') as f:
    pickle.dump(coha1910, f)
with open('data/COHA/coha1920.pickle', 'wb') as f:
    pickle.dump(coha1920, f)
with open('data/COHA/coha1930.pickle', 'wb') as f:
    pickle.dump(coha1930, f)
with open('data/COHA/coha1940.pickle', 'wb') as f:
    pickle.dump(coha1940, f)
with open('data/COHA/coha1950.pickle', 'wb') as f:
    pickle.dump(coha1950, f)
with open('data/COHA/coha1960.pickle', 'wb') as f:
    pickle.dump(coha1960, f)
with open('data/COHA/coha1970.pickle', 'wb') as f:
    pickle.dump(coha1970, f)
with open('data/COHA/coha1980.pickle', 'wb') as f:
    pickle.dump(coha1980, f)
with open('data/COHA/coha1990.pickle', 'wb') as f:
    pickle.dump(coha1990, f)
with open('data/COHA/coha2000.pickle', 'wb') as f:
    pickle.dump(coha2000, f)