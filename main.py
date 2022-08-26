#%% import packages

import pickle
import sim_occs
import semchange
import simdim
import sim_oneterm


#%% load data

google = {}
for i in range(1800, 2000, 10):
    google[i] = pickle.load(open("./data/Google/embeddings" + str(i) + ".pickle", 'rb'))

coha = {}
for i in range(1810, 2010, 10):
    coha[i] = pickle.load(open("./data/COHA/coha" + str(i) + ".pickle", 'rb'))


#%%  most similar terms

for x, y in google.items():
    print(x)
    print(y.most_similar("gay"))

for x, y in coha.items():
    print(x)
    print(y.most_similar("gay"))

# --> work has a different meaning before 1850

#%% visualize word embeddings over time (PCA mit keyword als passiv)

semchange.semchange(google, "gay", rangelow=1810, rangehigh=2000, rangestep=60, export=False)
semchange.semchange(coha, "gay", rangelow=1820, rangehigh=2010, rangestep=60, export=False)


#%% association of work with different dimension

# set up dictionary and define "work"

keywords = dict()

keywords['work'] = [
    "work", "works", "worked", "working", "job", "jobs",
    "career",
    "profession", "professions", "professional",
    "occupation", "occupations",
    "employment", "employed",
    "labor", "labors"
]

for i in keywords['work']:
    for year, model in google.items():
        if model[i].all() == google[1840]['biology'].all():
            if year >= 1850:
                print(str(year) + ": " + i)



#%% smith: toil (einzelne Begriffe anzeigen?)

keywords['toil'] = [
    "hard", "struggle", "toil", "trouble", "suffer", "endure", "arduous", "strenuous", "grind"
]

for i in keywords['toil']:
    for year, model in google.items():
        if model[i].all() == google[1840]['biology'].all():
            if year >= 1850:
                print(str(year) + ": " + i)


simdim.simdim(google, keywords, 'work', 'toil', trend=3, diff=False, rangelow=1850, rangehigh=2000, rangestep=10)
simdim.simdim(coha, keywords, 'work2', 'toil2', trend=3, diff=False, rangelow=1850, rangehigh=2010, rangestep=10)

simdim.simdim(google, keywords, 'work', 'toil')


keywords['leisure'] = ["leisure", "ease", "rest", "recreation", "relaxation"]
simdim.simdim(google, keywords, 'work', 'toil', 'leisure')
simdim.simdim(google, keywords, 'work', 'toil', 'leisure', diff=True)

for i in keywords['leisure']:
    for year, model in google.items():
        if model[i].all() == google[1840]['biology'].all():
            if year >= 1850:
                print(str(year) + ": " + i)


keywords['leisuretoil'] = keywords['leisure'] + keywords['toil']
simdim.simdim(google, keywords, 'work', 'leisuretoil')


keywords['hard'] = ['hard']
keywords['struggle'] = ['struggle']
keywords['toil'] = ['toil']
keywords['trouble'] = ['trouble']
keywords['suffer'] = ['suffer']
keywords['endure'] = ['endure']
keywords['arduous'] = ['arduous']
keywords['strenuous'] = ['strenuous']

simdim.simdim(google, keywords, 'work', 'hard', 'struggle', 'toil', 'trouble',
              'suffer', 'endure', 'arduous', 'strenuous')


keywords['fun'] = ["fun", "enjoy", "pleasant"]
simdim.simdim(google, keywords, 'work', 'fun')
simdim.simdim(coha, keywords, 'work', 'fun')


keywords['emotion'] = [
    "pleasant", "interesting", "boring", "fulfilling", "meaningful", "meaningless",
    "hard", "struggle", "toil", "trouble", "suffer", "endure", "arduous", "strenuous"
]
simdim.simdim(google, keywords, 'work', 'emotion')

keywords['commodity'] = [
    "market", "exchange", "trade", "hire", "rent"
]
simdim.simdim(google, keywords, 'work', 'commodity')  # nicht sehr spannend


sim_oneterm.sim_oneterm(google, keywords, 'work', 'duty')  # auch teil von "patriot"

sim_oneterm.sim_oneterm(google, keywords, 'work', 'pleasant')




#%% marx: alienation (extrinsic vs. intrinsic)

keywords['mat'] = [
                      "earn", "earning", "earnings",
                      "wage", "wages", "salary", "income", "remuneration", "secure", "pay"
]
keywords['postmat'] = ["interesting", "boring", "fulfilling", "meaningful", "meaningless", "useful", "useless"]

for i in keywords['mat']:
    for year, model in google.items():
        if model[i].all() == google[1840]['biology'].all():
            if year >= 1850:
                print(str(year) + ": " + i)


simdim.simdim(google, keywords, 'work', 'mat', 'postmat', trend=3, diff=False, rangelow=1850, rangehigh=2000, rangestep=10)
simdim.simdim(coha, keywords, 'work2', 'mat', 'postmat', trend=3, diff=False, rangelow=1850, rangehigh=2010, rangestep=10)

simdim.simdim(google, keywords, 'work', 'mat', 'postmat', trend=3, diff=True, rangelow=1850, rangehigh=2000, rangestep=10)
simdim.simdim(coha, keywords, 'work', 'mat', 'postmat', trend=3, diff=True, rangelow=1850, rangehigh=2010, rangestep=10)

keywords['useful'] = ["useful", "society"]
simdim.simdim(google, keywords, 'work', 'useful')


keywords['status'] = [
    "prestigious", "honorable", "esteemed", "influential", "reputable", "distinguished",
    "eminent", "illustrious", "renowned", "acclaimed"
]
simdim.simdim(google, keywords, 'work', 'status')

keywords['social'] = ["colleague", "colleague", "friend", "friends", "people"]
simdim.simdim(google, keywords, 'work', 'social')

simdim.simdim(google, keywords, 'work', 'mat', 'postmat', 'status', 'social')


#%% weber: wealth & religion



keywords['good'] = ['good', 'moral', 'good', 'honest', 'virtuous', 'virtue']
keywords['bad'] = ['evil', 'immoral', 'bad', 'dishonest', 'sinful', 'vice']
keywords['moral'] = keywords['good'] + keywords['bad']
simdim.simdim(google, keywords, 'work', 'good', 'bad')
simdim.simdim(google, keywords, 'work', 'good', 'bad', diff=True)
simdim.simdim(google, keywords, 'work', 'good', 'bad', trend=2, diff=True)
simdim.simdim(google, keywords, 'work', 'moral')  # --> Piketty!
simdim.simdim(coha, keywords, 'work', 'moral')  # --> Piketty!


keywords['rich'] = ["wealth", "wealthy", "rich", "affluence", "affluent"]
keywords['poor'] = ["poor", "poverty", "impoverished", "destitute", "needy"]
keywords['affluence'] = keywords['rich'] + keywords['poor']
simdim.simdim(google, keywords, 'work', 'rich', 'poor')
simdim.simdim(google, keywords, 'work', 'rich', 'poor', diff=True)
simdim.simdim(google, keywords, 'work', 'rich', 'poor', trend=2, diff=True)
simdim.simdim(google, keywords, 'work', 'affluence')  # --> Piketty!


keywords['success'] = ["success", "succeed", "failure", "fail"]
simdim.simdim(google, keywords, 'work', 'success')

keywords['religion'] = ["redemption", "salvation", "god", "religion", "pious"]
simdim.simdim(google, keywords, 'work', 'religion')
simdim.simdim(google, keywords, 'work', 'religion', 'moral', 'affluence')



keywords['vocation'] = ["vocation", "calling", "meaning", "purpose"]
simdim.simdim(google, keywords, 'work', 'vocation')



# work hard?

# Weber: was läuft bei WK?

keywords['patriot'] = ["duty", "country", "patriot", "fatherland", "home"]
simdim.simdim(google, keywords, 'work', 'patriot')


#%% VALIDATION

# faktisch

# connotation von Arbeit mit mann/frau

keywords['male'] = ["male", "man", "boy", "brother", "he", "him", "his", "son"]
keywords['female'] = ["female", "woman", "girl", "sister", "she", "her", "hers", "daughter"]
simdim.simdim(google, keywords, 'work', 'male', 'female', diff=False)
simdim.simdim(google, keywords, 'work', 'male', 'female', diff=True)
simdim.simdim(coha, keywords, 'work', 'male', 'female', diff=False, rangehigh=2010)
simdim.simdim(coha, keywords, 'work', 'male', 'female', diff=True, rangehigh=2010)

sim_oneterm.sim_oneterm(google, keywords, 'female', "housework")


# typische arbeitsgeräte für verschiedene epochen

keywords['plow'] = ['plow']
keywords['telephone'] = ['telephone']
keywords['computer'] = ['computer']

simdim.simdim(google, keywords, 'work', 'plow')
simdim.simdim(google, keywords, 'work', 'telephone')
simdim.simdim(google, keywords, 'work', 'computer')

simdim.simdim(google, keywords, 'work', 'plow', 'telephone', 'computer', trend=3)
simdim.simdim(coha, keywords, 'work', 'plow', 'telephone', 'computer', trend=2, rangehigh=2010)


# historisches wachstum von sektoren

keywords['sector1'] = ["agriculture", "farming", "logging", "fishing", "forestry", "mining"]

keywords['sector2'] = ["manufacturing", "textile", "car", "handicraft"]

keywords['sector3'] = ["service", "social", "information", "advice", "access"]

simdim.simdim(google, keywords, 'work', 'sector1', 'sector2', 'sector3', trend=3)
simdim.simdim(coha, keywords, 'work', 'sector1', 'sector2', 'sector3', trend=2, rangehigh=2010)


# typisch weibliche/männliche Berufe

keywords['male'] = ["male", "man", "boy", "brother", "he", "him", "his", "son"]
keywords['female'] = ["female", "woman", "girl", "sister", "she", "her", "hers", "daughter"]


sim_occs.sim_occs(google, keywords, 'mechanic', 'carpenter', 'engineer', 'nurse', "dancer", "housekeeper")
sim_occs.sim_occs(coha, keywords, 'mechanic', 'carpenter', 'engineer', 'nurse', "dancer", "housekeeper")


# SEMANTIC DRIFT

# housework --> work

keywords['housework'] = ["housework", "household"]
simdim.simdim(google, keywords, 'work', 'housework')
simdim.simdim(coha, keywords, 'work', 'housework', rangehigh=2010)


# beziehungsarbeit

keywords['relations'] = ["relationship"]
simdim.simdim(google, keywords, 'work', 'relations')
simdim.simdim(coha, keywords, 'work', 'relations', rangehigh=2010)


# DISKURS: Arbeiterbewegung

keywords['politics'] = ["party", "politics", "movement", "election"]
simdim.simdim(google, keywords, 'work', 'politics')
simdim.simdim(coha, keywords, 'work', 'politics', rangehigh=2010)












# Others

keywords['male'] = ["male", "man", "boy", "brother", "he", "him", "his", "son"]
keywords['female'] = ["female", "woman", "girl", "sister", "she", "her", "hers", "daughter"]

keywords['pos'] = ['good', 'positive', 'welcome', 'acceptable']
keywords['neg'] = ['bad', 'negative', 'unwelcome', 'unacceptable']

keywords['courage'] = ["courage", 'courageous', "brave", "bold"]
simdim.simdim(google, keywords, 'courage', 'male', 'female', diff=True)
simdim.simdim(google, keywords, 'courage', 'pos', 'neg', diff=True)


keywords['anger'] = ["anger", 'angry']
simdim.simdim(google, keywords, 'anger', 'male', 'female', diff=False)
simdim.simdim(google, keywords, 'anger', 'pos', 'neg', diff=True)

keywords['jealous'] = ["jealous", 'jealousy']
simdim.simdim(google, keywords, 'jealous', 'male', 'female', diff=False)


keywords['love'] = ["love", 'loving']
simdim.simdim(google, keywords, 'love', 'male', 'female', diff=False)
simdim.simdim(google, keywords, 'love', 'pos', 'neg', diff=True)


keywords['disgust'] = ["disgust", 'disgusted']
simdim.simdim(google, keywords, 'disgust', 'male', 'female', diff=False)
simdim.simdim(google, keywords, 'disgust', 'pos', 'neg', diff=True)

keywords['fear'] = ["fear", 'fearful']
simdim.simdim(google, keywords, 'fear', 'male', 'female', diff=False)
simdim.simdim(google, keywords, 'fear', 'pos', 'neg', diff=True)

keywords['shame'] = ["shame", 'ashamed']
simdim.simdim(google, keywords, 'shame', 'male', 'female', diff=False)
simdim.simdim(google, keywords, 'shame', 'pos', 'neg', diff=True)


keywords['pride'] = ["pride", 'proud']
simdim.simdim(google, keywords, 'pride', 'male', 'female', diff=False)
simdim.simdim(google, keywords, 'pride', 'pos', 'neg', diff=True)


keywords['grief'] = ["grief", 'grieving']
simdim.simdim(google, keywords, 'grief', 'male', 'female', diff=False)
simdim.simdim(google, keywords, 'grief', 'pos', 'neg', diff=True)


keywords['hysterical'] = ['hysterical', 'hysteria']
simdim.simdim(google, keywords, 'hysterical', 'male', 'female', diff=False)
simdim.simdim(google, keywords, 'hysterical', 'pos', 'neg', diff=True)



keywords['beautiful'] = ["beautiful", 'beauty']
simdim.simdim(google, keywords, 'beautiful', 'male', 'female', diff=False)

simdim.simdim(google, keywords, 'hysterical', 'male', 'female', diff=False)
simdim.simdim(google, keywords, 'hysterical', 'male', 'female', diff=True)

keywords['emotional'] = ['emotional', "emotion", "emotions"]
simdim.simdim(google, keywords, 'work', 'emotional', diff=False)
simdim.simdim(google, keywords, 'emotional', 'male', 'female', diff=True)


keywords['gay'] = ['gay']
keywords['gay1'] = ['humoured', 'flutter', 'sprightly', "joyous", "giddy", "blooming"]
keywords['gay2'] = ['homosexual', "bisexual", "lesbian", "heterosexual"]
simdim.simdim(google, keywords, 'gay', 'gay1', 'gay2', diff=False)


keywords['physical'] = ["physical", "construct", "sew", "plow", "hammer", "manual", "factory"]
keywords['cognitive'] = ["cognitive", "think", "intellectual", "office"]
keywords['social'] = ["social", "care", "people"]
simdim.simdim(google, keywords, 'work', 'physical', 'cognitive', 'social', diff=False)
