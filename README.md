The Meaning of Work
=================
Code associated with my paper on the Meaning of Work. Preprint available here: https://osf.io/preprints/socarxiv/pxj7m/

## Instructions
<i>Load data</i>
<br/>
 * Clone repository.
 * Create a folder named "data" in project main directory.<br/>
 * Download data from https://nlp.stanford.edu/projects/histwords/ (SGNS All English (1800s-1990s)), unzip, and save ".txt"-files in data folder.<br/>
 * Use "convert embeddings.py" to convert word vectors into the format used by the gensim package.<br/>
 <br/>

<i>Analyze data using the following functions</i>
<br/>
 * semchange: Visualizes how a given input term changes its position in the embeddings space over time relative to its nearest neighbors.<br/>
 * listsim: Estimates the cosine similarity for all combinations of words within a word list and returns a dataframe. Can be used to check the internal consistency of word lists.<br/>
 * simdim: Estimates the cosine similarity over time between a word list ("key") and any number of additional word lists. Creates one shared figure for all word lists as output.<br/>
 * simdim2: Same functionality as simdim, but slightly different method. Simdim first takes the mean vector of word lists and then estimates the distance between these vectors. Simdim2 estimates the distance between all word-pair combinations across two word lists and then calculates the average of these values. Simdim2 is considerably slower but can be used for robustness checks.<br/>

Also check out my interactive web app to explore diachronic word embeddings: https://huggingface.co/spaces/BrickTamland/Histwords-Webapp