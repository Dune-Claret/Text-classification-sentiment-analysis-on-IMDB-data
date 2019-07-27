# A brief introduction and specifications on the notebook
I worked under the direction of Patrice Bellot, member of the LSIS (CNRS UMR 7296) of Marseille and DIMAG team leader, with who I was supposed to do a 2 weeks training in June 2019. I deviated a little from what I have been ask to do, because I wanted to delve into the subject, so I worked for a month to developp and compare neural networks for Text Classification, to visualize word embeddings and by being more restrictive to visualize only those who are directly linked to emotions. To do so I used a a set of 50,000 highly-polarized reviews from the Internet Movie Database and the pre trained embedding GloVe.

The first part is about formating data, then we come to the different neural networks and their results on both train and test data, and finally we visualize word embeddings obtained with the best model.

In the subparts Train of Comparison of different architectures I call the history of the models which you can find in the history file. Notice that to fit the models so to get the histories I had to use Python intead of Jupyter as its kernel died, all the codes are stored in the scripts file.

Some folders or files are too big to be uploaded on Github, so I give you the link to download it :

- EmoLex (NRC Word-Emotion Association Lexicon) a database of English words which clusters words associated with sentiments (positive, negative) and emotions (anger, fear, anticipation, trust, surprise, sadness, joy, disgust)  : 
http://sentiment.nrc.ca/lexicons-for-research/

- GloVe (Global Vectors for Word Representation) from which I used the pre-trained word vectors :
https://nlp.stanford.edu/projects/glove/

- Large Movie Review Dataset a set of 50,000 highly-polarized reviews from the Internet Movie Database :
https://www.kaggle.com/pankrzysiu/keras-imdb

Notice that there are also all_embs.npy, emb_mean.pkl, embeddings_index.pickle, roots_emotions.pkl and texts.txt that I couldn't upload and I call them in my Python files (on the scripts folder) but also in the notebook. So they need to be created once in the notebook. 
