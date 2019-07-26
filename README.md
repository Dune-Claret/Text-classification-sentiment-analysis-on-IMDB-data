# Text_classification_IMDB
I worked under the direction of Patrice Bellot, member of the LSIS (CNRS UMR 7296) of Marseille and DIMAG team leader, with who I was supposed to do a 2 weeks training in June 2019. I deviated a little from what I have been ask to do, because I wanted to delve into the subject, so I worked for a month to developp and compare neural networks for Text Classification, to visualize word embeddings and by being more restrictive to visualize only those who are directly linked to emotions. To do so I used a a set of 50,000 highly-polarized reviews from the Internet Movie Database.

To fit the models I had to use Pyzo intead of Jupyter as the kernel died, all the codes are stored in the field code.

Some folders or files are too big to be uploaded on Github, so I give you the link to download it :

- EmoLex (NRC Word-Emotion Association Lexicon) a database of English words which clusters words associated with sentiments (positive, negative) and emotions (anger, fear, anticipation, trust, surprise, sadness, joy, disgust)  : 
http://sentiment.nrc.ca/lexicons-for-research/

- GloVe (Global Vectors for Word Representation) from which I used the pre-trained word vectors :
https://nlp.stanford.edu/projects/glove/

- Large Movie Review Dataset a set of 50,000 highly-polarized reviews from the Internet Movie Database :
https://www.kaggle.com/pankrzysiu/keras-imdb

Notice that there are also all_embs.npy, emb_mean.pkl, embeddings_index.pickle, roots_emotions.pkl and texts.txt that I couldn't upload and I call them in my Pyzo files (on the folder code) but also in the notebook. So they need to be executed once, as they are defined in the notebook. 
