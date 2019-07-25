## Importations
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import nltk
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras.callbacks import ModelCheckpoint
from sklearn.manifold import TSNE
from nltk.stem.porter import PorterStemmer
from keras import backend as K

## Parameters
# Not all words that are in our IMDB vocabulary might be in the GloVe embeddings and for missing words we use random embeddings with the same mean and standard deviation as the GloVe embeddings

cwd = os.getcwd()
if cwd != '/Users/macbook/Documents/Stages/CNRS/IMDB' :
    os.chdir('./Documents/Stages/CNRS/IMDB')

os.environ['KMP_DUPLICATE_LIB_OK']='True'

all_embs = np.load('all_embs.npy')    
emb_mean = all_embs.mean()
emb_std = all_embs.std()

x_train = np.load('x_train.npy')
x_val = np.load('x_val.npy')
y_train = np.load('y_train.npy')
y_val = np.load('y_val.npy')
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')
print(x_test.shape, y_test.shape)

## Model
max_words = 10000
maxlen = 100
with open("texts.txt", "rb") as fp:   
    texts = pickle.load(fp)
tokenizer = Tokenizer(num_words=max_words) 
tokenizer.fit_on_texts(texts)
embedding_dim = 100
word_index = tokenizer.word_index
nb_words = min(max_words, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embedding_dim))
pickle_in = open("embeddings_index.pickle","rb")
embeddings_index = pickle.load(pickle_in)


# Loop over all words in the word index
for word, i in word_index.items():
    if i >= max_words: 
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: 
        embedding_matrix[i] = embedding_vector
  

# Model with an embedding layer not trainable & 2 dense layers (1 hidden) & 10 epochs
K.clear_session()
#for k in range(10) :
model_glove = Sequential()
model_glove.add(Embedding(max_words, embedding_dim, input_length=maxlen, weights = [embedding_matrix], trainable = False))
model_glove.add(Flatten())
model_glove.add(Dense(32, activation='relu'))
model_glove.add(Dense(1, activation='sigmoid'))

model_glove.summary()

# model_glove.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# filepath="model_not_train1_weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# callbacks_list = [checkpoint]
# 
# model_glove_not_train_1_history = model_glove.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val), verbose = 2, callbacks = callbacks_list)

## Test
K.clear_session()

model_not_trainable1 = Sequential()
model_not_trainable1.add(Embedding(max_words, embedding_dim, input_length=maxlen, weights = [embedding_matrix], trainable = False))
model_not_trainable1.add(Flatten())
model_not_trainable1.add(Dense(32, activation='relu'))
model_not_trainable1.add(Dense(1, activation='sigmoid'))
model_not_trainable1.summary()

model_not_trainable1.load_weights("model_not_train1_weights-improvement-01-0.72.hdf5")
model_not_trainable1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

print(model_not_trainable1.evaluate(x_test, y_test, verbose = 1))

## Visualization
# We want to visualize word embedding weights obtained from trained model

conv_embds = model_glove.layers[0].get_weights()[0]
with open('roots_emotions.pkl', 'rb') as f:
    roots_emotions = pickle.load(f)

# Plotting function
def plot_words(data, words, roots_emotions, start, stop, step):
    cond = np.array([word in roots_emotions for word in words])
    x = data[start:stop:step,0][cond]
    y = data[start:stop:step, 1][cond]
    plt.scatter(x, y, marker = ".", s = 7)
    for i, word in enumerate(np.array(words[start:stop:step])[cond]):
        plt.text(x[i]+1.5, y[i]+1.5, word, fontsize = 7)
        assert word in roots_emotions
    print (np.array(words[start:stop:step])[cond].shape)
    plt.show()

# Visualize words in two dimensions 
words = [tokenizer.index_word[i] for i in range (1,10001)]
conv_tsne_embds = TSNE(n_components=2).fit_transform(conv_embds)
plot_words(conv_tsne_embds, words, roots_emotions, 0, 10000, 1)