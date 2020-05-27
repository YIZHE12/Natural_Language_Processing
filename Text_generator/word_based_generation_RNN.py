import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np

!wget 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/irish-lyrics-eof.txt'

tokenizer = Tokenizer()

file = 'irish-lyrics-eof.txt'
data = open(file, "r")
corpus = []
for x in data:
  corpus.append(x.lower().split())

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

print(tokenizer.word_index)
print(total_words)

tokenizer.texts_to_sequences([corpus[0]])

input_sequences = []
for line in corpus:
  token_list = tokenizer.texts_to_sequences([line])[0] # convert each line to token
  for i in range(1, len(token_list)):
    n_gram_sequence = token_list[:i+1] # generate n-gram, the first one is just the first word with pad 0 in front
    input_sequences.append(n_gram_sequence)
    
max_sequence_len = max([len(x) for x in input_sequences]) # get the max length of the sequence
input_sequences = np.array(pad_sequences(input_sequences,\
                                          maxlen = max_sequence_len,\
                                          padding = 'pre'))

xs, labels = input_sequences[:,:-1], input_sequences[:,-1]
# xs is from the first except the last token
# labels is the last token

ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)
# one hot encoded to get y


model = Sequential()
model.add(Embedding(total_words, 64, input_length=max_sequence_len -1)) # the last word is taken out as the label
model.add(Bidirectional(LSTM(20)))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = 'accuracy')
history = model.fit(xs, ys, epochs = 500, verbose = 1)

import matplotlib.pyplot as plt

def plot_graph(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()

plot_graphs(history, 'accuracy')

seed_text = 'Laurence went to dublin'
next_words = 100

for _ in range(next_words):
  token_list = tokenizer.texts_to_sequences([seed_text])[0]
  token_list = pad_sequences([token_list], maxlen = max_sequence_len-1,padding = 'pre'),
  predicted = model.predict_classes(token_list, verbose = 0)
  output_word = ""
  for word, index in tokenizer.word_index.items():
    if index == predicted:
      output_word = word

"""For a larger corpus, we will soon run into memory issue for one hot encoding our words as the number of unique words increase. One solution is to build a charater based text generator, as we only have 26 characters in English!"""

