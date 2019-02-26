#!/usr/bin/env python
# -*- coding: utf-8
#
# Mariana Gleason Freidberg <mar.freig@gmail.com>
# IIMAS, UNAM
# 2019
#
# -------------------------------------------------------------------------
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# -------------------------------------------------------------------------
"""
Functions to fetch and store the 20newsgroup dataset in corpus format.
"""
import argparse
import itertools
import sys
import codecs
import re
from collections import Counter
from sklearn.datasets import fetch_20newsgroups

import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.initializers import Constant

import embeddings











class corpus:
    def __init__(self, type, 
                 MAX_NUM_WORDS=MAX_NUM_WORDS,
                 num_valid = 0.2,
                 num_test = 0.16
                 ):

        loadCorpus()

        tokenise_and_pad(MAX_NUM_WORDS)

        split_data(num_valid, num_test)



    def loadCorpus(type):

        print "Loading he corpus: ", type

        if type == "20NG":
            load20NG()
        elif type == "reuters":
            print "Not yet"
        else:
            print "No valid corpus"


    def load20NG():
        print 'Download 20NewsGroups from SciKitLearn'
        newsgroups_dataset = fetch_20newsgroups(subset = 'all',
                                                remove = ('headers','footers', 'quotes'))
        print 'Saving texts into lists'
        texts = newsgroups_dataset.data  # list of text samples
        labels = newsgroups_dataset.target.tolist() # actual labels of the corresponding text
        labels_index = { target:i  for i, target in enumerate(newsgroups_dataset.target_names) }


        self.texts = texts
        self.labels = labels
        self.labels_index = labels_index


    def tokenise_and_pad(MAX_NUM_WORDS):
        """
        Uses a Tokenizer to index words with numbers, and preerving MAX_NUM_WORDS
        most common words.

        Stores the word_index, and the data (tokenized and paded texts).
        """
        tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
        tokenizer.fit_on_texts(self.texts)
        self.tokenizer = tokenizer
        self.word_index = tokenizer.word_index

        sequences = tokenizer.texts_to_sequences(self.texts)
        print 'Found %s unique tokens.' % len(self.word_index)

        self.data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        self.labels = to_categorical(np.asarray(self.labels))


    def split_data(num_valid, num_test):
        """
        Splits data in Train Test and Validation sets.
        We use a random seed to always obtain the same Validation Set.
        """

        np.random.seed(42)

        indices = np.arange(self.data.shape[0])
        np.random.shuffle(indices)

        self.data = self.data[indices]
        self.labels = self.labels[indices]
        
        num_validation_samples = int(num_valid * data.shape[0])
        num_test_samples = int(num_test * data.shape[0])


        self.x_train = data[num_test_samples:-num_validation_samples]
        self.y_train = labels[num_test_samples:-num_validation_samples]

        self.x_test = data[:num_test_samples]
        self.y_test = labels[:num_test_samples]

        self.x_val = data[-num_validation_samples:]
        self.y_val = labels[-num_validation_samples:]















def newsgroupsTrain( embedding_type ):
    """
    Utilizes pre-trained glove embbedings located at de following GLOVE_DIR dir
    and creates a Keras Embbeding Layer
    """


# OJOOOOOO:    Estabamos cambiando los embeddings a otro archivo.


    print "Loading embbeding-dictionary"

    if embedding_type == "glove":
        embeddings_dic = gloveEmbbedingDic()
    else :
        print "Embbeding type not supported yet."


    print 'Loaded %s word vectors into embeddings_dic dictionary.' % len(embeddings_dic)




def get_XY_20NG():



    print 'Shape of data tensor:', data.shape
    print 'Shape of label tensor:', labels.shape

    # split the data into a training set, test set, and validation set








    print 'Preparing embedding matrix. Using specified embedding dictionary.'

    # prepare embedding matrix
    num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

    for word, i in word_index.items():
        if i > MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_dic.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)



    print "Creating Keras Sequential Model"

    # train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(len(labels_index), activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])


    print 'Training model.'

    model.fit(x_train, y_train,
              batch_size=128,
              epochs=15,
              validation_data=(x_val, y_val))






def main():
    """
    Main function
    """





    # try:
    #     parser = argparse.ArgumentParser(
    #         description="Downloads the 20 newsgroups corpus and creates reference and labels files")
    #     parser.add_argument("dirpath",
    #                         help="directory where the reference and labels files are to be saved")

    #     args = parser.parse_args()


            
    # except SystemExit:
    #     print "for help use --help"
    #     sys.exit(2)



def main2():
    print "enteres main2"
    a = corpus(args.wordVecs)



if __name__ == "__main__":

    MAX_SEQUENCE_LENGTH = 1000
    MAX_NUM_WORDS = 20000
    EMBEDDING_DIM = 100
    VALIDATION_SPLIT = 0.2



    parser = argparse.ArgumentParser()
    parser.add_argument("wordVecs", choices=['smh', 'oneH', 'word2vec', 'glove',
                        'contextVec', 'smh + contextVec', 'word2vec + contextVec'], 
                        help="El tipo de representacion de las palabras \
                        en el documento. (Sustituimos las palabras de cada documento \
                        por dichos vectores, y sobre esa secuencia entrenamos la LSTM)")

    args = parser.parse_args()

    print "Training 20 News Groups with ", args.wordVecs, " embbedings"

    # newsgroupsTrain(args.wordVecs)
    main2()


