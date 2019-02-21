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






def gloveEmbbedingDic():
    """
    Returns dictionary with pre-trained word embbedings.
    glove.6B.2 must be downloaded to "pwd/3rdParty/glove.6B.2/glove.6B.100d.txt"
    before running this script.
    """

    # dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.getcwd()
    GLOVE_DIR = os.path.join( dir_path, "3rdParty", "glove.6B.2")

    embeddings_dic = {}
    with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_dic[word] = coefs

    return embeddings_dic






def newsgroupsTrain( embedding_type ):
    """
    Utilizes pre-trained glove embbedings located at de following GLOVE_DIR dir
    and creates a Keras Embbeding Layer
    """


    print "Loading embbeding-dictionary"

    if embedding_type == "glove":
        embeddings_dic = gloveEmbbedingDic()
    else :
        print "Embbeding type not supported yet."


    print 'Loaded %s word vectors into embeddings_dic dictionary.' % len(embeddings_dic)



    print 'Download 20NewsGroups from SciKitLearn'

    newsgroups_dataset = fetch_20newsgroups(subset = 'all',
                                            remove = ('headers','footers', 'quotes'))


    print 'Saving texts into lists'

    texts = newsgroups_dataset.data  # list of text samples

    labels = newsgroups_dataset.target.tolist() # actual labels of the corresponding text

    labels_index = { target:i  for i, target in enumerate(newsgroups_dataset.target_names) }






    print 'Preparing Tokenizer to be able to use Keras Imput Layer easily'


    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print 'Found %s unique tokens.' % len(word_index)

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    labels = to_categorical(np.asarray(labels))
    print 'Shape of data tensor:', data.shape
    print 'Shape of label tensor:', labels.shape

    # split the data into a training set, test set, and validation set
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    x_train = data[:-num_validation_samples]
    y_train = labels[:-num_validation_samples]
    x_val = data[-num_validation_samples:]
    y_val = labels[-num_validation_samples:]





    print 'Preparing embedding matrix. Using specifyed embedding dictionary.'

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

    newsgroupsTrain(args.wordVecs)



