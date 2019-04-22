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
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding
from tensorflow.keras.initializers import Constant

import os
import numpy as np
import json

# Our classes:
import embeddings
import corpus
import kerasModel as km


from time import time, localtime
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping







def getEmbeddingLayer(embedding_type, corpus, MAX_NUM_WORDS=20000, EMBEDDING_DIM=300):
    print "Loading embbeding-dictionary"


    if embedding_type == "glove":
        embeddings_dic = embeddings.glove_get_embeddings()
    elif embedding_type == "w2v":
        embeddings_dic = embeddings.word2vec_get_embeddings(args.filePrefix, corpus, reCalculate=args.reCalculate)
    elif embedding_type == "smh":
        embeddings_dic = embeddings.smh_get_embeddings( args.filePrefix, reCalculate=args.reCalculate)
    elif embedding_type == "smh_logN":
        embeddings_dic = embeddings.smh_get_embeddings( args.filePrefix, reCalculate=args.reCalculate, logNormal=True)
    elif embedding_type == "topicAvg":
        embeddings_dic = embeddings.topicAvg_get_embeddings(args.filePrefix, args.corpus, reCalculate=args.reCalculate)

    elif embedding_type == 'contextVec':
        embeddings_dic = embeddings.contextSMH_get_embeddings( args.filePrefix, args.size, reCalculate=args.reCalculate)
    elif embedding_type == 'contextVec_logN':
        embeddings_dic = embeddings.contextSMH_get_embeddings( args.filePrefix, args.size, reCalculate=args.reCalculate, logNormal=True)
    elif embedding_type == "glove+contextVec":
        embeddings_dic = embeddings.glove_and_context_embeddings( args.filePrefix, args.size, reCalculate=args.reCalculate)
    elif embedding_type == "glove+contextVec_logN":
        embeddings_dic = embeddings.glove_and_context_embeddings( args.filePrefix, args.size, reCalculate=args.reCalculate, logNormal=True)
    elif embedding_type == "w2v+smh":
        embeddings_dic = embeddings.smh_and_word2vec_embeddings( args.filePrefix, args.corpus, reCalculate=args.reCalculate)
    elif embedding_type == "w2v+smh_logN":
        embeddings_dic = embeddings.smh_and_word2vec_embeddings( args.filePrefix, args.corpus, reCalculate=args.reCalculate, logNormal=True)

    elif embedding_type == 'oneH':
        # embeddings_dic = 
        print "oneH in progress"
    else :
        print "Embbeding type not supported yet."



    EMBEDDING_DIM = len(embeddings_dic[embeddings_dic.keys()[0]])


    print 'Preparing embedding matrix. Using ', embedding_type ,' embedding dictionary.'
    # prepare embedding matrix
    num_words = min(MAX_NUM_WORDS, len(corpus.word_index)) + 1
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

    for word, i in corpus.word_index.items():
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

    return embedding_layer





def getModel(model_type, embedding_layer, numLabels, MAX_SEQUENCE_LENGTH):
    if model_type == "conv":
        model = km.getConvModel(embedding_layer, numLabels, MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH)
    if model_type == "conv+lstm":
        model = km.getConvLSTMmodel(embedding_layer, numLabels, MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH)
    if model_type == "lstm":
        model = km.getLSTMmodel(embedding_layer, numLabels, MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH)

    return model


def getCorpus(args):
    corpus.getCorpus(args.corpus, args.nameCorpus, MAX_NUM_WORDS, MAX_SEQUENCE_LENGTH, VALIDATION_SPLIT, TEST_SPLIT, args.reCalculate)


def main(args):


    modelName = "./savedModels/{}_{}".format(args.corpus, args.embedding_type)
    if args.logNormal :
        modelName += "_logN"


    if args.restore & (os.path.isfile(modelName)):
        print "Restoring last instance of Model : ", modelName
        model = keras.models.load_model(modelName)
    else :

        print "Creating Model : ", modelName

        corpusA = getCorpus(args)
        numLabels = len(corpusA.y_train[0]) # labels are in cathegorical shape, this is the number of clases

        embedding_layer = getEmbeddingLayer(args.embedding_type, corpusA, MAX_NUM_WORDS, EMBEDDING_DIM)

        model = getModel(args.kerasModel, embedding_layer, numLabels, MAX_SEQUENCE_LENGTH)


        callBackName = "{}__{}-{}--{}:{}".format( 
            args.name, localtime().tm_mon, 
            localtime().tm_mday, localtime().tm_hour, localtime().tm_min)

        tensorboard = TensorBoard( log_dir="logs/"+callBackName,
            write_graph=False, histogram_freq=1, write_grads=True
            )

        checkPoint = ModelCheckpoint(
            "checkPoints/"+callBackName,
            monitor='val_acc',
            verbose=0,
            save_best_only=True,
            save_weights_only=False,
            mode='auto',
            period=1
            )

        earlyStopping = EarlyStopping(
            monitor='val_acc',
            min_delta=0,
            patience=15,
            verbose=0,
            mode='auto',
            baseline=None,
            restore_best_weights=False
            )



    print 'Training model.'
    history = model.fit(corpusA.x_train, corpusA.y_train,
              batch_size=18,
              epochs=EPOCHS,
              validation_data=(corpusA.x_test, corpusA.y_test),
              callbacks=[tensorboard, checkPoint])

    history_dic = history.history

    model.save(modelName)

    histName = os.path.join("history",callBackName)
    with open(histName,'w') as f:
        json.dump(history_dic, f)







def preMain(aaaargs=[]):
    """
    With this method, you get the 'args' object you would've gotten had you run 
    this script directly with the corresponding options.

    USAGE:
    In python environment:

    import train
    args = train.preMain(["-e", "w2v", "-c", "20ng"])
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_type", "-et", "-e", 
                        choices=['smh', 'oneH', 'w2v', 'glove', 'contextVec', 'w2v+smh', 'topicAvg', 'w2v+contextVec', 'glove+contextVec'], 
                        help="Type of word representation used to train the model.")
    parser.add_argument("--corpus", "-c", 
                        choices=[ '20NG', '20ng', 'r', 'reuters', 'w', 'wiki', 'wikipedia'],
                        help="Corpus to be used", 
                        default='20ng')
    parser.add_argument("--kerasModel", "-km", "-model", "-keras", 
                        choices=['conv', 'lstm', 'conv+lstm'],
                        default='conv+lstm',
                        help="Architecture of the neural network used to classify texts")
    
    parser.add_argument("--convFilters", "-convF", type=int, default=32, help="Number of Conv1D Filters used in conv1D Keras Model")

    parser.add_argument("--lstmNeurons", "-lstmN", "-lstm", type=int, default=128, help="Number of neurons in lstm layer of Keras Model")

    parser.add_argument("--size", type=int)

    parser.add_argument("--nameBoard", type=str)

    parser.add_argument("--nameCorpus", type=str, default='')

    parser.add_argument("--reCalculate", 
                        help="re-calculate chosen word-vector embeddings", 
                        action="store_true")
    
    parser.add_argument("--logNormal", 
                        help="utilize log-Normalization in smh word-vector embeddings", 
                        action="store_true")

    parser.add_argument("--restore", 
                        help="restore Keras model from latest training moment", 
                        action="store_true")


    if aaaargs:
        args = parser.parse_args(aaaargs)
    else :
        args = parser.parse_args()

    print " \n Training ", args.corpus, "with ", args.embedding_type, " embbedings"


    if args.logNormal:
        args.embedding_type += "_logN"
        print "Using _logNormal smh embeddings."


    if args.corpus in ['20NG','20ng']:
        args.corpus = '20newsgroups'
    if args.corpus in ['r', 'reuters']:
        args.corpus = 'reuters'
    if args.corpus in ['w', 'wiki', 'wikipedia']:
        args.corpus = 'wikipedia'


# PREFIX fix
    # Adding file-prefix to have a well organized way of saving pre-calculated embeddings.
    filePrefix = 'data/'
    if args.corpus in ['20NG', '20ng', '20newsgroups']:
        filePrefix = os.path.join(filePrefix, '20newsgroups', '20newsgroups')
    elif args.corpus in ['r', 'reuters']:
        filePrefix = os.path.join(filePrefix, 'reuters', 'reuters')
    elif args.corpus in ['w', 'wiki', 'wikipedia']:
        filePrefix = os.path.join(filePrefix, 'wikipedia', 'wiki')
    else :
        print " \n Couldn't find corresponding filePrefix \n"

    if args.logNormal:
        filePrefix += '_logNorm'  #If you change this string '_logNorm' you have to change it in embeddings.smh_get_model() too

    args.filePrefix = filePrefix

    print "\n \n \n \n" + filePrefix + "\n \n"




# NAME fix
    if not args.nameBoard:
        args.nameBoard = ''
    else :
        args.nameBoard =  '(' + args.nameBoard + ')'

    args.nameBoard = "{}_{}_{}_[{}-{}]_{}".format(args.corpus, args.embedding_type, 
        args.kerasModel, args.convFilters, args.lstmNeurons, args.nameBoard)




    if args.size == None:
        args.size = windowSize



    return args



# Global Variables

EPOCHS = 100


MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100

VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.18

windowSize = 5


if __name__ == "__main__":


    args = preMain()

    main(args)


