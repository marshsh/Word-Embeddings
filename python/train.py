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

# Our classes:
import embeddings
import corpus
import kerasModel as km
import docsKerasModel as docsKM
import arguments as a
import tools
import gensimW2V

from time import time, localtime
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping







def getEmbeddingLayer(args, embedding_type, corpus, MAX_NUM_WORDS=20000, EMBEDDING_DIM=300):
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
    elif embedding_type == 'w2v+topicAvg':
        embeddings_dic = embeddings.w2v_and_topicAvg_embeddings( args.filePrefix, args.corpus, reCalculate=args.reCalculate)
    elif embedding_type == 'w2v+context':
        embeddings_dic = embeddings.context_and_word2vec_embeddings( args.filePrefix, args.corpus, reCalculate=args.reCalculate)
    elif embedding_type == 'w2v+context_logN':
        embeddings_dic = embeddings.context_and_word2vec_embeddings( args.filePrefix, args.corpus, reCalculate=args.reCalculate, logNormal=True)
    elif embedding_type == 'gensim':
        embeddings_dic = gensimW2V.gensimW2V_embeddings(args.corpus, epochsN=args.epochsN, reCalculate=args.reCalculate)
    elif embedding_type == 'smh_reduced':
        embeddings_dic = embeddings.smh_reduced_topicN( filePrefix, topicN=args.topicN, topTopicWords=a.TOP_TOPIC_WORDS, reCalculate=args.reCalculate)
    elif embedding_type == 'smh_reduced_logN':
        embeddings_dic = embeddings.smh_reduced_topicN( filePrefix, topicN=args.topicN, topTopicWords=a.TOP_TOPIC_WORDS, reCalculate=args.reCalculate, logNormal=True)


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
                                input_length=a.MAX_SEQUENCE_LENGTH,
                                trainable=False)

    return embedding_layer





def getWordsModel(model_type, embedding_layer, numLabels, MAX_SEQUENCE_LENGTH, incomplete):
    if model_type == "conv":
        model = km.getConvModel(embedding_layer, numLabels, MAX_SEQUENCE_LENGTH=a.MAX_SEQUENCE_LENGTH, incomplete=incomplete)
    if model_type == "conv+lstm":
        model = km.getConvLSTMmodel(embedding_layer, numLabels, MAX_SEQUENCE_LENGTH=a.MAX_SEQUENCE_LENGTH, incomplete=incomplete)
    if model_type == "lstm":
        model = km.getLSTMmodel(embedding_layer, numLabels, MAX_SEQUENCE_LENGTH=a.MAX_SEQUENCE_LENGTH, incomplete=incomplete)

    return model


def getCorpus(args):
    corpusA = corpus.getCorpus(args.corpus, args.nameCorpus, a.MAX_NUM_WORDS, a.MAX_SEQUENCE_LENGTH, a.VALIDATION_SPLIT, a.TEST_SPLIT, args.reCalculate)
    return corpusA



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
# Spliting into two different types of neural network models. The ones intended for word embeddings and the 
# ones intended for document embeddings.
        if args.layout == 'words' :
            embedding_layer = getEmbeddingLayer(args, args.embedding_type, corpusA, a.MAX_NUM_WORDS, a.EMBEDDING_DIM)
            model = getWordsModel(args.kerasModel, embedding_layer, numLabels, a.MAX_SEQUENCE_LENGTH, incomplete=False)
        elif args.layout == 'docs' :
            model = docsKM.getDocsModel(args, corpusA, args.kerasModel, numLabels, a.MAX_SEQUENCE_LENGTH, a.MAX_NUM_WORDS, a.EMBEDDING_DIM)




    callBackName = "{}__M{}-D{}-time{}:{}".format( 
        args.nameBoard, localtime().tm_mon, 
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

        # earlyStopping = EarlyStopping(
        #     monitor='val_acc',
        #     min_delta=0,
        #     patience=15,
        #     verbose=0,
        #     mode='auto',
        #     baseline=None,
        #     restore_best_weights=False
        #     )


    print 'Training model.'
    history = model.fit(corpusA.x_train, corpusA.y_train,
              batch_size=18,
              epochs=a.EPOCHS,
              validation_data=(corpusA.x_test, corpusA.y_test),
              callbacks=[tensorboard, checkPoint])
              # callbacks=[tensorboard])

    history_dic = history.history

    model.save(modelName)

    histName = os.path.join("history",callBackName)
    tools.dumpPickle(histName, history_dic)






if __name__ == "__main__":


    args = a.preMain()

    main(args)


