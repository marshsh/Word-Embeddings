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

import os
import numpy as np
import keras

import argparse
import itertools
import sys
import codecs
import re
from collections import Counter, Iterator


from sklearn.datasets import fetch_20newsgroups
from keras.datasets import reuters

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

import tools
import arguments as a


import gensim
from gensim.parsing.preprocessing import strip_non_alphanum


class corpus:
    def __init__(self, nameC, fileN, 
                 MAX_NUM_WORDS = 20000,
                 MAX_SEQUENCE_LENGTH = 1000,
                 num_valid = 0.2,
                 num_test = 0.18,
                 ):

        self.num_valid = num_valid
        self.num_test = num_test

        self.loadCorpus(nameC)

        self.removeStopWords()

        self.tokenise_and_pad(MAX_NUM_WORDS, MAX_SEQUENCE_LENGTH)

        self.split_data()



        # self.split_raw_data(num_valid, num_test) # With all stopwords (for w2v)

        # fileN is defined in getCorpus() at the end of this file
        # tools.dumpPickle(fileN, self)


    def __iter__(self):

        print "Not yet" 

    def loadCorpus(self, nameC):

        print "Loading he corpus: ", nameC

        if nameC in ['20NG','20ng','20newsgroups']:
            self.load20NG()
        elif nameC in ['reuters','r']:
            self.loadReuters()
        else:
            print "No valid corpus"



    def load20NG(self):
        print 'Download 20NewsGroups from SciKitLearn'
        newsgroups_dataset = fetch_20newsgroups(subset = 'all',
                                                remove = ['headers','footers'])
                                                # remove = ['headers','footers', 'quotes'])
        print 'Saving texts into lists'
        textsRaw = newsgroups_dataset.data  # list of text samples
        labels = newsgroups_dataset.target.tolist() # actual labels of the corresponding text
        labels_index = { target:i  for i, target in enumerate(newsgroups_dataset.target_names) }


        self.textsRaw = textsRaw
        self.labels = labels
        self.labels_index = labels_index
        print '20NG texts saved. \n'


    def removeStopWords(self):
        print '\nLoading StopWords'
        with open('./data/stopwords_english.txt','r') as f:
            l = f.readlines()
        
        self.stopWords = { x.strip('\r\n'):True for x in l }

        print 'Removing StopWords from texts'
        self.texts = []

        for doc in self.textsRaw:
            doc = doc.split(' ')
            newDoc = [x for x in doc if x not in self.stopWords]
            newDoc = ' '.join(newDoc)
            self.texts.append(newDoc)



    def loadReuters(self):
    	(x_train, y_train), (x_test, y_test) = reuters.load_data(test_split = 0.0)

        self.textsRaw = x_train
        self.labels = y_train
        self.word_index_reuters = reuters.get_word_index(path="reuters_word_index.json")


    def tokenise_and_pad(self, MAX_NUM_WORDS, MAX_SEQUENCE_LENGTH):
        """
        Uses a Tokenizer to index words with numbers, and preserving MAX_NUM_WORDS
        most common words.

        Stores the word_index, and the data (tokenized and paded texts).
        """
        print 'Tokenizing'
        tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
        tokenizer.fit_on_texts(self.texts)
        self.tokenizer = tokenizer
        self.word_index = tokenizer.word_index
        self.inv_index = {v: k for k, v in tokenizer.word_index.items()}

        sequences = tokenizer.texts_to_sequences(self.texts)
        print 'Found %s unique tokens.' % len(self.word_index)

        self.data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        self.labels = to_categorical(np.asarray(self.labels))
        print 'Tokenizing finished. \n'



    def split_data(self):

        x_train, y_train, x_test, y_test, x_val, y_val = \
            self.aux_split_data(self.data, self.labels)

        self.x_train = x_train
        self.y_train = y_train

        self.x_test = x_test
        self.y_test = y_test

        self.x_val = x_val
        self.y_val = y_val



    def aux_split_data(self, data, labels):
        """
        Splits data in Train Test and Validation sets.
        We use a random seed to always obtain the same Validation Set.
        """
        num_valid = self.num_valid
        num_test = self.num_test


        np.random.seed(42)

        indices = np.arange(self.data.shape[0])
        np.random.shuffle(indices)

        # data = np.asarray(data)
        data = data[indices]
        labels = labels[indices]
        
        num_validation_samples = int(num_valid * data.shape[0])
        num_test_samples = int(num_test * data.shape[0])


        self.train_indices = { x:True for x in indices[num_test_samples:-num_validation_samples]}


        x_train = data[num_test_samples:-num_validation_samples]
        y_train = labels[num_test_samples:-num_validation_samples]

        x_test = data[:num_test_samples]
        y_test = labels[:num_test_samples]

        x_val = data[-num_validation_samples:]
        y_val = labels[-num_validation_samples:]

        return x_train, y_train, x_test, y_test, x_val, y_val

    ################################################################################################

    # def get_w2v_iterator():


    def w2v_iterator(self, texts=None, labels=None):

        if not texts:
            texts = self.textsRaw
        if not labels:
            labels = self.labels

        class w2v_documents():
            SPLIT_SENTENCES = re.compile(u"(?:\n\n)|[.!?:]+")

            def __init__(self, corpusA):
                print "W2V iterator"
                self.corpusA = corpusA

            def __iter__(self):
                for i, doc in enumerate(self.corpusA.textsRaw):
                    if i in self.corpusA.train_indices:
                        for sentence in self.SPLIT_SENTENCES.split(doc):
                            sentence.replace('\n',' ')
                            sentence = gensim.utils.simple_preprocess(sentence, deacc=True)
                            yield sentence

        return w2v_documents(self)




    






	################################################################################################

    def stream_x_train(self):
        itera = Stream(self.x_train)
        return itera

    def stream_x_test(self):
        itera = Stream(self.x_test)
        return itera


    def LDA_stream_x_train(self):
        itera = StreamLDA(self.x_train)
        return itera

    def LDA_stream_x_test(self):
        itera = StreamLDA(self.x_test)
        return itera






class Stream(Iterator):
    def __init__(self, data):
        self.stop = data.shape[0]
        self.data = data
        self.i = 0
# 
    def __iter__(self):
        while True:
            yield self.next()
# 
    def next(self):
        if self.i < self.stop:
            vec = self.data[self.i]
            sentence = [str(x) for x in vec if x != 0]
            self.i += 1
            return sentence
        else:
            self.i = 0
            raise StopIteration
# 
    __next__ = next # Python 3 compatibility


class StreamLDA(Iterator):
    def __init__(self, data):
        self.stop = data.shape[0]
        self.data = data
        self.i = 0
# 
    def __iter__(self):
        return self
# 
    def next(self):
        if self.i < self.stop:
            vec = self.data[self.i]
            sentence = [int(x) for x in vec if x != 0]
            bow = BOW(sentence)
            self.i += 1
            return bow
        else:
            self.i = 0
            raise StopIteration
# 
    __next__ = next # Python 3 compatibility



def BOW(sentence):
    dic = {}
    for word in sentence:
        if word in dic:
            dic[word] += 1
        else :
            dic[word] = 1
    r = list(dic.items())
    return r




def getCorpus(nameC, extraName='', 
                 MAX_NUM_WORDS = a.MAX_NUM_WORDS,
                 MAX_SEQUENCE_LENGTH = a.MAX_SEQUENCE_LENGTH,
                 num_valid = a.VALIDATION_SPLIT,
                 num_test = a.TEST_SPLIT,
                 reCalculate = False
                 ):


    extraName = '[SeqLen_{}]'.format(MAX_SEQUENCE_LENGTH)


    fileName = nameC + "." + extraName + ".ready"
    fileN = os.path.join("data",nameC,fileName)

    if (not reCalculate) & os.path.exists(fileN):
        print "Loading corpus {}".format(nameC)
        corpusA = tools.loadPickle(fileN)
        print "Corpus loaded"
        return corpusA

    else :
        print "Constructing corpus {}".format(nameC)
        corpusA = corpus(nameC, fileN,
                 MAX_NUM_WORDS = MAX_NUM_WORDS,
                 MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH,
                 num_valid = num_valid,
                 num_test = num_test
                 )

        tools.dumpPickle(fileN, corpusA)

        print "Corpus constructed and saved"
        return corpusA









