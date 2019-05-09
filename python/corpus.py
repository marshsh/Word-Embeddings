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


class corpus:
    def __init__(self, nameC, fileN, 
                 MAX_NUM_WORDS = 20000,
                 MAX_SEQUENCE_LENGTH = 1000,
                 num_valid = 0.2,
                 num_test = 0.18,
                 ):

        self.loadCorpus(nameC)

        self.tokenise_and_pad(MAX_NUM_WORDS, MAX_SEQUENCE_LENGTH)

        self.split_data(num_valid, num_test)

        tools.dumpPickle(fileN, self)





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
                                                remove = ('headers','footers', 'quotes'))
        print 'Saving texts into lists'
        texts = newsgroups_dataset.data  # list of text samples
        labels = newsgroups_dataset.target.tolist() # actual labels of the corresponding text
        labels_index = { target:i  for i, target in enumerate(newsgroups_dataset.target_names) }


        self.texts = texts
        self.labels = labels
        self.labels_index = labels_index
        print '20NG texts saved. \n'



    def loadReuters(self):
    	(x_train, y_train), (x_test, y_test) = reuters.load_data(test_split = 0.0)

        self.texts = x_train
        self.labels = y_train
        self.word_index_reuters = reuters.get_word_index(path="reuters_word_index.json")


    def tokenise_and_pad(self, MAX_NUM_WORDS, MAX_SEQUENCE_LENGTH):
        """
        Uses a Tokenizer to index words with numbers, and preerving MAX_NUM_WORDS
        most common words.

        Stores the word_index, and the data (tokenized and paded texts).
        """
        print 'Tokenizing'
        tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
        tokenizer.fit_on_texts(self.texts)
        self.tokenizer = tokenizer
        self.word_index = tokenizer.word_index

        sequences = tokenizer.texts_to_sequences(self.texts)
        print 'Found %s unique tokens.' % len(self.word_index)

        self.data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        self.labels = to_categorical(np.asarray(self.labels))
        print 'Tokenizing finished. \n'

    def split_data(self, num_valid, num_test):
        """
        Splits data in Train Test and Validation sets.
        We use a random seed to always obtain the same Validation Set.
        """


        np.random.seed(42)

        indices = np.arange(self.data.shape[0])
        np.random.shuffle(indices)

        data = self.data[indices]
        labels = self.labels[indices]
        
        num_validation_samples = int(num_valid * data.shape[0])
        num_test_samples = int(num_test * data.shape[0])


        self.x_train = data[num_test_samples:-num_validation_samples]
        self.y_train = labels[num_test_samples:-num_validation_samples]

        self.x_test = data[:num_test_samples]
        self.y_test = labels[:num_test_samples]

        self.x_val = data[-num_validation_samples:]
        self.y_val = labels[-num_validation_samples:]



	################################################################################################

    def stream_x_train(self):
        itera = Stream(self.x_train)
        return itera

    def stream_x_test(self):
        itera = Stream(self.x_test)
        return itera





class Stream(Iterator):
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
            for vec in self.data:
                sentence = [str(x) for x in vec if x != 0]
                self.i += 1
            return sentence
        else:
            self.i = 0
            raise StopIteration
# 
    __next__ = next # Python 3 compatibility






def getCorpus(nameC, extraName='', 
                 MAX_NUM_WORDS = a.MAX_NUM_WORDS,
                 MAX_SEQUENCE_LENGTH = a.MAX_SEQUENCE_LENGTH,
                 num_valid = a.VALIDATION_SPLIT,
                 num_test = a.TEST_SPLIT,
                 reCalculate = False
                 ):

    fileName = nameC + extraName + ".ready"
    fileN = os.path.join("data",nameC,fileName)

    if (not reCalculate) & os.path.exists(fileN):
        print "Loading corpus {}".format(nameC)
        corpusA = tools.loadPickle(fileN)
        print "Corpus loaded"
        return corpusA

    else :
        print "Constructing corpus {}".format(nameC)
        corpusA = corpus(nameC, fileN,
                 MAX_NUM_WORDS = 20000,
                 MAX_SEQUENCE_LENGTH = 1000,
                 num_valid = 0.2,
                 num_test = 0.18
                 )
        print "Corpus constructed and saved"
        return corpusA