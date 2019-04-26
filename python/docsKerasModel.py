

import numpy as np
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM
from tensorflow.keras.models import Model, Sequential

import kerasModel as km
import train
import embeddings as emb


def getDocsModel(args, corpusA, model_type, numLabels, MAX_SEQUENCE_LENGTH, MAX_NUM_WORDS, EMBEDDING_DIM):

    first_embedding_layer = train.getEmbeddingLayer(args, "w2v", corpusA, MAX_NUM_WORDS, EMBEDDING_DIM)

    first = train.getWordsModel(model_type, first_embedding_layer, numLabels, MAX_SEQUENCE_LENGTH, incomplete=True)



    second = Sequential()

    second_embedding_layer = train.getEmbeddingLayer(args, "smh", corpusA, MAX_NUM_WORDS, EMBEDDING_DIM)

    second.add(second_embedding_layer)



    merged = concatenate([first, second])


    model = Sequential()
    model.add(merged)


    model.add(Dense(numLabels, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # model.build()    
    # print(model.summary())

    return model


