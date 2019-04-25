

import numpy as np
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM
from tensorflow.keras.models import Model, Sequential

import kerasModel as km
import train
import embeddings as emb


def getDocsModel(corpusA, model_type, numLabels, MAX_SEQUENCE_LENGTH):

	first_embedding_layer = train.getEmbeddingLayer("w2v", corpusA, MAX_NUM_WORDS, EMBEDDING_DIM)

	first = train.getWordsModel(model_type, first_embedding_layer, numLabels, MAX_SEQUENCE_LENGTH, , incomplete=True)



	second = Sequential()

	second_embedding_layer = train.getEmbeddingLayer("smh", corpusA, MAX_NUM_WORDS, EMBEDDING_DIM)

	second.add(embedding_layer)



	merged = Concatenate([first, second])


	model = Sequential()
	model.add(merged)


    model.add(Dense(numLabels, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())
    return model


