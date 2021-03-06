# Some parts taken from:   https://github.com/keras-team/keras/blob/master/examples/pretrained_word_embeddings.py



import numpy as np
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM
from tensorflow.keras.models import Model, Sequential




def getConvModel(embedding_layer, numLabels, MAX_SEQUENCE_LENGTH=1000):

    print "Creating Keras Conv1D Sequential Model"

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
    preds = Dense(numLabels, activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])


    print(model.summary())
    return model



def getConvLSTMmodel(embedding_layer, numLabels, lstmN=128, convFilters=32, MAX_SEQUENCE_LENGTH=1000, incomplete=False):

    print "Compiling Keras LSTM model."

    model = Sequential()
    model.add(embedding_layer)
    model.add(Conv1D(filters=convFilters, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(lstmN))

    if incomplete:
        return model

    model.add(Dense(numLabels, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())
    return model


def getLSTMmodel(embedding_layer, numLabels,  lstmN=128, MAX_SEQUENCE_LENGTH=1000, incomplete=False):
    model = Sequential()
    model.add(embedding_layer)
    # model.add(Embedding(max_features, 128))
    # model.add(LSTM(lstmN, dropout=0.2, recurrent_dropout=0.2))

    model.add(LSTM(lstmN))

    if incomplete:
        return model


    model.add(Dense(numLabels, activation='softmax'))   
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    print(model.summary())
    return model


