# coding: utf-8

__author__      = "Ciprian-Octavian Truică"
__copyright__   = "Copyright 2021, University Politehnica of Bucharest"
__license__     = "GNU GPL"
__version__     = "0.1"
__email__       = "ciprian.truica@upb.ro"
__status__      = "Development"

import random
import sys
import math
import time
import os
import random as rnd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import io as sio
# helpers


# classification
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Embedding, GRU, LSTM, Bidirectional, Input, Concatenate, Conv1D, Flatten, MaxPooling1D, Reshape
from tensorflow.keras.initializers import Constant
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"


num_classes = 2
vocabulary_size = 5556
max_size = 76
batch_size = 1000
epochs_n = 100
units = 128
filters = int(units/2)
no_attributes_gru = units # no units
kernel_size_gru = int(no_attributes_gru/2)
no_attributes_bigru = int(units * 2) # no units
kernel_size_bigru = int(no_attributes_bigru/2)



execution = {}
accuracies = {}
precisions = {}
recalls = {}


def evaluate(y_test, y_pred, modelName='GRU', wordemb='w2v_sg', iters=0):
    y_pred_norm = []

    for elem in y_pred:
        line = [ 0 ] * len(elem)
        try:
            # if an error appears here
            # get a random class
            elem[np.isnan(elem)] = 0
            line[elem.tolist().index(max(elem.tolist()))] = 1
        except:
            print("Error for getting predicted class")
            print(elem.tolist())
            line[rnd.randint(0, len(elem)-1)] = 1
        y_pred_norm.append(line)

    y_p = np.argmax(np.array(y_pred_norm), 1)
    y_t = np.argmax(np.array(y_test), 1)
    accuracy = accuracy_score(y_t, y_p)
    accuracies[wordemb][modelName].append(accuracy)
    precision = precision_score(y_t, y_p, average='weighted')
    precisions[wordemb][modelName].append(precision)
    recall = recall_score(y_t, y_p, average='weighted')
    recalls[wordemb][modelName].append(recall)

    print(modelName, wordemb, "Accuracy", accuracy)
    print(modelName, wordemb, "Precision", precision)
    print(modelName, wordemb, "Recall", recall)
    print("accuracies['", wordemb, "']['", modelName, "'].append(", accuracy, ")")
    print("precisions['", wordemb, "']['", modelName, "'].append(", precision, ")")
    print("recalls['", wordemb, "']['", modelName, "'].append(", recall, ")")
    return y_p, y_t

def modelContentNetworkGRU_00CNN(X_train_docs, X_val_docs, X_test_docs, X_train_net, X_val_net, X_test_net, y_train, y_val, y_test, w2v, num_classes, wordemb, idx):
    # Input Docs
    input_docs = Input(shape=(X_train_docs.shape[1]), name='DOCS_INPUT')
    model_docs = Embedding(input_dim=vocabulary_size, output_dim=units, weights=[w2v], input_length=max_size, name="DOCS_EMBEDDING")(input_docs)
    model_docs = GRU(units = units, dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True, name = 'DOCS_GRU_1')(model_docs)
    model_docs = Flatten(name='DOCS_FLATTEN_1')(model_docs)

    # Input Network
    input_net = Input(shape=(X_train_net.shape[1], X_train_net.shape[2]), name='NETS_INPUT')
    model_net = GRU(units = units, dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True, name = 'NETS_GRU_1')(input_net)
    model_net = Flatten(name='NETS_FLATTEN_1')(model_net)

    combined = Concatenate(name='MODEL_CONCAT')([model_docs, model_net])

    output = Dense(units=num_classes, activation = 'softmax', name = 'MODEL_OUTPUT')(combined) #sigmoid #softmax

    model = Model(inputs=[input_docs, input_net], outputs=output, name="GRU-00CNN-ContentNets")
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    print(model.summary())

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

    start_time = time.time()
    history = model.fit(x = [X_train_docs, X_train_net], y = y_train, epochs=epochs_n, verbose=True, validation_data=([X_val_docs, X_val_net], y_val), batch_size=batch_size, callbacks=[es])
    end_time = time.time()

    y_pred = model.predict([ X_test_docs, X_test_net ], verbose=False)
    y_p, y_t = evaluate(y_test, y_pred, modelName=model.name, wordemb=wordemb, iters=idx)
    exc_time = (end_time - start_time)
    execution[wordemb]["GRU-00CNN-ContentNets"].append(exc_time)
    print("Time taken to train: ", exc_time)
    print("execution['", wordemb, "']['",model.name, "'].append(", exc_time, ")")

def modelContentNetworkGRU_01CNN(X_train_docs, X_val_docs, X_test_docs, X_train_net, X_val_net, X_test_net, y_train, y_val, y_test, w2v, num_classes, wordemb, idx):
    # Input Docs
    input_docs = Input(shape=(X_train_docs.shape[1]), name='DOCS_INPUT')
    model_docs = Embedding(input_dim=vocabulary_size, output_dim=units, weights=[w2v], input_length=max_size, name="DOCS_EMBEDDING")(input_docs)
    model_docs = GRU(units = units, dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True, name = 'DOCS_GRU_1')(model_docs)
    model_docs = Flatten(name='DOCS_FLATTEN_1')(model_docs)

    # Input Network
    input_net = Input(shape=(X_train_net.shape[1], X_train_net.shape[2]), name='NETS_INPUT')
    model_net = GRU(units = units, dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True, name = 'NETS_GRU_1')(input_net)
    model_net = Reshape((no_attributes_gru, 1), name = 'NETS_RESHAPE_1')(model_net) # reshape to number of units
    model_net = Conv1D(filters = filters, kernel_size=kernel_size_gru, activation='relu', name = 'NETS_CNN_1')(model_net)
    model_net = MaxPooling1D(name='NETS_MAXPOLLING_1')(model_net)
    model_net = Flatten(name='NETS_FLATTEN_1')(model_net)

    combined = Concatenate(name='MODEL_CONCAT')([model_docs, model_net])

    output = Dense(units=num_classes, activation = 'softmax', name = 'MODEL_OUTPUT')(combined) #sigmoid #softmax

    model = Model(inputs=[input_docs, input_net], outputs=output, name="GRU-01CNN-ContentNets")
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    print(model.summary())


    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

    start_time = time.time()
    history = model.fit(x = [X_train_docs, X_train_net], y = y_train, epochs=epochs_n, verbose=True, validation_data=([X_val_docs, X_val_net], y_val), batch_size=batch_size, callbacks=[es])
    end_time = time.time()

    y_pred = model.predict([ X_test_docs, X_test_net ], verbose=False)
    y_p, y_t = evaluate(y_test, y_pred, modelName=model.name, wordemb=wordemb, iters=idx)
    exc_time = (end_time - start_time)
    execution[wordemb]["GRU-01CNN-ContentNets"].append(exc_time)
    print("Time taken to train: ", exc_time)
    print("execution['", wordemb, "']['",model.name, "'].append(", exc_time, ")")

def modelContentNetworkGRU_10CNN(X_train_docs, X_val_docs, X_test_docs, X_train_net, X_val_net, X_test_net, y_train, y_val, y_test, w2v, num_classes, wordemb, idx):
    # Input Docs
    input_docs = Input(shape=(X_train_docs.shape[1]), name='DOCS_INPUT')
    model_docs = Embedding(input_dim=vocabulary_size, output_dim=units, weights=[w2v], input_length=max_size, name="DOCS_EMBEDDING")(input_docs)
    model_docs = GRU(units = units, dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True, name = 'DOCS_GRU_1')(model_docs)
    model_docs = Conv1D(filters = int(filters/2), kernel_size=int(kernel_size_gru/2), activation='relu', name = 'DOCS_CNN_1')(model_docs)
    model_docs = MaxPooling1D(name='DOCS_MAXPOLLING_1')(model_docs)
    model_docs = Flatten(name='DOCS_FLATTEN_1')(model_docs)

    # Input Network
    input_net = Input(shape=(X_train_net.shape[1], X_train_net.shape[2]), name='NETS_INPUT')
    model_net = GRU(units = units, dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True, name = 'NETS_GRU_1')(input_net)
    model_net = Flatten(name='NETS_FLATTEN_1')(model_net)

    combined = Concatenate(name='MODEL_CONCAT')([model_docs, model_net])

    output = Dense(units=num_classes, activation = 'softmax', name = 'MODEL_OUTPUT')(combined) #sigmoid #softmax

    model = Model(inputs=[input_docs, input_net], outputs=output, name="GRU-10CNN-ContentNets")
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    print(model.summary())


    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

    start_time = time.time()
    history = model.fit(x = [X_train_docs, X_train_net], y = y_train, epochs=epochs_n, verbose=True, validation_data=([X_val_docs, X_val_net], y_val), batch_size=batch_size, callbacks=[es])
    end_time = time.time()

    y_pred = model.predict([ X_test_docs, X_test_net ], verbose=False)
    y_p, y_t = evaluate(y_test, y_pred, modelName=model.name, wordemb=wordemb, iters=idx)
    exc_time = (end_time - start_time)
    execution[wordemb]["GRU-10CNN-ContentNets"].append(exc_time)
    print("Time taken to train: ", exc_time)
    print("execution['", wordemb, "']['",model.name, "'].append(", exc_time, ")")

def modelContentNetworkGRU_11CNN(X_train_docs, X_val_docs, X_test_docs, X_train_net, X_val_net, X_test_net, y_train, y_val, y_test, w2v, num_classes, wordemb, idx):
    # Input Docs
    input_docs = Input(shape=(X_train_docs.shape[1]), name='DOCS_INPUT')
    model_docs = Embedding(input_dim=vocabulary_size, output_dim=units, weights=[w2v], input_length=max_size, name="DOCS_EMBEDDING")(input_docs)
    model_docs = GRU(units = units, dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True, name = 'DOCS_GRU_1')(model_docs)
    model_docs = Conv1D(filters = int(filters/2), kernel_size=int(kernel_size_gru/2), activation='relu', name = 'DOCS_CNN_1')(model_docs)
    model_docs = MaxPooling1D(name='DOCS_MAXPOLLING_1')(model_docs)
    model_docs = Flatten(name='DOCS_FLATTEN_1')(model_docs)

    # Input Network
    input_net = Input(shape=(X_train_net.shape[1], X_train_net.shape[2]), name='NETS_INPUT')
    model_net = GRU(units = units, dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True, name = 'NETS_GRU_1')(input_net)
    model_net = Reshape((no_attributes_gru, 1), name = 'NETS_RESHAPE_1')(model_net) # reshape to number of units
    model_net = Conv1D(filters = filters, kernel_size=kernel_size_gru, activation='relu', name = 'NETS_CNN_1')(model_net)
    model_net = MaxPooling1D(name='NETS_MAXPOLLING_1')(model_net)
    model_net = Flatten(name='NETS_FLATTEN_1')(model_net)

    combined = Concatenate(name='MODEL_CONCAT')([model_docs, model_net])

    output = Dense(units=num_classes, activation = 'softmax', name = 'MODEL_OUTPUT')(combined) #sigmoid #softmax

    model = Model(inputs=[input_docs, input_net], outputs=output, name="GRU-11CNN-ContentNets")
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    print(model.summary())


    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

    start_time = time.time()
    history = model.fit(x = [X_train_docs, X_train_net], y = y_train, epochs=epochs_n, verbose=True, validation_data=([X_val_docs, X_val_net], y_val), batch_size=batch_size, callbacks=[es])
    end_time = time.time()

    y_pred = model.predict([ X_test_docs, X_test_net ], verbose=False)
    y_p, y_t = evaluate(y_test, y_pred, modelName=model.name, wordemb=wordemb, iters=idx)
    exc_time = (end_time - start_time)
    execution[wordemb]["GRU-11CNN-ContentNets"].append(exc_time)
    print("Time taken to train: ", exc_time)
    print("execution['", wordemb, "']['",model.name, "'].append(", exc_time, ")")

def modelContentGRU(X_train_docs, X_val_docs, X_test_docs, y_train, y_val, y_test, w2v, num_classes, wordemb, idx):
    # Input Docs
    input_docs = Input(shape=(X_train_docs.shape[1]), name='DOCS_INPUT')
    model_docs = Embedding(input_dim=vocabulary_size, output_dim=units, weights=[w2v], name="DOCS_EMBEDDING")(input_docs)
    model_docs = GRU(units = units, dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True, name = 'DOCS_GRU_1')(model_docs)
    model_docs = Flatten(name='DOCS_FLATTEN_1')(model_docs)

    output = Dense(units=num_classes, activation = 'softmax', name = 'MODEL_OUTPUT')(model_docs) #sigmoid #softmax

    model = Model(inputs=input_docs, outputs=output, name="GRU-Content")
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    print(model.summary())


    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

    start_time = time.time()
    history = model.fit(x=X_train_docs, y=y_train, epochs=epochs_n, verbose=True, validation_data=(X_val_docs, y_val), batch_size=batch_size, callbacks=[es])
    end_time = time.time()

    y_pred = model.predict(X_test_docs, verbose=False)
    y_p, y_t = evaluate(y_test, y_pred, modelName=model.name, wordemb=wordemb, iters=idx)
    exc_time = (end_time - start_time)
    execution[wordemb]["GRU-Content"].append(exc_time)
    print("Time taken to train: ", exc_time)
    print("execution['", wordemb, "']['",model.name, "'].append(", exc_time, ")")

def modelContentGRUCNN(X_train_docs, X_val_docs, X_test_docs, y_train, y_val, y_test, w2v, num_classes, wordemb, idx):
    # Input Docs
    input_docs = Input(shape=(X_train_docs.shape[1]), name='DOCS_INPUT')
    model_docs = Embedding(input_dim=vocabulary_size, output_dim=units, weights=[w2v], name="DOCS_EMBEDDING")(input_docs)
    model_docs = GRU(units = units, dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True, name='DOCS_GRU_1')(model_docs)
    model_docs = Conv1D(filters=int(filters/2), kernel_size=int(kernel_size_gru/2), activation='relu', name='DOCS_CNN_1')(model_docs)
    model_docs = MaxPooling1D(name='DOCS_MAXPOLLING_1')(model_docs)
    model_docs = Flatten(name='DOCS_FLATTEN_1')(model_docs)

    output = Dense(units=num_classes, activation = 'softmax', name = 'MODEL_OUTPUT')(model_docs) #sigmoid #softmax

    model = Model(inputs=input_docs, outputs=output, name="GRU-CNN-Content")
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    print(model.summary())


    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

    start_time = time.time()
    history = model.fit(x=X_train_docs, y=y_train, epochs=epochs_n, verbose=True, validation_data=(X_val_docs, y_val), batch_size=batch_size, callbacks=[es])
    end_time = time.time()

    y_pred = model.predict(X_test_docs, verbose=False)
    y_p, y_t = evaluate(y_test, y_pred, modelName=model.name, wordemb=wordemb, iters=idx)
    exc_time = (end_time - start_time)
    execution[wordemb]["GRU-CNN-Content"].append(exc_time)
    print("Time taken to train: ", exc_time)
    print("execution['", wordemb, "']['",model.name, "'].append(", exc_time, ")")

def modelContentNetworkBiGRU_00CNN(X_train_docs, X_val_docs, X_test_docs, X_train_net, X_val_net, X_test_net, y_train, y_val, y_test, w2v, num_classes, wordemb, idx):
    # Input Docs
    input_docs = Input(shape=(X_train_docs.shape[1]), name='DOCS_INPUT')
    model_docs = Embedding(input_dim=vocabulary_size, output_dim=units, weights=[w2v], input_length=max_size, name="DOCS_EMBEDDING")(input_docs)
    model_docs = Bidirectional(GRU(units = units, dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True), name = 'DOCS_BIGRU_1')(model_docs)
    model_docs = Flatten(name='DOCS_FLATTEN_1')(model_docs)

    # Input Network
    input_net = Input(shape=(X_train_net.shape[1], X_train_net.shape[2]), name='NETS_INPUT')
    model_net = Bidirectional(GRU(units = units, dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True), name = 'NETS_BIGRU_1')(input_net)
    model_net = Flatten(name='NETS_FLATTEN_1')(model_net)

    combined = Concatenate(name='MODEL_CONCAT')([model_docs, model_net])

    output = Dense(units=num_classes, activation = 'softmax', name = 'MODEL_OUTPUT')(combined) #sigmoid #softmax

    model = Model(inputs=[input_docs, input_net], outputs=output, name="BiGRU-00CNN-ContentNets")
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    print(model.summary())


    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

    start_time = time.time()
    history = model.fit(x = [X_train_docs, X_train_net], y = y_train, epochs=epochs_n, verbose=True, validation_data=([X_val_docs, X_val_net], y_val), batch_size=batch_size, callbacks=[es])
    end_time = time.time()

    y_pred = model.predict([ X_test_docs, X_test_net ], verbose=False)
    y_p, y_t = evaluate(y_test, y_pred, modelName=model.name, wordemb=wordemb, iters=idx)
    exc_time = (end_time - start_time)
    execution[wordemb]["BiGRU-00CNN-ContentNets"].append(exc_time)
    print("Time taken to train: ", exc_time)
    print("execution['", wordemb, "']['",model.name, "'].append(", exc_time, ")")

def modelContentNetworkBiGRU_01CNN(X_train_docs, X_val_docs, X_test_docs, X_train_net, X_val_net, X_test_net, y_train, y_val, y_test, w2v, num_classes, wordemb, idx):
    # Input Docs
    input_docs = Input(shape=(X_train_docs.shape[1]), name='DOCS_INPUT')
    model_docs = Embedding(input_dim=vocabulary_size, output_dim=units, weights=[w2v], input_length=max_size, name="DOCS_EMBEDDING")(input_docs)
    model_docs = Bidirectional(GRU(units = units, dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True), name = 'DOCS_BIGRU_1')(model_docs)
    model_docs = Flatten(name='DOCS_FLATTEN_1')(model_docs)

    # Input Network
    input_net = Input(shape=(X_train_net.shape[1], X_train_net.shape[2]), name='NETS_INPUT')
    model_net = Bidirectional(GRU(units = units, dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True), name = 'NETS_BIGRU_1')(input_net)
    model_net = Reshape((no_attributes_bigru, 1), name = 'NETS_RESHAPE_1')(model_net) # reshape to number of units
    model_net = Conv1D(filters = filters, kernel_size=kernel_size_bigru, activation='relu', name = 'NETS_CNN_1')(model_net)
    model_net = MaxPooling1D(name='NETS_MAXPOLLING_1')(model_net)
    model_net = Flatten(name='NETS_FLATTEN_1')(model_net)

    combined = Concatenate(name='MODEL_CONCAT')([model_docs, model_net])

    output = Dense(units=num_classes, activation = 'softmax', name = 'MODEL_OUTPUT')(combined) #sigmoid #softmax

    model = Model(inputs=[input_docs, input_net], outputs=output, name="BiGRU-01CNN-ContentNets")
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    print(model.summary())


    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

    start_time = time.time()
    history = model.fit(x = [X_train_docs, X_train_net], y = y_train, epochs=epochs_n, verbose=True, validation_data=([X_val_docs, X_val_net], y_val), batch_size=batch_size, callbacks=[es])
    end_time = time.time()

    y_pred = model.predict([ X_test_docs, X_test_net ], verbose=False)
    y_p, y_t = evaluate(y_test, y_pred, modelName=model.name, wordemb=wordemb, iters=idx)
    exc_time = (end_time - start_time)
    execution[wordemb]["BiGRU-01CNN-ContentNets"].append(exc_time)
    print("Time taken to train: ", exc_time)
    print("execution['", wordemb, "']['",model.name, "'].append(", exc_time, ")")

def modelContentNetworkBiGRU_10CNN(X_train_docs, X_val_docs, X_test_docs, X_train_net, X_val_net, X_test_net, y_train, y_val, y_test, w2v, num_classes, wordemb, idx):
    # Input Docs
    input_docs = Input(shape=(X_train_docs.shape[1]), name='DOCS_INPUT')
    model_docs = Embedding(input_dim=vocabulary_size, output_dim=units, weights=[w2v], input_length=max_size, name="DOCS_EMBEDDING")(input_docs)
    model_docs = Bidirectional(GRU(units = units, dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True), name = 'DOCS_BIGRU_1')(model_docs)
    model_docs = Conv1D(filters = int(filters/2), kernel_size=int(kernel_size_bigru/2), activation='relu', name = 'DOCS_CNN_1')(model_docs)
    model_docs = MaxPooling1D(name='DOCS_MAXPOLLING_1')(model_docs)
    model_docs = Flatten(name='DOCS_FLATTEN_1')(model_docs)

    # Input Network
    input_net = Input(shape=(X_train_net.shape[1], X_train_net.shape[2]), name='NETS_INPUT')
    model_net = Bidirectional(GRU(units = units, dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True), name = 'NETS_BIGRU_1')(input_net)
    model_net = Flatten(name='NETS_FLATTEN_1')(model_net)

    combined = Concatenate(name='MODEL_CONCAT')([model_docs, model_net])

    output = Dense(units=num_classes, activation = 'softmax', name = 'MODEL_OUTPUT')(combined) #sigmoid #softmax

    model = Model(inputs=[input_docs, input_net], outputs=output, name="BiGRU-10CNN-ContentNets")
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    print(model.summary())


    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

    start_time = time.time()
    history = model.fit(x = [X_train_docs, X_train_net], y = y_train, epochs=epochs_n, verbose=True, validation_data=([X_val_docs, X_val_net], y_val), batch_size=batch_size, callbacks=[es])
    end_time = time.time()

    y_pred = model.predict([ X_test_docs, X_test_net ], verbose=False)
    y_p, y_t = evaluate(y_test, y_pred, modelName=model.name, wordemb=wordemb, iters=idx)
    exc_time = (end_time - start_time)
    execution[wordemb]["BiGRU-10CNN-ContentNets"].append(exc_time)
    print("Time taken to train: ", exc_time)
    print("execution['", wordemb, "']['",model.name, "'].append(", exc_time, ")")

def modelContentNetworkBiGRU_11CNN(X_train_docs, X_val_docs, X_test_docs, X_train_net, X_val_net, X_test_net, y_train, y_val, y_test, w2v, num_classes, wordemb, idx):
    # Input Docs
    input_docs = Input(shape=(X_train_docs.shape[1]), name='DOCS_INPUT')
    model_docs = Embedding(input_dim=vocabulary_size, output_dim=units, weights=[w2v], input_length=max_size, name="DOCS_EMBEDDING")(input_docs)
    model_docs = Bidirectional(GRU(units = units, dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True), name = 'DOCS_BIGRU_1')(model_docs)
    model_docs = Conv1D(filters = int(filters/2), kernel_size=int(kernel_size_bigru/2), activation='relu', name = 'DOCS_CNN_1')(model_docs)
    model_docs = MaxPooling1D(name='DOCS_MAXPOLLING_1')(model_docs)
    model_docs = Flatten(name='DOCS_FLATTEN_1')(model_docs)

    # Input Network
    input_net = Input(shape=(X_train_net.shape[1], X_train_net.shape[2]), name='NETS_INPUT')
    model_net = Bidirectional(GRU(units = units, dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True), name = 'NETS_BIGRU_1')(input_net)
    model_net = Reshape((no_attributes_bigru, 1), name = 'NETS_RESHAPE_1')(model_net) # reshape to number of units
    model_net = Conv1D(filters = filters, kernel_size=kernel_size_bigru, activation='relu', name = 'NETS_CNN_1')(model_net)
    model_net = MaxPooling1D(name='NETS_MAXPOLLING_1')(model_net)
    model_net = Flatten(name='NETS_FLATTEN_1')(model_net)

    combined = Concatenate(name='MODEL_CONCAT')([model_docs, model_net])

    output = Dense(units=num_classes, activation = 'softmax', name = 'MODEL_OUTPUT')(combined) #sigmoid #softmax

    model = Model(inputs=[input_docs, input_net], outputs=output, name="BiGRU-11CNN-ContentNets")
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    print(model.summary())


    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

    start_time = time.time()
    history = model.fit(x = [X_train_docs, X_train_net], y = y_train, epochs=epochs_n, verbose=True, validation_data=([X_val_docs, X_val_net], y_val), batch_size=batch_size, callbacks=[es])
    end_time = time.time()

    y_pred = model.predict([ X_test_docs, X_test_net ], verbose=False)
    y_p, y_t = evaluate(y_test, y_pred, modelName=model.name, wordemb=wordemb, iters=idx)
    exc_time = (end_time - start_time)
    execution[wordemb]["BiGRU-11CNN-ContentNets"].append(exc_time)
    print("Time taken to train: ", exc_time)
    print("execution['", wordemb, "']['",model.name, "'].append(", exc_time, ")")

def modelContentBiGRU(X_train_docs, X_val_docs, X_test_docs, y_train, y_val, y_test, w2v, num_classes, wordemb, idx):
    # Input Docs
    input_docs = Input(shape=(X_train_docs.shape[1]), name='DOCS_INPUT')
    model_docs = Embedding(input_dim=vocabulary_size, output_dim=units, weights=[w2v], name="DOCS_EMBEDDING")(input_docs)
    model_docs = Bidirectional(GRU(units = units, dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True), name = 'DOCS_BIGRU_1')(model_docs)
    model_docs = Flatten(name='DOCS_FLATTEN_1')(model_docs)

    output = Dense(units=num_classes, activation = 'softmax', name = 'MODEL_OUTPUT')(model_docs) #sigmoid #softmax

    model = Model(inputs=input_docs, outputs=output, name="BiGRU-Content")
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    print(model.summary())


    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

    start_time = time.time()
    history = model.fit(x=X_train_docs, y=y_train, epochs=epochs_n, verbose=True, validation_data=(X_val_docs, y_val), batch_size=batch_size, callbacks=[es])
    end_time = time.time()

    y_pred = model.predict(X_test_docs, verbose=False)
    y_p, y_t = evaluate(y_test, y_pred, modelName=model.name, wordemb=wordemb, iters=idx)
    exc_time = (end_time - start_time)
    execution[wordemb]["BiGRU-Content"].append(exc_time)
    print("Time taken to train: ", exc_time)
    print("execution['", wordemb, "']['",model.name, "'].append(", exc_time, ")")

def modelContentBiGRUCNN(X_train_docs, X_val_docs, X_test_docs, y_train, y_val, y_test, w2v, num_classes, wordemb, idx):
    # Input Docs
    input_docs = Input(shape=(X_train_docs.shape[1]), name='DOCS_INPUT')
    model_docs = Embedding(input_dim=vocabulary_size, output_dim=units, weights=[w2v], name="DOCS_EMBEDDING")(input_docs)
    model_docs = Bidirectional(GRU(units = units, dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True), name='DOCS_BIGRU_1')(model_docs)
    model_docs = Conv1D(filters=int(filters/2), kernel_size=int(kernel_size_bigru/2), activation='relu', name='DOCS_CNN_1')(model_docs)
    model_docs = MaxPooling1D(name='DOCS_MAXPOLLING_1')(model_docs)
    model_docs = Flatten(name='DOCS_FLATTEN_1')(model_docs)

    output = Dense(units=num_classes, activation = 'softmax', name = 'MODEL_OUTPUT')(model_docs) #sigmoid #softmax

    model = Model(inputs=input_docs, outputs=output, name="BiGRU-CNN-Content")
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    print(model.summary())


    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

    start_time = time.time()
    history = model.fit(x=X_train_docs, y=y_train, epochs=epochs_n, verbose=True, validation_data=(X_val_docs, y_val), batch_size=batch_size, callbacks=[es])
    end_time = time.time()

    y_pred = model.predict(X_test_docs, verbose=False)
    y_p, y_t = evaluate(y_test, y_pred, modelName=model.name, wordemb=wordemb, iters=idx)
    exc_time = (end_time - start_time)
    execution[wordemb]["BiGRU-CNN-Content"].append(exc_time)
    print("Time taken to train: ", exc_time)
    print("execution['", wordemb, "']['",model.name, "'].append(", exc_time, ")")

if __name__ == "__main__":

    ##################### LABELS ############################
    y = sio.loadmat('labels.mat')['y'][0]
    print(y)
    print(len(y))

    ##################### Network ############################
    X_net_std = sio.loadmat('network.mat')['X_net_std']
    print(X_net_std)
    print(len(X_net_std))

    ##################### Content ############################
    X_docs = sio.loadmat('corpus.mat')['X']
    print(X_docs)
    print(len(X_docs))


    for wordemb in ['w2v_cbow', 'w2v_sg', 'ft_cbow', 'ft_sg', 'glove', 'mittens']:
        accuracies[wordemb] = {}
        precisions[wordemb] = {}
        recalls[wordemb] = {}
        execution[wordemb] = {}

        execution[wordemb]["GRU-00CNN-ContentNets"] = []
        accuracies[wordemb]["GRU-00CNN-ContentNets"] = []
        precisions[wordemb]["GRU-00CNN-ContentNets"] = []
        recalls[wordemb]["GRU-00CNN-ContentNets"] = []

        execution[wordemb]["GRU-01CNN-ContentNets"] = []
        accuracies[wordemb]["GRU-01CNN-ContentNets"] = []
        precisions[wordemb]["GRU-01CNN-ContentNets"] = []
        recalls[wordemb]["GRU-01CNN-ContentNets"] = []

        execution[wordemb]["GRU-10CNN-ContentNets"] = []
        accuracies[wordemb]["GRU-10CNN-ContentNets"] = []
        precisions[wordemb]["GRU-10CNN-ContentNets"] = []
        recalls[wordemb]["GRU-10CNN-ContentNets"] = []

        execution[wordemb]["GRU-11CNN-ContentNets"] = []
        accuracies[wordemb]["GRU-11CNN-ContentNets"] = []
        precisions[wordemb]["GRU-11CNN-ContentNets"] = []
        recalls[wordemb]["GRU-11CNN-ContentNets"] = []

        execution[wordemb]["GRU-CNN-Content"] = []
        accuracies[wordemb]["GRU-CNN-Content"] = []
        precisions[wordemb]["GRU-CNN-Content"] = []
        recalls[wordemb]["GRU-CNN-Content"] = []

        execution[wordemb]["GRU-Content"] = []
        accuracies[wordemb]["GRU-Content"] = []
        precisions[wordemb]["GRU-Content"] = []
        recalls[wordemb]["GRU-Content"] = []

        execution[wordemb]["BiGRU-00CNN-ContentNets"] = []
        accuracies[wordemb]["BiGRU-00CNN-ContentNets"] = []
        precisions[wordemb]["BiGRU-00CNN-ContentNets"] = []
        recalls[wordemb]["BiGRU-00CNN-ContentNets"] = []

        execution[wordemb]["BiGRU-01CNN-ContentNets"] = []
        accuracies[wordemb]["BiGRU-01CNN-ContentNets"] = []
        precisions[wordemb]["BiGRU-01CNN-ContentNets"] = []
        recalls[wordemb]["BiGRU-01CNN-ContentNets"] = []

        execution[wordemb]["BiGRU-10CNN-ContentNets"] = []
        accuracies[wordemb]["BiGRU-10CNN-ContentNets"] = []
        precisions[wordemb]["BiGRU-10CNN-ContentNets"] = []
        recalls[wordemb]["BiGRU-10CNN-ContentNets"] = []

        execution[wordemb]["BiGRU-11CNN-ContentNets"] = []
        accuracies[wordemb]["BiGRU-11CNN-ContentNets"] = []
        precisions[wordemb]["BiGRU-11CNN-ContentNets"] = []
        recalls[wordemb]["BiGRU-11CNN-ContentNets"] = []

        execution[wordemb]["BiGRU-CNN-Content"] = []
        accuracies[wordemb]["BiGRU-CNN-Content"] = []
        precisions[wordemb]["BiGRU-CNN-Content"] = []
        recalls[wordemb]["BiGRU-CNN-Content"] = []

        execution[wordemb]["BiGRU-Content"] = []
        accuracies[wordemb]["BiGRU-Content"] = []
        precisions[wordemb]["BiGRU-Content"] = []
        recalls[wordemb]["BiGRU-Content"] = []

        w2v = sio.loadmat(wordemb + '.mat')[wordemb]

        for idx in range(0, 10):

            X_train_docs, X_test_docs, X_train_net, X_test_net, y_train, y_test = train_test_split(X_docs, X_net_std, y, test_size=0.30, shuffle = True, stratify = y)

            X_train_docs, X_val_docs, X_train_net, X_val_net, y_train, y_val = train_test_split(X_train_docs, X_train_net, y_train, test_size=0.20, shuffle = True, stratify = y_train)

            y_train = to_categorical(y_train, num_classes=num_classes)
            y_test = to_categorical(y_test, num_classes=num_classes)
            y_val = to_categorical(y_val, num_classes=num_classes)

            print(X_train_docs.shape)
            print(X_val_docs.shape)
            print(X_test_docs.shape)
            print(X_train_net.shape)
            print(X_val_net.shape)
            print(X_test_net.shape)
            print(y_train.shape)
            print(y_val.shape)
            print(y_test.shape)

            print("===========================================================")
            print("===========================================================")
            print("WORD EMBEDDING:", wordemb)

            print("GRU CONTENT-AWARE RESULTS:", wordemb)
            modelContentGRU(X_train_docs, X_val_docs, X_test_docs, y_train, y_val, y_test, w2v, num_classes, wordemb, idx)
            print("===========================================================")

            print("GRU-CNN CONTENT-AWARE RESULTS:", wordemb)
            modelContentGRUCNN(X_train_docs, X_val_docs, X_test_docs, y_train, y_val, y_test, w2v, num_classes, wordemb, idx)
            print("===========================================================")

            print("GRU-00CNN CONTENT AND NETWORK AWARE RESULTS:", wordemb)
            modelContentNetworkGRU_00CNN(X_train_docs, X_val_docs, X_test_docs, X_train_net, X_val_net, X_test_net, y_train, y_val, y_test, w2v, num_classes, wordemb, idx)
            print("===========================================================")

            print("GRU-01CNN CONTENT AND NETWORK AWARE RESULTS:", wordemb)
            modelContentNetworkGRU_01CNN(X_train_docs, X_val_docs, X_test_docs, X_train_net, X_val_net, X_test_net, y_train, y_val, y_test, w2v, num_classes, wordemb, idx)
            print("===========================================================")

            print("GRU-10CNN CONTENT AND NETWORK AWARE RESULTS:", wordemb)
            modelContentNetworkGRU_10CNN(X_train_docs, X_val_docs, X_test_docs, X_train_net, X_val_net, X_test_net, y_train, y_val, y_test, w2v, num_classes, wordemb, idx)
            print("===========================================================")
            
            print("GRU-11CNN CONTENT AND NETWORK AWARE RESULTS:", wordemb)
            modelContentNetworkGRU_11CNN(X_train_docs, X_val_docs, X_test_docs, X_train_net, X_val_net, X_test_net, y_train, y_val, y_test, w2v, num_classes, wordemb, idx)
            print("===========================================================")

            print("BiGRU CONTENT-AWARE RESULTS:", wordemb)
            modelContentBiGRU(X_train_docs, X_val_docs, X_test_docs, y_train, y_val, y_test, w2v, num_classes, wordemb, idx)
            print("===========================================================")

            print("BiGRU-CNN CONTENT-AWARE RESULTS:", wordemb)
            modelContentBiGRUCNN(X_train_docs, X_val_docs, X_test_docs, y_train, y_val, y_test, w2v, num_classes, wordemb, idx)
            print("===========================================================")

            print("BiGRU-00CNN CONTENT AND NETWORK AWARE RESULTS:", wordemb)
            modelContentNetworkBiGRU_00CNN(X_train_docs, X_val_docs, X_test_docs, X_train_net, X_val_net, X_test_net, y_train, y_val, y_test, w2v, num_classes, wordemb, idx)
            print("===========================================================")

            print("BiGRU-01CNN CONTENT AND NETWORK AWARE RESULTS:", wordemb)
            modelContentNetworkBiGRU_01CNN(X_train_docs, X_val_docs, X_test_docs, X_train_net, X_val_net, X_test_net, y_train, y_val, y_test, w2v, num_classes, wordemb, idx)
            print("===========================================================")

            print("BiGRU-10CNN CONTENT AND NETWORK AWARE RESULTS:", wordemb)
            modelContentNetworkBiGRU_10CNN(X_train_docs, X_val_docs, X_test_docs, X_train_net, X_val_net, X_test_net, y_train, y_val, y_test, w2v, num_classes, wordemb, idx)
            print("===========================================================")
            
            print("BiGRU-11CNN CONTENT AND NETWORK AWARE RESULTS:", wordemb)
            modelContentNetworkBiGRU_11CNN(X_train_docs, X_val_docs, X_test_docs, X_train_net, X_val_net, X_test_net, y_train, y_val, y_test, w2v, num_classes, wordemb, idx)
            print("===========================================================")
            print("===========================================================")

        print("===========================================================")
        print("===========================================================")

        print("GRU-Content", wordemb, "ACCURACY",        round(np.mean(accuracies[wordemb]["GRU-Content"])*100, 2), round(np.std(accuracies[wordemb]["GRU-Content"])*100, 2))
        print("GRU-Content", wordemb, "PRECISION",       round(np.mean(precisions[wordemb]["GRU-Content"])*100, 2), round(np.std(precisions[wordemb]["GRU-Content"])*100, 2))
        print("GRU-Content", wordemb, "RECALL",          round(np.mean(recalls[wordemb]["GRU-Content"])*100, 2), round(np.std(recalls[wordemb]["GRU-Content"])*100, 2))
        print("GRU-Content", wordemb, "EXECUTION TIME",  round(np.mean(execution[wordemb]["GRU-Content"]), 2), round(np.std(execution[wordemb]["GRU-Content"]), 2))

        print("GRU-CNN-Content", wordemb, "ACCURACY",        round(np.mean(accuracies[wordemb]["GRU-CNN-Content"])*100, 2), round(np.std(accuracies[wordemb]["GRU-CNN-Content"])*100, 2))
        print("GRU-CNN-Content", wordemb, "PRECISION",       round(np.mean(precisions[wordemb]["GRU-CNN-Content"])*100, 2), round(np.std(precisions[wordemb]["GRU-CNN-Content"])*100, 2))
        print("GRU-CNN-Content", wordemb, "RECALL",          round(np.mean(recalls[wordemb]["GRU-CNN-Content"])*100, 2), round(np.std(recalls[wordemb]["GRU-CNN-Content"])*100, 2))
        print("GRU-CNN-Content", wordemb, "EXECUTION TIME",  round(np.mean(execution[wordemb]["GRU-CNN-Content"]), 2), round(np.std(execution[wordemb]["GRU-CNN-Content"]), 2))
        
        print("GRU-00CNN-ContentNets ", wordemb, "ACCURACY",        round(np.mean(accuracies[wordemb]["GRU-00CNN-ContentNets"])*100, 2), round(np.std(accuracies[wordemb]["GRU-00CNN-ContentNets"])*100, 2))
        print("GRU-00CNN-ContentNets ", wordemb, "PRECISION", round(np.mean(precisions[wordemb]["GRU-00CNN-ContentNets"])*100, 2), round(np.std(precisions[wordemb]["GRU-00CNN-ContentNets"])*100, 2))
        print("GRU-00CNN-ContentNets ", wordemb, "RECALL",    round(np.mean(recalls[wordemb]["GRU-00CNN-ContentNets"])*100, 2), round(np.std(recalls[wordemb]["GRU-00CNN-ContentNets"])*100, 2))
        print("GRU-00CNN-ContentNets ", wordemb, "EXECUTION TIME",  round(np.mean(execution[wordemb]["GRU-00CNN-ContentNets"]), 2), round(np.std(execution[wordemb]["GRU-00CNN-ContentNets"]), 2))

        print("GRU-01CNN-ContentNets ", wordemb, "ACCURACY",        round(np.mean(accuracies[wordemb]["GRU-01CNN-ContentNets"])*100, 2), round(np.std(accuracies[wordemb]["GRU-01CNN-ContentNets"])*100, 2))
        print("GRU-01CNN-ContentNets ", wordemb, "PRECISION", round(np.mean(precisions[wordemb]["GRU-01CNN-ContentNets"])*100, 2), round(np.std(precisions[wordemb]["GRU-01CNN-ContentNets"])*100, 2))
        print("GRU-01CNN-ContentNets ", wordemb, "RECALL",    round(np.mean(recalls[wordemb]["GRU-01CNN-ContentNets"])*100, 2), round(np.std(recalls[wordemb]["GRU-01CNN-ContentNets"])*100, 2))
        print("GRU-01CNN-ContentNets ", wordemb, "EXECUTION TIME",  round(np.mean(execution[wordemb]["GRU-01CNN-ContentNets"]), 2), round(np.std(execution[wordemb]["GRU-01CNN-ContentNets"]), 2))

        print("GRU-10CNN-ContentNets ", wordemb, "ACCURACY",        round(np.mean(accuracies[wordemb]["GRU-10CNN-ContentNets"])*100, 2), round(np.std(accuracies[wordemb]["GRU-10CNN-ContentNets"])*100, 2))
        print("GRU-10CNN-ContentNets ", wordemb, "PRECISION", round(np.mean(precisions[wordemb]["GRU-10CNN-ContentNets"])*100, 2), round(np.std(precisions[wordemb]["GRU-10CNN-ContentNets"])*100, 2))
        print("GRU-10CNN-ContentNets ", wordemb, "RECALL",    round(np.mean(recalls[wordemb]["GRU-10CNN-ContentNets"])*100, 2), round(np.std(recalls[wordemb]["GRU-10CNN-ContentNets"])*100, 2))
        print("GRU-10CNN-ContentNets ", wordemb, "EXECUTION TIME",  round(np.mean(execution[wordemb]["GRU-10CNN-ContentNets"]), 2), round(np.std(execution[wordemb]["GRU-10CNN-ContentNets"]), 2))

        print("GRU-11CNN-ContentNets ", wordemb, "ACCURACY",        round(np.mean(accuracies[wordemb]["GRU-11CNN-ContentNets"])*100, 2), round(np.std(accuracies[wordemb]["GRU-11CNN-ContentNets"])*100, 2))
        print("GRU-11CNN-ContentNets ", wordemb, "PRECISION", round(np.mean(precisions[wordemb]["GRU-11CNN-ContentNets"])*100, 2), round(np.std(precisions[wordemb]["GRU-11CNN-ContentNets"])*100, 2))
        print("GRU-11CNN-ContentNets ", wordemb, "RECALL",    round(np.mean(recalls[wordemb]["GRU-11CNN-ContentNets"])*100, 2), round(np.std(recalls[wordemb]["GRU-11CNN-ContentNets"])*100, 2))
        print("GRU-11CNN-ContentNets ", wordemb, "EXECUTION TIME",  round(np.mean(execution[wordemb]["GRU-11CNN-ContentNets"]), 2), round(np.std(execution[wordemb]["GRU-11CNN-ContentNets"]), 2))

        print("BiGRU-Content", wordemb, "ACCURACY",        round(np.mean(accuracies[wordemb]["BiGRU-Content"])*100, 2), round(np.std(accuracies[wordemb]["BiGRU-Content"])*100, 2))
        print("BiGRU-Content", wordemb, "PRECISION",       round(np.mean(precisions[wordemb]["BiGRU-Content"])*100, 2), round(np.std(precisions[wordemb]["BiGRU-Content"])*100, 2))
        print("BiGRU-Content", wordemb, "RECALL",          round(np.mean(recalls[wordemb]["BiGRU-Content"])*100, 2), round(np.std(recalls[wordemb]["BiGRU-Content"])*100, 2))
        print("BiGRU-Content", wordemb, "EXECUTION TIME",  round(np.mean(execution[wordemb]["BiGRU-Content"]), 2), round(np.std(execution[wordemb]["BiGRU-Content"]), 2))

        print("BiGRU-CNN-Content", wordemb, "ACCURACY",        round(np.mean(accuracies[wordemb]["BiGRU-CNN-Content"])*100, 2), round(np.std(accuracies[wordemb]["BiGRU-CNN-Content"])*100, 2))
        print("BiGRU-CNN-Content", wordemb, "PRECISION",       round(np.mean(precisions[wordemb]["BiGRU-CNN-Content"])*100, 2), round(np.std(precisions[wordemb]["BiGRU-CNN-Content"])*100, 2))
        print("BiGRU-CNN-Content", wordemb, "RECALL",          round(np.mean(recalls[wordemb]["BiGRU-CNN-Content"])*100, 2), round(np.std(recalls[wordemb]["BiGRU-CNN-Content"])*100, 2))
        print("BiGRU-CNN-Content", wordemb, "EXECUTION TIME",  round(np.mean(execution[wordemb]["BiGRU-CNN-Content"]), 2), round(np.std(execution[wordemb]["BiGRU-CNN-Content"]), 2))
        
        print("BiGRU-00CNN-ContentNets ", wordemb, "ACCURACY",        round(np.mean(accuracies[wordemb]["BiGRU-00CNN-ContentNets"])*100, 2), round(np.std(accuracies[wordemb]["BiGRU-00CNN-ContentNets"])*100, 2))
        print("BiGRU-00CNN-ContentNets ", wordemb, "PRECISION", round(np.mean(precisions[wordemb]["BiGRU-00CNN-ContentNets"])*100, 2), round(np.std(precisions[wordemb]["BiGRU-00CNN-ContentNets"])*100, 2))
        print("BiGRU-00CNN-ContentNets ", wordemb, "RECALL",    round(np.mean(recalls[wordemb]["BiGRU-00CNN-ContentNets"])*100, 2), round(np.std(recalls[wordemb]["BiGRU-00CNN-ContentNets"])*100, 2))
        print("BiGRU-00CNN-ContentNets ", wordemb, "EXECUTION TIME",  round(np.mean(execution[wordemb]["BiGRU-00CNN-ContentNets"]), 2), round(np.std(execution[wordemb]["BiGRU-00CNN-ContentNets"]), 2))


        print("BiGRU-01CNN-ContentNets ", wordemb, "ACCURACY",        round(np.mean(accuracies[wordemb]["BiGRU-01CNN-ContentNets"])*100, 2), round(np.std(accuracies[wordemb]["BiGRU-01CNN-ContentNets"])*100, 2))
        print("BiGRU-01CNN-ContentNets ", wordemb, "PRECISION", round(np.mean(precisions[wordemb]["BiGRU-01CNN-ContentNets"])*100, 2), round(np.std(precisions[wordemb]["BiGRU-01CNN-ContentNets"])*100, 2))
        print("BiGRU-01CNN-ContentNets ", wordemb, "RECALL",    round(np.mean(recalls[wordemb]["BiGRU-01CNN-ContentNets"])*100, 2), round(np.std(recalls[wordemb]["BiGRU-01CNN-ContentNets"])*100, 2))
        print("BiGRU-01CNN-ContentNets ", wordemb, "EXECUTION TIME",  round(np.mean(execution[wordemb]["BiGRU-01CNN-ContentNets"]), 2), round(np.std(execution[wordemb]["BiGRU-01CNN-ContentNets"]), 2))


        print("BiGRU-10CNN-ContentNets ", wordemb, "ACCURACY",        round(np.mean(accuracies[wordemb]["BiGRU-10CNN-ContentNets"])*100, 2), round(np.std(accuracies[wordemb]["BiGRU-10CNN-ContentNets"])*100, 2))
        print("BiGRU-10CNN-ContentNets ", wordemb, "PRECISION", round(np.mean(precisions[wordemb]["BiGRU-10CNN-ContentNets"])*100, 2), round(np.std(precisions[wordemb]["BiGRU-10CNN-ContentNets"])*100, 2))
        print("BiGRU-10CNN-ContentNets ", wordemb, "RECALL",    round(np.mean(recalls[wordemb]["BiGRU-10CNN-ContentNets"])*100, 2), round(np.std(recalls[wordemb]["BiGRU-10CNN-ContentNets"])*100, 2))
        print("BiGRU-10CNN-ContentNets ", wordemb, "EXECUTION TIME",  round(np.mean(execution[wordemb]["BiGRU-10CNN-ContentNets"]), 2), round(np.std(execution[wordemb]["BiGRU-10CNN-ContentNets"]), 2))

        print("BiGRU-11CNN-ContentNets ", wordemb, "ACCURACY",        round(np.mean(accuracies[wordemb]["BiGRU-11CNN-ContentNets"])*100, 2), round(np.std(accuracies[wordemb]["BiGRU-11CNN-ContentNets"])*100, 2))
        print("BiGRU-11CNN-ContentNets ", wordemb, "PRECISION", round(np.mean(precisions[wordemb]["BiGRU-11CNN-ContentNets"])*100, 2), round(np.std(precisions[wordemb]["BiGRU-11CNN-ContentNets"])*100, 2))
        print("BiGRU-11CNN-ContentNets ", wordemb, "RECALL",    round(np.mean(recalls[wordemb]["BiGRU-11CNN-ContentNets"])*100, 2), round(np.std(recalls[wordemb]["BiGRU-11CNN-ContentNets"])*100, 2))
        print("BiGRU-11CNN-ContentNets ", wordemb, "EXECUTION TIME",  round(np.mean(execution[wordemb]["BiGRU-11CNN-ContentNets"]), 2), round(np.std(execution[wordemb]["BiGRU-11CNN-ContentNets"]), 2))



        print("===========================================================")
        print("===========================================================")

    for wordemb in ['w2v_cbow', 'w2v_sg', 'ft_cbow', 'ft_sg', 'glove', 'mittens']:
        print("GRU-Content", wordemb, \
            round(np.mean(accuracies[wordemb]["GRU-Content"])*100, 2), \
            round(np.std(accuracies[wordemb]["GRU-Content"])*100, 2), \
            round(np.mean(precisions[wordemb]["GRU-Content"])*100, 2), \
            round(np.std(precisions[wordemb]["GRU-Content"])*100, 2), \
            round(np.mean(recalls[wordemb]["GRU-Content"])*100, 2), \
            round(np.std(recalls[wordemb]["GRU-Content"])*100, 2), \
            round(np.mean(execution[wordemb]["GRU-Content"]), 2), \
            round(np.std(execution[wordemb]["GRU-Content"]), 2))

        print("GRU-CNN-Content", wordemb, \
            round(np.mean(accuracies[wordemb]["GRU-CNN-Content"])*100, 2), \
            round(np.std(accuracies[wordemb]["GRU-CNN-Content"])*100, 2), \
            round(np.mean(precisions[wordemb]["GRU-CNN-Content"])*100, 2), \
            round(np.std(precisions[wordemb]["GRU-CNN-Content"])*100, 2), \
            round(np.mean(recalls[wordemb]["GRU-CNN-Content"])*100, 2), \
            round(np.std(recalls[wordemb]["GRU-CNN-Content"])*100, 2), \
            round(np.mean(execution[wordemb]["GRU-CNN-Content"]), 2), \
            round(np.std(execution[wordemb]["GRU-CNN-Content"]), 2))

        print("GRU-00CNN-ContentNets", wordemb, \
            round(np.mean(accuracies[wordemb]["GRU-00CNN-ContentNets"])*100, 2), \
            round(np.std(accuracies[wordemb]["GRU-00CNN-ContentNets"])*100, 2), \
            round(np.mean(precisions[wordemb]["GRU-00CNN-ContentNets"])*100, 2), \
            round(np.std(precisions[wordemb]["GRU-00CNN-ContentNets"])*100, 2), \
            round(np.mean(recalls[wordemb]["GRU-00CNN-ContentNets"])*100, 2), \
            round(np.std(recalls[wordemb]["GRU-00CNN-ContentNets"])*100, 2), \
            round(np.mean(execution[wordemb]["GRU-00CNN-ContentNets"]), 2), \
            round(np.std(execution[wordemb]["GRU-00CNN-ContentNets"]), 2))

        print("GRU-01CNN-ContentNets", wordemb, \
            round(np.mean(accuracies[wordemb]["GRU-01CNN-ContentNets"])*100, 2), \
            round(np.std(accuracies[wordemb]["GRU-01CNN-ContentNets"])*100, 2), \
            round(np.mean(precisions[wordemb]["GRU-01CNN-ContentNets"])*100, 2), \
            round(np.std(precisions[wordemb]["GRU-01CNN-ContentNets"])*100, 2), \
            round(np.mean(recalls[wordemb]["GRU-01CNN-ContentNets"])*100, 2), \
            round(np.std(recalls[wordemb]["GRU-01CNN-ContentNets"])*100, 2), \
            round(np.mean(execution[wordemb]["GRU-01CNN-ContentNets"]), 2), \
            round(np.std(execution[wordemb]["GRU-01CNN-ContentNets"]), 2))

        print("GRU-10CNN-ContentNets", wordemb, \
            round(np.mean(accuracies[wordemb]["GRU-10CNN-ContentNets"])*100, 2), \
            round(np.std(accuracies[wordemb]["GRU-10CNN-ContentNets"])*100, 2), \
            round(np.mean(precisions[wordemb]["GRU-10CNN-ContentNets"])*100, 2), \
            round(np.std(precisions[wordemb]["GRU-10CNN-ContentNets"])*100, 2), \
            round(np.mean(recalls[wordemb]["GRU-10CNN-ContentNets"])*100, 2), \
            round(np.std(recalls[wordemb]["GRU-10CNN-ContentNets"])*100, 2), \
            round(np.mean(execution[wordemb]["GRU-10CNN-ContentNets"]), 2), \
            round(np.std(execution[wordemb]["GRU-10CNN-ContentNets"]), 2))

        print("GRU-11CNN-ContentNets", wordemb, \
            round(np.mean(accuracies[wordemb]["GRU-11CNN-ContentNets"])*100, 2), \
            round(np.std(accuracies[wordemb]["GRU-11CNN-ContentNets"])*100, 2), \
            round(np.mean(precisions[wordemb]["GRU-11CNN-ContentNets"])*100, 2), \
            round(np.std(precisions[wordemb]["GRU-11CNN-ContentNets"])*100, 2), \
            round(np.mean(recalls[wordemb]["GRU-11CNN-ContentNets"])*100, 2), \
            round(np.std(recalls[wordemb]["GRU-11CNN-ContentNets"])*100, 2), \
            round(np.mean(execution[wordemb]["GRU-11CNN-ContentNets"]), 2), \
            round(np.std(execution[wordemb]["GRU-11CNN-ContentNets"]), 2))

        print("BiGRU-Content", wordemb, \
            round(np.mean(accuracies[wordemb]["BiGRU-Content"])*100, 2), \
            round(np.std(accuracies[wordemb]["BiGRU-Content"])*100, 2), \
            round(np.mean(precisions[wordemb]["BiGRU-Content"])*100, 2), \
            round(np.std(precisions[wordemb]["BiGRU-Content"])*100, 2), \
            round(np.mean(recalls[wordemb]["BiGRU-Content"])*100, 2), \
            round(np.std(recalls[wordemb]["BiGRU-Content"])*100, 2), \
            round(np.mean(execution[wordemb]["BiGRU-Content"]), 2), \
            round(np.std(execution[wordemb]["BiGRU-Content"]), 2))

        print("BiGRU-CNN-Content", wordemb, \
            round(np.mean(accuracies[wordemb]["BiGRU-CNN-Content"])*100, 2), \
            round(np.std(accuracies[wordemb]["BiGRU-CNN-Content"])*100, 2), \
            round(np.mean(precisions[wordemb]["BiGRU-CNN-Content"])*100, 2), \
            round(np.std(precisions[wordemb]["BiGRU-CNN-Content"])*100, 2), \
            round(np.mean(recalls[wordemb]["BiGRU-CNN-Content"])*100, 2), \
            round(np.std(recalls[wordemb]["BiGRU-CNN-Content"])*100, 2), \
            round(np.mean(execution[wordemb]["BiGRU-CNN-Content"]), 2), \
            round(np.std(execution[wordemb]["BiGRU-CNN-Content"]), 2))

        print("BiGRU-00CNN-ContentNets", wordemb, \
            round(np.mean(accuracies[wordemb]["BiGRU-00CNN-ContentNets"])*100, 2), \
            round(np.std(accuracies[wordemb]["BiGRU-00CNN-ContentNets"])*100, 2), \
            round(np.mean(precisions[wordemb]["BiGRU-00CNN-ContentNets"])*100, 2), \
            round(np.std(precisions[wordemb]["BiGRU-00CNN-ContentNets"])*100, 2), \
            round(np.mean(recalls[wordemb]["BiGRU-00CNN-ContentNets"])*100, 2), \
            round(np.std(recalls[wordemb]["BiGRU-00CNN-ContentNets"])*100, 2), \
            round(np.mean(execution[wordemb]["BiGRU-00CNN-ContentNets"]), 2), \
            round(np.std(execution[wordemb]["BiGRU-00CNN-ContentNets"]), 2))

        print("BiGRU-01CNN-ContentNets", wordemb, \
            round(np.mean(accuracies[wordemb]["BiGRU-01CNN-ContentNets"])*100, 2), \
            round(np.std(accuracies[wordemb]["BiGRU-01CNN-ContentNets"])*100, 2), \
            round(np.mean(precisions[wordemb]["BiGRU-01CNN-ContentNets"])*100, 2), \
            round(np.std(precisions[wordemb]["BiGRU-01CNN-ContentNets"])*100, 2), \
            round(np.mean(recalls[wordemb]["BiGRU-01CNN-ContentNets"])*100, 2), \
            round(np.std(recalls[wordemb]["BiGRU-01CNN-ContentNets"])*100, 2), \
            round(np.mean(execution[wordemb]["BiGRU-01CNN-ContentNets"]), 2), \
            round(np.std(execution[wordemb]["BiGRU-01CNN-ContentNets"]), 2))

        print("BiGRU-10CNN-ContentNets", wordemb, \
            round(np.mean(accuracies[wordemb]["BiGRU-10CNN-ContentNets"])*100, 2), \
            round(np.std(accuracies[wordemb]["BiGRU-10CNN-ContentNets"])*100, 2), \
            round(np.mean(precisions[wordemb]["BiGRU-10CNN-ContentNets"])*100, 2), \
            round(np.std(precisions[wordemb]["BiGRU-10CNN-ContentNets"])*100, 2), \
            round(np.mean(recalls[wordemb]["BiGRU-10CNN-ContentNets"])*100, 2), \
            round(np.std(recalls[wordemb]["BiGRU-10CNN-ContentNets"])*100, 2), \
            round(np.mean(execution[wordemb]["BiGRU-10CNN-ContentNets"]), 2), \
            round(np.std(execution[wordemb]["BiGRU-10CNN-ContentNets"]), 2))

        print("BiGRU-11CNN-ContentNets", wordemb, \
            round(np.mean(accuracies[wordemb]["BiGRU-11CNN-ContentNets"])*100, 2), \
            round(np.std(accuracies[wordemb]["BiGRU-11CNN-ContentNets"])*100, 2), \
            round(np.mean(precisions[wordemb]["BiGRU-11CNN-ContentNets"])*100, 2), \
            round(np.std(precisions[wordemb]["BiGRU-11CNN-ContentNets"])*100, 2), \
            round(np.mean(recalls[wordemb]["BiGRU-11CNN-ContentNets"])*100, 2), \
            round(np.std(recalls[wordemb]["BiGRU-11CNN-ContentNets"])*100, 2), \
            round(np.mean(execution[wordemb]["BiGRU-11CNN-ContentNets"]), 2), \
            round(np.std(execution[wordemb]["BiGRU-11CNN-ContentNets"]), 2))
