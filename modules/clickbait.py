# -*- coding: utf-8 -*-
"""
"""
import tensorflow as tf
tf.to_float = lambda x: tf.cast(x, tf.float32)

from nltk import word_tokenize
from gensim.models import KeyedVectors
import re
import numpy as np
import os
path = os.path.join(os.path.dirname(__file__), 'models')


#w2v = KeyedVectors.load_word2vec_format(WORD2VEC_VECTORS_BIN, binary=True)
w2v = KeyedVectors.load_word2vec_format(os.path.join(path, 'models/GoogleNews-vectors-negative300.bin.gz'), binary=True, limit=30000)

# ******* START CLICKBAIT MODELING ************
dimsize = 300
sequence_size = 15
maxlen = 250

def binarize(x, sz=37):
    return tf.to_float(tf.one_hot(x, sz, on_value=1, off_value=0, axis=-1))
  
def binarize_outshape(in_shape):
    return in_shape[0], in_shape[1], 37

def load_model():
    filter_length = [5, 3, 3]
    nb_filter = [196, 196, 300]
    pool_length = 2
    
    in_sentence = tf.keras.layers.Input(shape=(maxlen,), dtype='int64')
    
    # binarize function creates a onehot encoding of each character index
    embedded = tf.keras.layers.Lambda(binarize, output_shape=binarize_outshape)(in_sentence)
    # embedded: encodes sentence
    for i in range(len(nb_filter)):
      embedded = tf.keras.layers.Conv1D(filters=nb_filter[i], kernel_size=filter_length[i], padding='valid',
                                        activation='relu', kernel_initializer='glorot_normal')(embedded)
                                        
      embedded = tf.keras.layers.Dropout(0.1)(embedded)
      embedded = tf.keras.layers.MaxPooling1D(pool_size=pool_length)(embedded)
      
    lstm_input = tf.keras.layers.Input(shape=(sequence_size, dimsize))
    combined_features = tf.keras.layers.concatenate([embedded, lstm_input], axis=1)
    forward_sent = tf.keras.layers.LSTM(128, dropout=0.2, recurrent_dropout= 0.2)(combined_features)
    backward_sent = tf.keras.layers.LSTM(128, dropout=0.2, recurrent_dropout= 0.2, go_backwards=True)(combined_features)
    
    sent_encode = tf.keras.layers.concatenate([forward_sent, backward_sent])
    sent_encode = tf.keras.layers.Dropout(0.3)(sent_encode)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(sent_encode)
    
    model = tf.keras.Model(inputs=[in_sentence,lstm_input], outputs=output)
    
    model.load_weights(os.path.join(path, 'models/clickbait_weights.h5'))
    return model
  
def clean(text):
    text = re.sub('[\W_]+', ' ', text)
    text = text.strip()
    return text

def tokenize(text):
    text = re.sub('[\W_]+', ' ', text)
    text = text.strip()
    words = [w for w in word_tokenize(text) if w != 's']
    return words

def process_char(hl):
    all_txt = ''
    all_txt += hl
    chars = set(all_txt)
    char_indices = dict((c, i) for i, c in enumerate(chars))

    return char_indices

def create_char_feature_matrix(hl, char_indices):
    X = np.ones((1, maxlen), dtype=np.int64) * -1
    for t, char in enumerate(hl):
        X[0, t] = char_indices[char]

    return X

def create_word_feature_matrix(Text):
    X = np.zeros((1, sequence_size, dimsize))
    sequence = np.zeros((sequence_size, dimsize))
    tokens = Text
    count = 0
    for token in tokens:
        if count == 15: # Number of words, change for headline/ content
            break
        try:
            token = token
            sequence[count] = w2v[token]
            count += 1
        except:
            pass
    X[0] = sequence
    return X

def create_test_matrix(hl):

    Clean_test = clean(hl)
    Text_test = tokenize(hl)
    
    # create test character features
    char_indices = process_char(Clean_test)
    X_test_char = create_char_feature_matrix(Clean_test, char_indices)
    
    # create test word features
    X_test_word = create_word_feature_matrix(Text_test)

    return X_test_word, X_test_char


def predict_classes(model, X_test_char, X_test_word):

    proba = model.predict([X_test_char, X_test_word])
    if proba.shape[-1] > 1:
        return proba.argmax(axis=-1)
    else:
        return (proba > 0.5).astype('int32')

def click_prediction(hl):
    X_test_word, X_test_char = create_test_matrix(hl)
    model_clickbait = load_model()
    
    y_pred = predict_classes(model_clickbait, X_test_char, X_test_word).item()
    y_score = model_clickbait.predict([X_test_char, X_test_word]).item()
    if(y_pred == 0):
        y_score = 1 - y_score
        non_clickbait = round(y_score, 2) * 100
        clickbait = 100 - non_clickbait 
    else:
        clickbait = round(y_score, 2) * 100
        non_clickbait = 100 - clickbait

    return clickbait, non_clickbait