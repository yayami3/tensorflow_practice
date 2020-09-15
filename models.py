import tensorflow as tf
import numpy as np

from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.python.keras.metrics import Mean

class LSTMM(tf.keras.Model):
    def __init__(self):
        super(LSTMM, self).__init__()
        self.lstm = LSTM(20)
        self.dense = Dense(1, activation='sigmoid')

    def call(self, x):
        x = self.lstm(x)
        x = self.dense(x)
        return x


class Transformer(tf.keras.Model):
    def __init__(self, name='encoder', n_hidden=12, n_rnn=10):
        super(Transformer, self).__init__(name=name)
        self.n_hidden = n_hidden
        self.n_rnn = n_rnn
        self.encoder = LSTM(n_hidden, return_state=True, return_sequences=True)
        # call内でDenseインスタンス生成すると怒られる
        self.dense1 = Dense(1, activation="sigmoid")
        self.dense2 = Dense(1, activation='sigmoid')

    def call(self, x):
        query, h, c = self.encoder(x)
        #print(f'query.shape={query.shape}')
        #hidden_state = [h, c]
        inner_product = tf.matmul(query, query, transpose_b=True)
        #print(inner_product.shape)

        attention_weights = tf.nn.softmax(inner_product)
        #tf.print(attention_weights)
        #print(f'attention_weights.shape={attention_weights.shape}')
        #v = tf.matmul(query, attention_weights,  transpose_b=True)
        v = tf.matmul(attention_weights, query) + query

        #print(f'v.shape={v.shape}')
        v = self.dense1(v)
        cls_vec = v[:, 0:, :]

        logit = self.dense2(v)
        prob = tf.nn.softmax(logit)
        return prob