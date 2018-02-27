import keras
from keras.layers import Dense, Input, LSTM, Embedding, Bidirectional
from keras.layers import Dropout, BatchNormalization, GlobalMaxPool1D
from keras.layers import Conv1D, GlobalMaxPooling1D, Flatten, MaxPooling1D
from keras.layers import TimeDistributed, Lambda, GRU, GlobalAveragePooling1D
from keras.layers import CuDNNGRU, Convolution1D, Concatenate, SpatialDropout1D
from keras.layers.merge import concatenate, add
from keras.engine.topology import Layer
from keras.activations import relu

from keras.initializers import RandomNormal
from keras import initializers
from keras import backend as K
from keras import constraints
from keras import regularizers
from keras import optimizers as k_opt
import numpy as np


def get_optimizer(lr, optim_name):
    optimizer = None
    if optim_name == 'nadam':
        optimizer = k_opt.Nadam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None,
                                schedule_decay=0.004, clipvalue=1, clipnorm=1)
    elif optim_name == 'sgd':
        optimizer = k_opt.SGD(lr=lr, clipvalue=1, clipnorm=1)
    elif optim_name == 'rms':
        optimizer = k_opt.RMSprop(lr=lr, clipvalue=1, clipnorm=1)
    elif optim_name == 'adam':
        optimizer = k_opt.Adam(lr=lr, clipvalue=1, clipnorm=1)
    return optimizer


class BaseModel:
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.model = None

    def build_model(self):
        return NotImplemented


class BasicLSTM(BaseModel):
    def __init__(self, data, dense_size=50, embed_trainable=False, lr=0.001,
                 optim_name=None, batch_size=64):
        super().__init__(data, batch_size)
        if optim_name is None:
            optim_name = 'nadam'
        self.lr = lr
        self.embed_trainable = embed_trainable
        self.dense_size = dense_size
        self.optim_name = optim_name
        self.build_model()
        self.description = 'LSTM model'

    def build_model(self):
        data = self.data
        inp = Input(shape=(data.seq_length,))
        embed = Embedding(data.max_feature, data.embed_dim,
                          weights=[data.embed_matrix],
                          trainable=self.embed_trainable)(inp)
        lstm = Bidirectional(LSTM(data.embed_dim, return_sequences=True))(embed)
        pool = GlobalMaxPool1D()(lstm)
        dense = Dense(self.dense_size, activation="relu")(pool)
        output = Dense(6, activation="sigmoid")(dense)
        model = keras.models.Model(inputs=inp, outputs=output)
        optimizer = get_optimizer(self.lr, self.optim_name)
        model.compile(loss='binary_crossentropy', optimizer=optimizer,
                      metrics=['accuracy'])
        self.model = model
        model_descirption = f'''BaseLSTM model
        dense_size: {self.dense_size}
        embed_trainbale: {self.embed_trainable}
        lr: {self.lr}
        optim_name: {self.optim_name}
        batch_size: {self.batch_size}'''
        print(model_descirption)
        print(model.summary())


class Lstm(BaseModel):
    def __init__(self, data, dense_size=50, embed_trainable=False, lr=0.001,
                 optim_name=None, batch_size=128, dropout=0.1):
        super().__init__(data, batch_size)
        if optim_name is None:
            optim_name = 'nadam'
        self.lr = lr
        self.embed_trainable = embed_trainable
        self.dense_size = dense_size
        self.optim_name = optim_name
        self.dropout = dropout
        self.build_model()
        self.description = 'LSTM with dropout'

    def build_model(self):
        data = self.data
        inp = Input(shape=(data.seq_length,))
        embed = Embedding(data.max_feature, data.embed_dim,
                          weights=[data.embed_matrix],
                          trainable=self.embed_trainable)(inp)
        lstm = Bidirectional(LSTM(data.embed_dim, return_sequences=True,
                                  dropout=self.dropout,
                                  recurrent_dropout=self.dropout))(embed)
        pool = GlobalMaxPool1D()(lstm)
        bn = BatchNormalization()(pool)
        dense = Dense(self.dense_size, activation="relu")(bn)
        drop = Dropout(self.dropout)(dense)
        output = Dense(6, activation="sigmoid")(drop)
        model = keras.models.Model(inputs=inp, outputs=output)
        optimizer = get_optimizer(self.lr, self.optim_name)
        model.compile(loss='binary_crossentropy', optimizer=optimizer,
                      metrics=['accuracy'])
        self.model = model
        model_descirption = f'''LSTM model
                dense_size: {self.dense_size}
                embed_trainbale: {self.embed_trainable}
                lr: {self.lr}
                optim_name: {self.optim_name}
                batch_size: {self.batch_size}
                dropout: {self.dropout}'''
        print(model_descirption)
        print(model.summary())


class DoubleGRU(BaseModel):
    def __init__(self, data, dense_size=60, embed_trainable=False, lr=0.001,
                 optim_name=None, batch_size=256, dropout=0.5, recur_unit=100):
        super().__init__(data, batch_size)
        if optim_name is None:
            optim_name = 'rms'
        self.lr = lr
        self.embed_trainable = embed_trainable
        self.dense_size = dense_size
        self.optim_name = optim_name
        self.dropout = dropout
        self.recur_unit = recur_unit
        self.build_model()
        self.description = 'Double GRU'

    def build_model(self):
        data = self.data
        input_layer = Input(shape=(data.seq_length,))
        embedding_layer = Embedding(data.max_feature, data.embed_dim,
                                    weights=[data.embed_matrix],
                                    trainable=self.embed_trainable)(input_layer)
        x = SpatialDropout1D(self.dropout)(embedding_layer)
        x = Bidirectional(CuDNNGRU(self.recur_unit, return_sequences=True))(x)
        x = Bidirectional(CuDNNGRU(self.recur_unit, return_sequences=True))(x)
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        conc = concatenate([avg_pool, max_pool])
        x = Dense(self.dense_size, activation="relu")(conc)
        output_layer = Dense(6, activation="sigmoid")(x)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        optimizer = get_optimizer(self.lr, self.optim_name)
        model.compile(loss='binary_crossentropy', optimizer=optimizer,
                      metrics=['accuracy'])
        self.model = model
        model_descirption = f'''Double GRU model
                        dense_size: {self.dense_size}
                        embed_trainbale: {self.embed_trainable}
                        lr: {self.lr}
                        optim_name: {self.optim_name}
                        batch_size: {self.batch_size}
                        dropout: {self.dropout}'''
        print(model_descirption)
        print(model.summary())


class CNNModel(BaseModel):
    def __init__(self, data, batch_size=128, embed_trainable=False,
                 kernel_sizes=None, filter_count=256, lr=0.0015,
                 optim_name=None, dense_size=50, dropout=0):
        super().__init__(data, batch_size)
        if optim_name is None:
            optim_name = 'nadam'
        self.optim_name = optim_name
        self.embed_trainable = embed_trainable
        self.filter_count = filter_count
        self.lr = lr
        self.dense_size = dense_size
        self.kernel_sizes = kernel_sizes or [3, 4, 5]
        self.dropout = dropout
        self.use_embed = hasattr(data, 'embed_matrix')
        self.build_model()
        self.description = 'CNN Model'

    def build_model(self):
        data = self.data
        if self.use_embed:
            inputs = Input(shape=(data.seq_length,), dtype='int32')
            x = Embedding(data.max_feature, data.embed_dim,
                          weights=[data.embed_matrix],
                          trainable=self.embed_trainable)(inputs)
        else:
            inputs = Input(shape=(data.seq_length, data.embed_dim), dtype='float32')
            x = inputs
        # 词窗大小分别为3,4,5
        cnn1 = Convolution1D(self.filter_count, 3, padding='same', strides=1, activation='relu')(x)
        cnn1 = GlobalMaxPooling1D()(cnn1)
        cnn2 = Convolution1D(self.filter_count, 4, padding='same', strides=1, activation='relu')(x)
        cnn2 = GlobalMaxPooling1D()(cnn2)
        cnn3 = Convolution1D(self.filter_count, 5, padding='same', strides=1, activation='relu')(x)
        cnn3 = GlobalMaxPooling1D()(cnn3)
        cnn = concatenate([cnn1,cnn2,cnn3], axis=-1)
        dense1 = Dense(self.dense_size, activation='relu')(cnn)

        output = Dense(units=6, activation='sigmoid')(dense1)
        model = keras.models.Model(inputs=inputs, outputs=output)
        optimizer = get_optimizer(self.lr, self.optim_name)
        model.compile(loss='binary_crossentropy', optimizer=optimizer,
                      metrics=['accuracy'])
        self.model = model

        model_descirption = f'''CNN model
                dense_size: {self.dense_size}
                embed_trainbale: {self.embed_trainable}
                filter_count: {self.filter_count}
                lr: {self.lr}
                dropout: {self.dropout}
                kernel_size: {self.kernel_sizes}
                optim_name: {self.optim_name}
                batch_size: {self.batch_size}'''
        print(model_descirption)
        print(model.summary())


class RCNNModel(BaseModel):
    def __init__(self, data, batch_size=256, embed_trainable=False,
                 lr=0.0005, optim_name=None, dense_size=100, dropout=0.5):
        super().__init__(data, batch_size)
        if optim_name is None:
            optim_name = 'nadam'
        self.optim_name = optim_name
        self.embed_trainable = embed_trainable
        self.lr = lr
        self.dense_size = dense_size
        self.dropout = dropout
        self.build_model()
        self.description = 'RCNN Model'

    def build_model(self):
        data = self.data
        maxlen = data.seq_length
        max_features = data.max_feature
        embed_size = data.embed_dim
        embed_matrix = data.embed_matrix
        drop = self.dropout
        dsize = self.dense_size

        document = Input(shape=(maxlen,), dtype="int32")

        def k_slice(x, start, end):
            return x[:, start:end]

        left_1 = Lambda(k_slice, arguments={'start': maxlen - 1,
                                            'end': maxlen})(document)
        left_2 = Lambda(k_slice, arguments={'start': 0,
                                            'end': maxlen - 1})(document)
        left_context = concatenate([left_1, left_2], axis=1)

        right_1 = Lambda(k_slice, arguments={'start': 1,
                                             'end': maxlen})(document)
        right_2 = Lambda(k_slice, arguments={'start': 0, 'end': 1})(document)
        right_context = concatenate([right_1, right_2], axis=1)

        embedder = Embedding(max_features, embed_size,
                             weights=[embed_matrix], trainable=self.embed_trainable)

        doc_embedding = embedder(document)
        l_embedding = embedder(left_context)
        r_embedding = embedder(right_context)

        forward = CuDNNGRU(embed_size, return_sequences=True)(l_embedding)
        backward = CuDNNGRU(embed_size, return_sequences=True,
                            go_backwards=True)(r_embedding)

        together = concatenate([forward, doc_embedding, backward], axis=2)
        semantic = TimeDistributed(Dense(dsize, activation="tanh"))(together)
        pool_rnn = GlobalMaxPool1D()(semantic)
        output = Dense(6, activation="sigmoid")(pool_rnn)

        model = keras.models.Model(inputs=document, outputs=output)
        optimizer = get_optimizer(self.lr, self.optim_name)
        model.compile(loss='binary_crossentropy', optimizer=optimizer,
                      metrics=['accuracy'])
        self.model = model
        model_descirption = f'''RCNN model
                        dense_size: {self.dense_size}
                        embed_trainbale: {self.embed_trainable}
                        lr: {self.lr}
                        dropout: {self.dropout}
                        optim_name: {self.optim_name}
                        batch_size: {self.batch_size}'''
        print(model_descirption)
        print(model.summary())


class AttLayer(Layer):
    def __init__(self, init='glorot_uniform', kernel_regularizer=None,
                 bias_regularizer=None, kernel_constraint=None,
                 bias_constraint=None, **kwargs):
        self.supports_masking = True
        self.init = initializers.get(init)
        self.kernel_initializer = initializers.get(init)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(kernel_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight((input_shape[-1], 1),
                                 initializer=self.kernel_initializer,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)
        self.b = self.add_weight((input_shape[1],),
                                 initializer='zero',
                                 name='{}_b'.format(self.name),
                                 regularizer=self.bias_regularizer,
                                 constraint=self.bias_constraint)
        self.u = self.add_weight((input_shape[1],),
                                 initializer=self.kernel_initializer,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        uit = K.dot(x, self.W)  # (x, 40, 1)
        uit = K.squeeze(uit, -1)  # (x, 40)
        uit = uit + self.b  # (x, 40) + (40,)
        uit = K.tanh(uit)  # (x, 40)

        ait = uit * self.u  # (x, 40) * (40, 1) => (x, 1)
        ait = K.exp(ait)  # (X, 1)

        if mask is not None:
            mask = K.cast(mask, K.floatx())  # (x, 40)
            ait = mask * ait  # (x, 40) * (x, 40, )

        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(),
                      K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


class AttenModel(BaseModel):
    def __init__(self, data, dense_size=50, embed_trainable=False, lr=0.0015,
                 optim_name=None, batch_size=256, dropout=0.5):
        super().__init__(data, batch_size)
        if optim_name is None:
            optim_name = 'nadam'
        self.lr = lr
        self.embed_trainable = embed_trainable
        self.dense_size = dense_size
        self.optim_name = optim_name
        self.dropout = dropout
        self.build_model()
        self.description = 'Attention Model with dropout'

    def build_model(self):
        data = self.data
        input_layer = Input(shape=(data.seq_length,))
        embedding_layer = Embedding(data.max_feature, data.embed_dim,
                                    weights=[data.embed_matrix],
                                    trainable=self.embed_trainable)(input_layer)
        x = Bidirectional(GRU(data.embed_dim, return_sequences=True,
                              recurrent_dropout=self.dropout,
                              dropout=self.dropout))(embedding_layer)
        attention = AttLayer()(x)
        x = Dense(self.dense_size, activation="relu")(attention)
        output_layer = Dense(6, activation="sigmoid")(x)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        optimizer = get_optimizer(self.lr, self.optim_name)
        model.compile(loss='binary_crossentropy', optimizer=optimizer,
                      metrics=['accuracy'])
        self.model = model
        model_descirption = f'''Attention model
                        dense_size: {self.dense_size}
                        embed_trainbale: {self.embed_trainable}
                        lr: {self.lr}
                        optim_name: {self.optim_name}
                        batch_size: {self.batch_size}
                        dropout: {self.dropout}'''
        print(model_descirption)
        print(model.summary())


class DPCNNModel(BaseModel):
    def __init__(self, data, batch_size=128, lr=0.0015, optim_name=None):
        super().__init__(data, batch_size)
        if optim_name is None:
            optim_name = 'nadam'
        self.optim_name = optim_name
        self.lr = lr
        self.build_model()
        self.description = 'DPCNN Model'

    def build_model(self):
        data = self.data
        inputs = Input(shape=(data.seq_length,), dtype='int32')
        embedding = Embedding(data.max_feature, data.embed_dim, trainable=False,
                              weights=[data.embed_matrix])(inputs)

        filter_nr = data.embed_dim

        kernel_size = 3
        repeat_block = 3
        dense_size = 256

        reg_convo = 0.00001
        reg_dense = 0.00001
        dropout_convo = 0.0
        dropout_dense = 0.0

        x = _convolutional_block(filter_nr, kernel_size,
                                 dropout_convo, reg_convo)(embedding)
        x = _convolutional_block(filter_nr, kernel_size,
                                 dropout_convo, reg_convo)(x)
        x = add([embedding, x])

        for _ in range(repeat_block):
            x = MaxPooling1D(pool_size=3, strides=2)(x)
            main = _convolutional_block(filter_nr, kernel_size, dropout_convo,
                                        reg_convo)(x)
            main = _convolutional_block(filter_nr, kernel_size, dropout_convo,
                                        reg_convo)(main)
            x = add([main, x])

        x = GlobalMaxPool1D()(x)
        x = Dense(dense_size, activation='relu',
                  kernel_regularizer=regularizers.l2(reg_dense))(x)
        x = Dropout(dropout_dense)(x)
        output = Dense(6, activation="sigmoid")(x)

        model = keras.models.Model(inputs=inputs, outputs=output)
        optimizer = get_optimizer(self.lr, self.optim_name)
        model.compile(loss='binary_crossentropy', optimizer=optimizer,
                      metrics=['accuracy'])
        self.model = model

        model_descirption = f'''DPCNN model
                            batch_size : {self.batch_size}
                            lr: {self.lr}
                            kernel_size: {kernel_size}
                            repeat_block: {repeat_block}
                            dense_size: {dense_size}
                            reg_convo: {reg_convo}
                            reg_dense: {reg_dense}
                            dropout_convo: {dropout_convo}
                            dropout_dense: {dropout_dense}
                            '''
        print(model_descirption)
        print(model.summary())


def _convolutional_block(filter_nr, kernel_size, dropout, reg):
    def f(x):
        x = Conv1D(filter_nr, kernel_size=kernel_size, activation='relu',
                   padding='same', kernel_regularizer=regularizers.l2(reg))(x)
        x = Dropout(dropout)(x)
        return x

    return f

if __name__ == '__main__':
    class Data:
        def __init__(self):
            self.seq_length = 200
            self.max_feature = 170000
            self.embed_dim = 300
            self.embed_matrix = np.ones((self.max_feature, self.embed_dim))

    data = Data()
    model = CNNModel(data)
