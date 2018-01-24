import keras
from keras.layers import Dense, Input, LSTM, Embedding, Bidirectional
from keras.layers import Dropout, BatchNormalization, GlobalMaxPool1D
from keras.layers import Conv1D, GlobalMaxPooling1D, Activation
from keras import optimizers as k_opt


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


class CNNModel(BaseModel):
    def __init__(self, data, batch_size=64, embed_trainable=False,
                 filter_count=128, lr=0.001, optim_name=None, dense_size=100):
        super().__init__(data, batch_size)
        if optim_name is None:
            optim_name = 'nadam'
        self.optim_name = optim_name
        self.embed_trainable = embed_trainable
        self.filter_count = filter_count
        self.lr = lr
        self.dense_size = dense_size
        self.description = 'CNN Model'

    def build_model(self):
        data = self.data
        inputs = Input(shape=(data.seq_length,), dtype='int32')
        x = Embedding(data.max_feature, data.embed_dim,
                      weights=[data.embed_matrix],
                      trainable=self.embed_trainable)(inputs)
        con1 = Conv1D(self.filter_count, 3, activation='relu')(x)
        pool1 = GlobalMaxPooling1D()(con1)

        dense1 = Dense(self.dense_size, activation='relu')(pool1)

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
                optim_name: {self.optim_name}
                batch_size: {self.batch_size}'''
        print(model_descirption)
        print(model.summary())
