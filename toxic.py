import data_source
import model
import log
import train
import keras

log.init_log()

embed_file = 'data/glove.6B/glove.6B.100d.txt'

toxic_data = data_source.DataSource(embed_file, 100, use_clean=True)

print(toxic_data.description())

print(1)

train_model = model.CNNModel(toxic_data, batch_size=128)

train.train_folds(train_model, 10)
#
# print(2)

#
# filepath = 'models/2018-01-23 17:07:19.278381/LSTM modelepoch: 4 val_loss 0.04920.hdf5'
#
# keras.models.load_model(filepath)

