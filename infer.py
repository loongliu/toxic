import keras
import data_source
import numpy as np
import pandas as pd


model_path_list = ['logs/2018-01-23 22:04:29.044429/LSTM with dropout epoch: 4 val_loss 0.04474.hdf5',
                   'logs/2018-01-23 22:04:29.044429/LSTM with dropout epoch: 5 val_loss 0.04518.hdf5',
                   'logs/2018-01-23 22:04:29.044429/LSTM with dropout epoch: 6 val_loss 0.04388.hdf5',
                   'logs/2018-01-23 22:04:29.044429/LSTM with dropout epoch: 6 val_loss 0.04549.hdf5',
                   'logs/2018-01-23 22:04:29.044429/LSTM with dropout epoch: 6 val_loss 0.04804.hdf5',
                   'logs/2018-01-23 22:04:29.044429/LSTM with dropout epoch: 7 val_loss 0.04446.hdf5',
                   'logs/2018-01-23 22:04:29.044429/LSTM with dropout epoch: 7 val_loss 0.04575.hdf5',
                   'logs/2018-01-23 22:04:29.044429/LSTM with dropout epoch: 7 val_loss 0.04725.hdf5',
                   'logs/2018-01-23 22:04:29.044429/LSTM with dropout epoch: 7 val_loss 0.04828.hdf5',
                   'logs/2018-01-23 22:04:29.044429/LSTM with dropout epoch: 7 val_loss 0.04877.hdf5',]
output_path = 'logs/2018-01-23 22:04:29.044429/submit.csv'

embed_file = 'data/glove.6B/glove.6B.100d.txt'

toxic_data = data_source.DataSource(embed_file, 100, use_clean=True)

test_predicts_list = []
for model_path in model_path_list:
    print(f'load first mode {model_path}')
    toxic_model = keras.models.load_model(model_path)
    print(f'will train  model')
    test_predicts = toxic_model.predict(toxic_data.x_test, batch_size=128)
    print('model train finish')
    test_predicts_list.append(test_predicts)

test_predicts = np.ones(test_predicts_list[0].shape)
for fold_predict in test_predicts_list:
    test_predicts *= fold_predict

test_predicts **= (1. / len(test_predicts_list))
test_predicts = test_predicts / 1.2

test_df = pd.read_csv(toxic_data.test_file)

test_ids = test_df["id"].values
test_ids = test_ids.reshape((len(test_ids), 1))

CLASSES = data_source.CLASSES

test_predicts = pd.DataFrame(data=test_predicts, columns=CLASSES)
test_predicts["id"] = test_ids
test_predicts = test_predicts[["id"] + CLASSES]
test_predicts.to_csv(output_path, index=False)

