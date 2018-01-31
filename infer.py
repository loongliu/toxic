import data_source
import numpy as np
import pandas as pd
import os
import keras
from model import AttLayer

# keras.models.load_model(model_path)


def infer_result(models, toxic_data, result_dir, result_name='submit_devide1.2_0.060.csv'):
    if not isinstance(models, list):
        models = [models]
    test_predicts_list = []
    for toxic_model in models:
        print(f'will train  model')
        test_predicts = toxic_model.predict(toxic_data.x_test,
                                            batch_size=128)
        print('model train finish')
        test_predicts_list.append(test_predicts)

    test_predicts = np.ones(test_predicts_list[0].shape)
    for fold_predict in test_predicts_list:
        test_predicts *= fold_predict

    test_predicts **= (1. / len(test_predicts_list))

    test_ids = toxic_data.test_df["id"].values
    test_ids = test_ids.reshape((len(test_ids), 1))

    CLASSES = data_source.CLASSES

    test_predicts = pd.DataFrame(data=test_predicts, columns=CLASSES)
    test_predicts["id"] = test_ids
    test_predicts = test_predicts[["id"] + CLASSES]

    output_path = os.path.join(result_dir, result_name)
    test_predicts.to_csv(output_path, index=False)


if __name__ == '__main__':
    model_path_list = [
        'logs/2018-01-24 22:10:36.654893/Double GRU fold 0 epoch: 2 val_loss 0.04406.hdf5',
        'logs/2018-01-24 22:10:36.654893/Double GRU fold 1 epoch: 2 val_loss 0.04661.hdf5',
        'logs/2018-01-24 22:10:36.654893/Double GRU fold 2 epoch: 2 val_loss 0.04275.hdf5',
        'logs/2018-01-24 22:10:36.654893/Double GRU fold 3 epoch: 2 val_loss 0.04552.hdf5',
        'logs/2018-01-24 22:10:36.654893/Double GRU fold 4 epoch: 2 val_loss 0.04382.hdf5',
        'logs/2018-01-24 22:10:36.654893/Double GRU fold 5 epoch: 2 val_loss 0.04232.hdf5',
        'logs/2018-01-24 22:10:36.654893/Double GRU fold 6 epoch: 2 val_loss 0.04299.hdf5',
        'logs/2018-01-24 22:10:36.654893/Double GRU fold 7 epoch: 2 val_loss 0.04510.hdf5',
        'logs/2018-01-24 22:10:36.654893/Double GRU fold 8 epoch: 2 val_loss 0.04363.hdf5',]

    model_list = [keras.models.load_model(path, custom_objects={'AttLayer':AttLayer}) for path in model_path_list]
    result_path = 'logs/2018-01-24 22:10:36.654893'

    embed_file = 'data/crawl-300d-2M.vec'
    toxic_data = data_source.DataSource(embed_file, 300)

    infer_result(model_list, toxic_data, result_path, 'double_gru_submit.csv')

