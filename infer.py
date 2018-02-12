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
        'logs/2018-02-11 12:53:04.199236/Attention Model with dropout fold 1 epoch: 12 val_auc 0.99015.hdf5',
        'logs/2018-02-11 12:53:04.199236/Attention Model with dropout fold 2 epoch: 7 val_auc 0.99041.hdf5',
        'logs/2018-02-11 12:53:04.199236/Attention Model with dropout fold 3 epoch: 13 val_auc 0.98759.hdf5',
        'logs/2018-02-11 12:53:04.199236/Attention Model with dropout fold 4 epoch: 10 val_auc 0.98773.hdf5',
        'logs/2018-02-11 12:53:04.199236/Attention Model with dropout fold 5 epoch: 7 val_auc 0.98882.hdf5',
        'logs/2018-02-11 12:53:04.199236/Attention Model with dropout fold 6 epoch: 11 val_auc 0.98978.hdf5',
        'logs/2018-02-11 12:53:04.199236/Attention Model with dropout fold 7 epoch: 11 val_auc 0.98917.hdf5',
        'logs/2018-02-11 12:53:04.199236/Attention Model with dropout fold 8 epoch: 10 val_auc 0.98990.hdf5',
        'logs/2018-02-11 12:53:04.199236/Attention Model with dropout fold 9 epoch: 14 val_auc 0.99065.hdf5',
        'logs/2018-02-11 08:14:30.405323/Attention Model with dropout epoch: 19 val_auc 0.98986.hdf5',]

    model_list = [keras.models.load_model(path, custom_objects={'AttLayer':AttLayer}) for path in model_path_list]
    result_path = 'logs/2018-02-11 12:53:04.199236'

    embed_file = 'data/crawl-300d-2M.vec'
    toxic_data = data_source.FastData(seq_length=200)

    infer_result(model_list, toxic_data, result_path, 'attention.csv')

