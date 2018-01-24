import data_source
import numpy as np
import pandas as pd
import os

# keras.models.load_model(model_path)


def infer_result(models, toxic_data, result_dir, result_name='submit.csv'):
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
    test_predicts = test_predicts / 1.2

    test_ids = toxic_data.test_df["id"].values
    test_ids = test_ids.reshape((len(test_ids), 1))

    CLASSES = data_source.CLASSES

    test_predicts = pd.DataFrame(data=test_predicts, columns=CLASSES)
    test_predicts["id"] = test_ids
    test_predicts = test_predicts[["id"] + CLASSES]

    output_path = os.path.join(result_dir, result_name)
    test_predicts.to_csv(output_path, index=False)

