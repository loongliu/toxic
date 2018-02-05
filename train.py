import os
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score
import datetime
from keras.callbacks import LearningRateScheduler


def _train_model(toxic_model, batch_size, train_x, train_y,
                 val_x, val_y, model_dir):
    best_auc = -1
    best_weights = None
    best_epoch = 0

    current_epoch = 0
    print(f'will train new model train.size {train_x.shape[0]},'
          f' val.size {val_x.shape[0]}')

    hdf5_path = ''
    while True:
        # def step_decay(epoch):
        #     if epoch <= 2:
        #         return 0.003
        #     if epoch <= 4:
        #         return 0.0015
        #     if epoch <= 6 :
        #         return 0.001
        #     return 0.0005
        # lrate = LearningRateScheduler(step_decay, 1)
        # toxic_model.model.fit(train_x, train_y, batch_size=batch_size,
        #                       initial_epoch=current_epoch,
        #                       epochs = current_epoch + 1,
        #                       verbose = 0, callbacks=[lrate])
        toxic_model.model.fit(train_x, train_y, batch_size=batch_size,
                              epochs=1, verbose=0)
        y_pred = toxic_model.model.predict(val_x, batch_size=batch_size)

        total_loss = 0
        total_auc = 0
        for j in range(6):
            loss = log_loss(val_y[:, j], y_pred[:, j], eps=1e-5)
            total_loss += loss
            auc = roc_auc_score(val_y[:, j], y_pred[:, j])
            total_auc += auc

        total_loss /= 6.
        total_auc /= 6.

        print("Epoch {0} loss {1} auc {2} best_auc {3}".
              format(current_epoch, total_loss, total_auc, best_auc))
        current_epoch += 1
        if total_auc > best_auc:
            best_auc = total_auc
            best_weights = toxic_model.model.get_weights()
            best_epoch = current_epoch

            if hdf5_path != '':
                os.remove(hdf5_path)
            model_name = toxic_model.description + \
                f' epoch: {current_epoch} val_auc {best_auc:.5f}.hdf5'
            hdf5_path = os.path.join(model_dir, model_name)
            toxic_model.model.save(hdf5_path)
        else:
            if current_epoch - best_epoch == 3:
                break

    toxic_model.model.set_weights(best_weights)


def create_model_path():
    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_dir = os.path.join(model_dir, str(datetime.datetime.now()))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return model_dir


def train(toxic_model, model_dir, valid_split=0.1):
    print(f'call train funciton')
    if toxic_model.data.x_valid is not None:
        val_x = toxic_model.data.x_valid
        val_y = toxic_model.data.y_valid
        train_x = toxic_model.data.x_train
        train_y = toxic_model.data.y_train
    else:
        x = toxic_model.data.x_train
        y = toxic_model.data.y_train
        size = x.shape[0]
        split_index = int(size * valid_split)
        val_x = x[:split_index]
        val_y = y[:split_index]
        train_x = x[split_index:]
        train_y = y[split_index:]

    if toxic_model.model is None:
        raise ValueError('model not defined!')
    _train_model(toxic_model, toxic_model.batch_size,
                 train_x, train_y, val_x, val_y, model_dir)
    return toxic_model.model


def train_folds(toxic_model, fold_count, log_dir):
    print(f'call train_folds fold_count: {fold_count}')
    X = toxic_model.data.x_train
    y = toxic_model.data.y_train
    fold_size = len(X) // fold_count
    origin_description = toxic_model.description
    models = []
    for fold_id in range(0, fold_count):
        fold_start = fold_size * fold_id
        fold_end = fold_start + fold_size

        if fold_id == fold_size - 1:
            fold_end = len(X)

        train_x = np.concatenate([X[:fold_start], X[fold_end:]])
        train_y = np.concatenate([y[:fold_start], y[fold_end:]])

        val_x = X[fold_start:fold_end]
        val_y = y[fold_start:fold_end]
        toxic_model.build_model()
        toxic_model.description = origin_description + f" fold {fold_id}"
        _train_model(toxic_model, toxic_model.batch_size,
                     train_x, train_y, val_x, val_y, log_dir)
        models.append(toxic_model.model)
    return models
