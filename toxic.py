import data_source
import model
import log
import train
import infer

log_dir = log.init_log()

embed_file = 'data/glove.6B/glove.6B.100d.txt'

toxic_data = data_source.DataSource(embed_file, 100, use_clean=True)

print(toxic_data.description())

train_model = model.CNNModel(toxic_data, batch_size=128)

train_fold = False

if train_fold:
    result_model = train.train_folds(train_model, 10, log_dir)
else:
    result_model = train.train(train_model, log_dir)
print('train finish', 'result_model: ', result_model)
result_file = train_model.description + '.csv'
infer.infer_result(result_model, toxic_data, log_dir, result_file)

