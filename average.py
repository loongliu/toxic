import pandas as pd


def average_res(files, output, weights=None):
    if weights and len(weights) != len(files):
        raise ValueError('weights should has same lengths with files')
    if weights is None:
        weights = [1] * len(files)

    label_cols = ['toxic', 'severe_toxic', 'obscene',
                  'threat', 'insult', 'identity_hate']

    p_res = None
    for file, weight in zip(files, weights):
        df = pd.read_csv(file)
        if p_res is None:
            p_res = df.copy()
            p_res[label_cols] = df[label_cols] ** weight
        else:
            p_res[label_cols] += df[label_cols] ** weight
    if p_res is not None:
        p_res[label_cols] = p_res[label_cols] ** (1. / sum(weights))
        p_res.to_csv(output, index=False)


__all__ = [average_res]

if __name__ == '__main__':
    average_res(['logs/2018-02-10 11:22:01.665868/dpcnn.csv',
                 'logs/2018-02-08 16:38:07.051641/RCNN Model fold 9.csv',
                 'logs/2018-02-09 04:09:04.798535/Double GRU fold 9.csv',
                 'logs/2018-02-07 17:36:52.237852/CNN Model fold 9.csv',
                 'logs/2018-02-11 12:53:04.199236/attention.csv'],
                'logs/average_attention_dpcnn_cnn_rcnn_doublegru_0212.csv')
