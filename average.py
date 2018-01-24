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
            p_res[label_cols] = df[label_cols] * weight
        else:
            p_res[label_cols] += df[label_cols] * weight
    if p_res is not None:
        p_res[label_cols] = p_res[label_cols] / sum(weights)
        p_res.to_csv(output, index=False)


__all__ = [average_res]

if __name__ == '__main__':
    average_res(['logs/2018-01-24 10:37:06.073794/CNN Model divide1.2_0.060.csv',
                 'logs/2018-01-23 22:04:29.044429/submit_devide1.2_0.060.csv',],
                'logs/average_lstm_cnn.csv')
