import pandas as pd
import numpy as np

result_file = 'logs/2018-01-23 22:04:29.044429/submit_devide1.2_0.060.csv'
output_file = 'logs/2018-01-23 22:04:29.044429/submit_origin.csv'
df = pd.read_csv(result_file)
label_cols = ['toxic', 'severe_toxic', 'obscene',
              'threat', 'insult', 'identity_hate']

df[label_cols] = df[label_cols] * 1.2

df.to_csv(output_file, index=False)

