import pandas as pd
import numpy as np
import nltk
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from fastText import load_model

path = 'data/'

TRAIN_DATA_FILE = f'{path}train.csv'
TEST_DATA_FILE = f'{path}test.csv'
VALID_DATA_FILE = f'{path}valid.csv'
TRAIN_PROCESS_FILES = [
    f'{path}train_clean.csv',
    f'{path}train_drop.csv',
    f'{path}train_shuffle.csv',
    f'{path}train_de.csv',
    f'{path}train_es.csv',
    f'{path}train_fr.csv',
]

UNKNOWN_WORD = "_UNK_"
END_WORD = "_END_"
NAN_WORD = "_NAN_"

CLASSES = ["toxic", "severe_toxic", "obscene",
           "threat", "insult", "identity_hate"]


class FastData:
    def __init__(self, seq_length=300):
        self.seq_length = seq_length
        self.train_file = TRAIN_DATA_FILE
        self.test_file = TEST_DATA_FILE
        self.train_df = pd.read_csv(self.train_file)  # [0:3000]
        self.test_df = pd.read_csv(self.test_file)  # [0:3000]
        train_sentences = self.train_df["comment_text"].fillna(NAN_WORD).values
        test_sentences = self.test_df["comment_text"].fillna(NAN_WORD).values
        self.y_train = self.train_df[CLASSES].values

        print(f'train_sentences.shape {train_sentences.shape}')
        print(f'test_sentences.shape {test_sentences.shape}')
        print(f'y_train.shape {self.y_train.shape}')

        print('tokenzie sentence from train and test')
        self.x_train, self.x_test, words_dict = tokenize_sentences_sample(
            train_sentences, test_sentences, self.seq_length)
        ft_model = load_model('data/wiki.en.bin')
        self.embed_dim = ft_model.get_dimension()
        self.max_feature = len(words_dict) + 1
        self.embed_matrix = np.zeros((self.max_feature, self.embed_dim), dtype=np.float32)
        for word, index in words_dict.items():
            self.embed_matrix[index, :] = ft_model.get_word_vector(word).astype('float32')
        print(f'train_x.shape {self.x_train.shape}')
        print(f'train_y.shape {self.y_train.shape}')
        print(f'test_x.shape {self.x_test.shape}')

    def description(self):
        return f'''fast text data source
        seq_length: {self.seq_length}
        max_feature: {self.max_feature}
        train_x.shape: {self.x_train.shape}
        train_y.shape: {self.y_train.shape}
        test_x.shape: {self.x_test.shape}
        '''


def tokenize_sentences_sample(train_sentences, test_sentences, maxlen):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(list(train_sentences))
    list_tokenized_train = tokenizer.texts_to_sequences(train_sentences)
    list_tokenized_test = tokenizer.texts_to_sequences(test_sentences)

    x_train = pad_sequences(list_tokenized_train, maxlen=maxlen)
    x_test = pad_sequences(list_tokenized_test, maxlen=maxlen)

    return x_train, x_test, tokenizer.word_index


if __name__ == '__main__':
    embed_f = 'data/glove.6B/glove.6B.50d.txt'
    toxic_data = FastData()
    print(toxic_data.description())
