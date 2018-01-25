import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

path = 'data/'

TRAIN_DATA_FILE = f'{path}train.csv'
TEST_DATA_FILE = f'{path}test.csv'
TRAIN_CLEAN_DATA_FILE = f'{path}train_clean.csv'
TEST_CLEAN_DATA_FILE = f'{path}test_clean.csv'

NAN_WORD = "_NAN_"

CLASSES = ["toxic", "severe_toxic", "obscene",
           "threat", "insult", "identity_hate"]


class DataSource:
    def __init__(self, embed_file, embed_dim, use_clean=True,
                 max_feature=20000):
        self.use_clean = use_clean
        self._embed_file = embed_file
        self.embed_dim = embed_dim
        self.max_feature = max_feature
        if use_clean:
            self.train_file = TRAIN_CLEAN_DATA_FILE
            self.test_file = TEST_CLEAN_DATA_FILE
        else:
            self.train_file = TRAIN_DATA_FILE
            self.test_file = TEST_DATA_FILE

        print(f'read train data: {self.train_file} '
              f'and test data: {self.test_file}')
        self.train_df = pd.read_csv(self.train_file)  # [0:3000]
        self.test_df = pd.read_csv(self.test_file)  # [0:3000]

        train_sentences = self.train_df["comment_text"].fillna(NAN_WORD).values
        test_sentences = self.test_df["comment_text"].fillna(NAN_WORD).values
        self.y_train = self.train_df[CLASSES].values

        print(f'train_sentences.shape {train_sentences.shape}')
        print(f'test_sentences.shape {test_sentences.shape}')
        print(f'y_train.shape {self.y_train.shape}')

        print('tokenzie sentence from train and test')
        self.x_train, self.x_test, words_dict, self.seq_length = \
            tokenize_sentences(train_sentences, test_sentences, max_feature)

        print(f'read embedding file {embed_file}')
        embeddings_index = read_embedding_list(self._embed_file, embed_dim)

        all_embs = np.stack(embeddings_index.values())
        emb_mean, emb_std = all_embs.mean(), all_embs.std()

        nb_words = min(max_feature, len(words_dict)+1)
        self.max_feature = nb_words
        embedding_matrix = np.random.normal(emb_mean, emb_std,
                                            (nb_words, embed_dim))
        in_word = 0
        total_word = len(words_dict)
        for word, i in words_dict.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                in_word = in_word + 1
            if i >= max_feature:
                continue
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        self.embed_matrix = embedding_matrix
        print(f'found {total_word} words in data, {in_word} words in embedding')

    def description(self):
        return f'''data source use 
        train data: {self.train_file}
        test data: {self.test_file}
        embed_file: {self._embed_file}
        embed_dim: {self.embed_dim}
        max_words: {self.max_feature}
        seq_length: {self.seq_length}
        '''


def tokenize_sentences(train_sentences, test_sentences, max_word):
    tokenizer = Tokenizer(num_words=max_word)
    tokenizer.fit_on_texts(list(train_sentences))
    list_tokenized_train = tokenizer.texts_to_sequences(train_sentences)
    list_tokenized_test = tokenizer.texts_to_sequences(test_sentences)
    total_count = len(list_tokenized_train) + len(list_tokenized_test)
    w = sum(map(len, list_tokenized_train)) + sum(map(len, list_tokenized_test))
    seq_len = 2 * int(w/total_count)
    seq_len = 100
    print(f'will use seq_len: {seq_len}')
    x_train = pad_sequences(list_tokenized_train, maxlen=seq_len)
    x_test = pad_sequences(list_tokenized_test, maxlen=seq_len)

    return x_train, x_test, tokenizer.word_index, seq_len


def read_embedding_list(file_path, embed_dim):
    res = {}
    for o in open(file_path, encoding='utf-8'):
        try:
            arr = o.strip().split()
            length = len(arr)
            if length != embed_dim+1:
                continue
            word = ''.join(arr[0:length-embed_dim])
            res[word] = np.asarray(arr[length-embed_dim:], dtype='float32')
        except Exception as e:
            print(e)
            print(o)
    return res


if __name__ == '__main__':
    embed_file = 'data/glove.840B.300d.txt'
    toxic_data = DataSource(embed_file, 300, use_clean=True)
