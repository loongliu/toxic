import pandas as pd
import numpy as np
import nltk
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

path = 'data/'

TRAIN_DATA_FILE = f'{path}train.csv'
TEST_DATA_FILE = f'{path}test.csv'
VALID_DATA_FILE = f'{path}valid.csv'

UNKNOWN_WORD = "_UNK_"
END_WORD = "_END_"
NAN_WORD = "_NAN_"

CLASSES = ["toxic", "severe_toxic", "obscene",
           "threat", "insult", "identity_hate"]


class DataSource:
    def __init__(self, embed_file, embed_dim, seq_length=200,
                 max_feature=20000, extra_valid=False):
        self._embed_file = embed_file
        self.embed_dim = embed_dim
        self.max_feature = max_feature
        self.train_file = TRAIN_DATA_FILE
        self.test_file = TEST_DATA_FILE
        self.seq_length = seq_length

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

        if extra_valid:
            self.valid_file = VALID_DATA_FILE
            print(f'and valid data: {self.valid_file}')
            self.valid_df = pd.read_csv(self.valid_file)  # [0:3000]
            valid_sentences = self.valid_df["comment_text"]\
                .fillna(NAN_WORD).values
            self.y_valid = self.valid_df[CLASSES].values
            print(f'valid_sentences.shape {valid_sentences.shape}')
            print(f'y_valid.shape {self.y_valid.shape}')
        else:
            valid_sentences = None
            self.y_valid = None
            self.valid_file = None

        print("Tokenizing sentences in train set...")
        tokenized_sentences_train, words_dict = tokenize_sentences(
            train_sentences, {})

        print("Tokenizing sentences in test set...")
        tokenized_sentences_test, words_dict = tokenize_sentences(
            test_sentences, words_dict)

        if extra_valid:
            tokenized_sentences_valid, words_dict = tokenize_sentences(
                valid_sentences, words_dict)
        else:
            tokenized_sentences_valid = None

        words_dict[UNKNOWN_WORD] = len(words_dict)

        print("Loading embeddings...")
        embedding_list, embedding_word_dict = read_embedding_list(embed_file,
                                                                  embed_dim)
        embedding_size = embed_dim
        print("Preparing data...")
        embedding_list, embedding_word_dict = clear_embedding_list(
            embedding_list, embedding_word_dict, words_dict)

        embedding_word_dict[UNKNOWN_WORD] = len(embedding_word_dict)
        embedding_list.append([0.] * embedding_size)
        embedding_word_dict[END_WORD] = len(embedding_word_dict)
        embedding_list.append([-1.] * embedding_size)

        self.embed_matrix = np.array(embedding_list)
        self.max_feature = self.embed_matrix.shape[0]

        id_to_word = dict((index, word) for word, index in words_dict.items())
        train_list_of_token_ids = convert_tokens_to_ids(
            tokenized_sentences_train,
            id_to_word,
            embedding_word_dict,
            self.seq_length)
        test_list_of_token_ids = convert_tokens_to_ids(
            tokenized_sentences_test,
            id_to_word,
            embedding_word_dict,
            self.seq_length)
        self.x_train = np.array(train_list_of_token_ids)
        self.x_test = np.array(test_list_of_token_ids)
        if extra_valid:
            valid_list_of_token_ids = convert_tokens_to_ids(
                tokenized_sentences_valid,
                id_to_word,
                embedding_word_dict,
                self.seq_length)
            self.x_valid = np.array(valid_list_of_token_ids)
        else:
            self.x_valid = None

    def description(self):
        return f'''data source use 
        train data: {self.train_file}
        test data: {self.test_file}
        valid data: {self.valid_file}
        embed_file: {self._embed_file}
        embed_dim: {self.embed_dim}
        max_words: {self.max_feature}
        seq_length: {self.seq_length}
        '''


class FastData:

    def __init__(self, embed_dim, seq_length=500, ngram_range=2,
                 max_feature=20000, extra_valid=False):
        self.embed_dim = embed_dim
        self.max_feature = max_feature
        self.train_file = TRAIN_DATA_FILE
        self.test_file = TEST_DATA_FILE
        self.seq_length = seq_length
        self.ngram_rage = ngram_range

        print(f'read train data: {self.train_file} '
              f'and test data: {self.test_file}')
        self.train_df = pd.read_csv(self.train_file)   [0:30000]
        self.test_df = pd.read_csv(self.test_file)   [0:3000]

        train_sentences = self.train_df["comment_text"].fillna(NAN_WORD).values
        test_sentences = self.test_df["comment_text"].fillna(NAN_WORD).values
        self.y_train = self.train_df[CLASSES].values

        print(f'train_sentences.shape {train_sentences.shape}')
        print(f'test_sentences.shape {test_sentences.shape}')
        print(f'y_train.shape {self.y_train.shape}')

        if extra_valid:
            self.valid_file = VALID_DATA_FILE
            print(f'and valid data: {self.valid_file}')
            self.valid_df = pd.read_csv(self.valid_file)
            valid_sentences = self.valid_df["comment_text"]\
                .fillna(NAN_WORD).values
            self.y_valid = self.valid_df[CLASSES].values
            print(f'valid_sentences.shape {valid_sentences.shape}')
            print(f'y_valid.shape {self.y_valid.shape}')
        else:
            valid_sentences = None
            self.y_valid = None
            self.valid_file = None

        print('tokenzie sentence from train and test')
        self.x_train, self.x_valid, self.x_test, words_dict = \
            tokenize_sentences_simple(train_sentences, valid_sentences,
                                      test_sentences, max_feature)

        print('Adding {}-gram features'.format(ngram_range))
        # Create set of unique n-gram from the training set.
        ngram_set = set()
        for input_list in self.x_train:
            for i in range(2, ngram_range + 1):
                set_of_ngram = create_ngram_set(input_list, ngram_value=i)
                ngram_set.update(set_of_ngram)

        # Dictionary mapping n-gram token to a unique integer.
        # Integer values are greater than max_feature in order
        # to avoid collision with existing features.
        start_index = self.max_feature + 1
        token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
        indice_token = {token_indice[k]: k for k in token_indice}

        self.max_feature = np.max(list(indice_token.keys())) + 1

        # Augmenting x_train and x_test with n-grams features
        x_train = add_ngram(self.x_train,
                            token_indice, ngram_range)
        x_test = add_ngram(self.x_test,
                           token_indice, ngram_range)
        print('Average train sequence length: {}'.format(
            np.mean(list(map(len, x_train)), dtype=int)))
        print('Average test sequence length: {}'.format(
            np.mean(list(map(len, x_test)), dtype=int)))

        print('Pad sequences (samples x time)')
        self.x_train = pad_sequences(x_train, maxlen=self.seq_length)
        self.x_test = pad_sequences(x_test, maxlen=self.seq_length)
        print('x_train shape:', self.x_train.shape)
        print('x_test shape:', self.x_test.shape)

        if extra_valid:
            x_valid = add_ngram(self.x_valid,
                                token_indice, ngram_range)
            print('Average valid sequence length: {}'.format(
                np.mean(list(map(len, x_valid)), dtype=int)))
            self.x_valid = pad_sequences(x_valid, maxlen=self.seq_length)
            print('x_valid shape:', self.x_valid.shape)

    def description(self):
        return f'''data source use 
        train data: {self.train_file}
        test data: {self.test_file}
        valid data: {self.valid_file}
        embed_dim: {self.embed_dim}
        max_feature: {self.max_feature}
        seq_length: {self.seq_length}
        '''


def tokenize_sentences_simple(train_sentences, valid_sentences, test_sentences, max_word):
    tokenizer = Tokenizer(num_words=max_word)
    tokenizer.fit_on_texts(list(train_sentences))
    x_train = tokenizer.texts_to_sequences(train_sentences)
    x_test = tokenizer.texts_to_sequences(test_sentences)
    # seq_len = 500
    # print(f'will use seq_len: {seq_len}')
    # x_train = pad_sequences(list_tokenized_train, maxlen=seq_len)
    # x_test = pad_sequences(list_tokenized_test, maxlen=seq_len)
    if valid_sentences is not None:
        x_valid = tokenizer.texts_to_sequences(valid_sentences)
        # x_valid = pad_sequences(list_tokenized_valid, maxlen=seq_len)
    else:
        x_valid = None
    return x_train, x_valid, x_test, tokenizer.word_index


def create_ngram_set(input_list, ngram_value=2):
    """
    Extract a set of n-grams from a list of integers.
    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}
    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    """
    Augment the input list of list (sequences) by appending n-grams values.
    Example: adding bi-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    >>> add_ngram(sequences, token_indice, ngram_range=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]
    Example: adding tri-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
    >>> add_ngram(sequences, token_indice, ngram_range=3)
    [[1, 3, 4, 5, 1337], [1, 3, 7, 9, 2, 1337, 2018]]
    """
    new_sequences = []
    for input_list in sequences:
        input_list = list(input_list)
        new_list = input_list[:]
        for i in range(len(new_list) - ngram_range + 1):
            for ngram_value in range(2, ngram_range + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences


def tokenize_sentences(sentences, words_dict):
    tokenized_sentences = []
    for sentence in sentences:
        if hasattr(sentence, "decode"):
            sentence = sentence.decode("utf-8")
        tokens = nltk.tokenize.word_tokenize(sentence)
        result = []
        for word in tokens:
            word = word.lower()
            if word not in words_dict:
                words_dict[word] = len(words_dict)
            word_index = words_dict[word]
            result.append(word_index)
        tokenized_sentences.append(result)
    return tokenized_sentences, words_dict


def read_embedding_list(file_path, embed_dim):
    embedding_word_dict = {}
    embedding_list = []
    for o in open(file_path, encoding='utf-8'):
        try:
            arr = o.strip().split()
            length = len(arr)
            if length != embed_dim+1:
                continue
            word = arr[0]
            embedding = np.array([float(num) for num in arr[1:]])
            embedding_list.append(embedding)
            embedding_word_dict[word] = len(embedding_word_dict)
        except Exception as e:
            print(e)
            print(o)

    embedding_list = np.array(embedding_list)
    return embedding_list, embedding_word_dict


def clear_embedding_list(embedding_list, embedding_word_dict, words_dict):
    cleared_embedding_list = []
    cleared_embedding_word_dict = {}

    for word in words_dict:
        if word not in embedding_word_dict:
            continue
        word_id = embedding_word_dict[word]
        row = embedding_list[word_id]
        cleared_embedding_list.append(row)
        cleared_embedding_word_dict[word] = len(cleared_embedding_word_dict)

    return cleared_embedding_list, cleared_embedding_word_dict


def convert_tokens_to_ids(tokenized_sentences, words_list, embedding_word_dict, sentences_length):
    words_train = []

    for sentence in tokenized_sentences:
        current_words = []
        for word_index in sentence:
            word = words_list[word_index]
            word_id = embedding_word_dict.get(word, len(embedding_word_dict) - 2)
            current_words.append(word_id)

        if len(current_words) >= sentences_length:
            current_words = current_words[:sentences_length]
        else:
            current_words += [len(embedding_word_dict) - 1] * (sentences_length - len(current_words))
        words_train.append(current_words)
    return words_train


if __name__ == '__main__':
    embed_f = 'data/glove.840B.300d.txt'
    toxic_data = FastData(300)
    print(toxic_data.description())
