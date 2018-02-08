import pandas as pd
import numpy as np
import nltk

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


class DataSource:
    def __init__(self, embed_file, embed_dim, seq_length=200,
                 max_feature=20000):
        self._embed_file = embed_file
        self.embed_dim = embed_dim
        self.max_feature = max_feature
        self.train_file = TRAIN_DATA_FILE
        self.test_file = TEST_DATA_FILE
        self.seq_length = seq_length

        print(f'read train data: {self.train_file} '
              f'and test data: {self.test_file}')
        self.train_df = pd.read_csv(self.train_file)  # [0:300]
        self.test_df = pd.read_csv(self.test_file)  # [0:300]

        train_sentences = self.train_df["comment_text"].fillna(NAN_WORD).values
        test_sentences = self.test_df["comment_text"].fillna(NAN_WORD).values
        self.y_train = self.train_df[CLASSES].values

        print(f'train_sentences.shape {train_sentences.shape}')
        print(f'test_sentences.shape {test_sentences.shape}')
        print(f'y_train.shape {self.y_train.shape}')

        print("Tokenizing sentences in train set...")
        tokenized_sentences_train, words_dict = tokenize_sentences(
            train_sentences, {})

        print("Tokenizing sentences in test set...")
        tokenized_sentences_test, words_dict = tokenize_sentences(
            test_sentences, words_dict)

        tokenized_train_list = []
        for train_pro in TRAIN_PROCESS_FILES:
            df = pd.read_csv(train_pro)  # [0:300]
            sent = df["comment_text"].fillna(NAN_WORD).values
            tokenized_sen, words_dict = tokenize_sentences(sent, words_dict)
            tokenized_train_list.append(tokenized_sen)

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
        self.x_pre = []
        for pro_sen in tokenized_train_list:
            pro_ids = convert_tokens_to_ids(pro_sen, id_to_word,
                                            embedding_word_dict,
                                            self.seq_length)
            self.x_pre.append(np.array(pro_ids))

    def description(self):
        return f'''data source use 
        train data: {self.train_file}
        test data: {self.test_file}
        embed_file: {self._embed_file}
        embed_dim: {self.embed_dim}
        max_words: {self.max_feature}
        seq_length: {self.seq_length}
        '''


class FastData:

    def __init__(self):
        import parse_fasttext as pf
        self.x_train, self.y_train = pf.parse_file(TRAIN_DATA_FILE, hasy=True)
        self.x_test = pf.parse_file(TEST_DATA_FILE)
        self.seq_length = self.x_train.shape[1]
        self.embed_dim = self.x_train.shape[2]
        print('train_x.shape', self.x_train.shape)
        print('train_y.shape', self.y_train.shape)
        print('test_x.shape', self.x_test.shape)

    def description(self):
        return f'''fast text data source
        train_x.shape: {self.x_train.shape}
        train_y.shape: {self.y_train.shape}
        test_x.shape: {self.x_test.shape}
        '''


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
    embed_f = 'data/glove.6B/glove.6B.50d.txt'
    toxic_data = FastData()
    print(toxic_data.description())
