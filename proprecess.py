import re
import random
import pandas as pd
import requests
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

CONTRACTION_MAP = {
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}

def _clean(comment):
    """
    This function receives comments and returns clean word-list
    :param comment: input comment sentences
    :return: cleaned sentence
    """
    tokenizer = TweetTokenizer()
    eng_stopwords = set(stopwords.words("english"))
    lem = WordNetLemmatizer()
    # Convert to lower case , so that Hi and hi are the same
    comment = comment.lower()
    # remove \n
    comment = re.sub("\\n", " ", comment)
    # remove leaky elements like ip,user
    comment = re.sub("\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}", "", comment)
    # removing usernames
    comment = re.sub("\[\[.*\]", "", comment)

    # Split the sentences into words
    words = tokenizer.tokenize(comment)

    # (')aphostophe  replacement (ie)   you're --> you are
    # ( basic dictionary lookup :
    # master dictionary present in a hidden block of code)
    words = [CONTRACTION_MAP[word] if word in CONTRACTION_MAP else word
             for word in words]
    words = [lem.lemmatize(word, "v") for word in words]
    words = [w for w in words if w not in eng_stopwords]

    clean_sent = " ".join(words)
    # remove any non alphanum,digit character
    clean_sent = re.sub("\W+", " ", clean_sent)
    clean_sent = re.sub("  ", " ", clean_sent)
    return clean_sent


def _shuffle(comment):
    """ shuffle the words in comment"""
    words = comment.split()
    random.shuffle(words)
    return " ".join(words)


def _drop(comment):
    """random drop some words in comment"""
    words = comment.split()
    return " ".join(word for word in words if random.random() > 0.3)


if __name__ == '__main__':
    train_path = 'data/train.csv'
    result_path = 'data/pre_train.csv'
    val_path = 'data/valid.csv'
    df = pd.read_csv(train_path)
    total_count = df.shape[0]
    split_index = int(total_count*0.1)
    val_df = df[:split_index]
    df = df[split_index:]
    funs = [_clean, _shuffle, _drop]
    df_list = [df]
    for fun in funs:
        source = df.copy()
        for index in df.index.values:
            com = df.at[index, 'comment_text']
            source.at[index, 'comment_text'] = fun(com)
        df_list.append(source)

    # # read csv from translated comments, see https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/48038
    # urls = ['https://kaggle2.blob.core.windows.net/forum-message-attachments/272289/8338/train_fr.csv',
    #         'https://kaggle2.blob.core.windows.net/forum-message-attachments/272289/8339/train_es.csv',
    #         'https://kaggle2.blob.core.windows.net/forum-message-attachments/272289/8340/train_de.csv']
    # for url in urls:
    #     df_list.append(pd.read_csv(url))

    result = pd.concat(df_list)
    result.to_csv(result_path, index=False)

    val_df.to_csv(val_path, index=False)



