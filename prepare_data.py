import torch
import torchtext
from torchtext.data import Field, BucketIterator
from sklearn.model_selection import train_test_split
import pandas as pd
from nltk import tokenize


def _tokenize_ru(text):
    return tokenize.word_tokenize(text)


def _tokenize_en(text):
    return tokenize.word_tokenize(text)


def _train_test_split(src_path, trg_path, batch_size, debug):
    with open(src_path) as fp:
        ru_lines = fp.readlines()

    with open(trg_path) as fp:
        en_lines = fp.readlines()
    if debug:
        ru_lines = ru_lines[:batch_size]
        en_lines = en_lines[:batch_size]
    raw_data = {'English': [line for line in en_lines], 'Russian': [line for line in ru_lines]}
    df = pd.DataFrame(raw_data, columns=["English", "Russian"])
    # remove very long sentences and sentences where translations are
    # not of roughly equal length
    df['eng_len'] = df['English'].str.count(' ')
    df['rus_len'] = df['Russian'].str.count(' ')
    df = df.query('rus_len < 80 & eng_len < 80')
    df = df.query('rus_len < eng_len * 1.5 & rus_len * 1.5 > eng_len')

    train, val = train_test_split(df, test_size=0.1)
    train.to_csv("train.csv", index=False)
    val.to_csv("val.csv", index=False)


def get_iterators(src_path, trg_path, batch_size=128, debug = False):
    """
    Prepare data for NMT task
    :param src_path: (str) path to file containing sentences on source language (Russian)
    :param trg_path: (str) path to file containing sentences on target language (English)
    :param batch_size: (int)
    :param debug: (bool) if True less lengths of  train_iterator, valid_iterator are only 2 batches
    :return: train_iterator, valid_iterator of type (torchtext.data.iterator.BucketIterator)
    """
    _train_test_split(src_path, trg_path)
    SRC = Field(tokenize=_tokenize_ru,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True,
                batch_first=True)
    TRG = Field(tokenize=_tokenize_en,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True,
                batch_first=True)

    data_fields = [('English', TRG), ('Russian', SRC)]
    # THIS LINES TAKES 10 MINUTES ON THE WHOLE DATASET
    train, val = torchtext.data.TabularDataset.splits(path='./',
                                                      train='train.csv',
                                                      validation='val.csv',
                                                      format='csv',
                                                      fields=data_fields)
    SRC.build_vocab(train, val)
    TRG.build_vocab(train, val)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_iterator, valid_iterator = BucketIterator.splits((train, val), batch_size=batch_size,
                                                           device=device)
    return train_iterator, valid_iterator
