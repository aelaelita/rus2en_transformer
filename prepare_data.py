import torch
import torchtext
from torchtext.data import Field, BucketIterator
from sklearn.model_selection import train_test_split
import pandas as pd
from nltk import tokenize


class NMTDataset:
    def __init__(self, src_path, trg_path):
        self.src_path = src_path
        self.trg_path = trg_path
        self.INPUT_DIM = None
        self.OUTPUT_DIM = None
        self.src_data = None
        self.trg_data = None

    def _tokenize_ru(self, text):
        return tokenize.word_tokenize(text)

    def _tokenize_en(self, text):
        return tokenize.word_tokenize(text)

    def _train_test_split(self, batch_size, debug):
        with open(self.src_path) as fp:
            ru_lines = fp.readlines()

        with open(self.trg_path) as fp:
            en_lines = fp.readlines()
        if debug:
            ru_lines = ru_lines[:5 * batch_size]
            en_lines = en_lines[:5 * batch_size]
        raw_data = {'English': [line for line in en_lines], 'Russian': [line for line in ru_lines]}
        df = pd.DataFrame(raw_data, columns=["English", "Russian"])
        train, val = train_test_split(df, test_size=0.1)
        train.to_csv("train.csv", index=False)
        val.to_csv("val.csv", index=False)

    def get_iterators(self, batch_size=128, debug=False):
        """
        Prepare data for NMT task
        :param src_path: (str) path to file containing sentences on source language (Russian)
        :param trg_path: (str) path to file containing sentences on target language (English)
        :param batch_size: (int)
        :param debug: (bool) if True less lengths of  train_iterator, valid_iterator are only 2 batches
        :return: train_iterator(torchtext.data.iterator.BucketIterator), valid_iterator (torchtext.data.iterator.BucketIterator),
         INPUT_DIM (int), OUTPUT_DIM (int)
        Example of usage:
        >>>train_iterator, valid_iterator = get_iterators(batch_size=128, debug=True)
        >>>for i, batch in enumerate(train_iterator):
        >>>    src = batch.Russian
        >>>    trg = batch.English
        """
        self._train_test_split(batch_size, debug)

        SRC = Field(tokenize=self._tokenize_ru,
                    init_token='<sos>',
                    eos_token='<eos>',
                    lower=True,
                    batch_first=True,
                    fix_length=100)
        TRG = Field(tokenize=self._tokenize_en,
                    init_token='<sos>',
                    eos_token='<eos>',
                    lower=True,
                    batch_first=True,
                    fix_length=100)

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

        self.src_data = SRC
        self.trg_data = TRG
        self.INPUT_DIM = len(SRC.vocab)
        self.OUTPUT_DIM = len(TRG.vocab)

        return train_iterator, valid_iterator

    def get_src_tag_idx(self, tag):
        return self.src_data.vocab.stoi[tag]

    def get_trg_tag_idx(self, tag):
        return self.trg_data.vocab.stoi[tag]

    def get_src_tag_from_idx(self, idx):
        return self.src_data.vocab.itos[idx]

    def get_trg_tag_from_idx(self, idx):
        return self.trg_data.vocab.itos[idx]
