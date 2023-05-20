import unicodedata
import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class WordDataset(Dataset):
    '''
    Reads an input CSV files containing word-pairs written in roman script and tamil script.
    Stores the words in onehot representation.
    '''
    def __init__(self, file = 'tam_train.csv'):
        self.file = file
        self.xy_str = np.loadtxt(self.file, delimiter=',', dtype=str, encoding='utf8')
        self.num_samples = self.xy_str.shape[0]
        self.x_str = self.xy_str[:, 0]
        self.y_str = self.xy_str[:, 1]
        self.get_onehot_encoding_x()
        self.get_onehot_encoding_y()
    
    def __getitem__(self, i):
        return self.x_onehot[i], self.y_onehot[i]

    def __len__(self):
        return self.num_samples

    ## Functions for onehot encoding of English words.

    def encode_eng_char(self, c):
        encoding = torch.zeros(128, dtype=torch.int32).view(1, 128)
        if(c == '$'):
            encoding[0][127] = 1
        elif (c == '.'):
            encoding[0][0] = 1
        else:
            encoding[0][ord(c)] = 1
        return encoding

    def decode_eng_char(self, v):
        '''
        Input: 1-D array of length 128.
        '''
        x = torch.where(v == 1)[0].item()
        if(x == 0):
            return('.')
        if (x == 127):
            return('$')
        return chr(x)
            
    def encode_eng_word(self, word):
        '''
        Output: Torch array of dimension 50 x 128.
                Each row corresponds to onehot encoding of a character.
        '''
        i = 0
        encoding = torch.zeros(50, 128, dtype=torch.int32)
        while i < len(word):
            encoding[i] = self.encode_eng_char(word[i])
            i += 1
        encoding[i] = self.encode_eng_char('$')
        i += 1
        while i < 50:
            encoding[i] = self.encode_eng_char('.')
            i += 1
        return encoding
    
    def decode_eng_word(self, arr):
        word = ''
        for i in range(arr.shape[0]):
            c = self.decode_eng_char(arr[i])
            if(c == '$'):
                break
            word += c
        return word
    
    def get_onehot_encoding_x(self):
        self.x_onehot = torch.zeros(self.num_samples, 50, 128, dtype=torch.int32)
        for i in tqdm(range(self.num_samples), desc = 'Encoding English Words...'):
            self.x_onehot[i] = self.encode_eng_word(self.x_str[i])
 
    ## Functions for onehot encoding of Tamil words.

    def encode_tam_char(self, c):
        encoding = torch.zeros(128, dtype=torch.int32).view(1, 128)
        if(c == '$'):
            encoding[0][127] = 1
        elif (c == '.'):
            encoding[0][126] = 1
        else:
            encoding[0][ord(c) - 0xb80] = 1
        return encoding

    def decode_tam_char(self, v):
        '''
        Input: 1-D array of length 128.
        '''
        x = torch.where(v == 1)[0].item()
        if(x == 0):
            return('.')
        if (x == 127):
            return('$')
        return chr(x + 0xb80)
            
    def encode_tam_word(self, word):
        '''
        Output: Torch array of dimension 50 x 128.
                Each row corresponds to onehot encoding of a character.
        '''
        i = 0
        encoding = torch.zeros(50, 128, dtype=torch.int32)
        while i < len(word):
            encoding[i] = self.encode_tam_char(word[i])
            i += 1
        encoding[i] = self.encode_tam_char('$')
        i += 1
        while i < 50:
            encoding[i] = self.encode_tam_char('.')
            i += 1
        return encoding
    
    def decode_tam_word(self, arr):
        word = ''
        for i in range(arr.shape[0]):
            c = self.decode_tam_char(arr[i])
            if(c == '$'):
                break
            word += c
        return word


    def get_onehot_encoding_y(self):
        self.y_onehot = torch.zeros(self.num_samples, 50, 128, dtype=torch.int32)
        for i in tqdm(range(self.num_samples), desc='Encoding Tamil Words...'):
            self.y_onehot[i] = self.encode_tam_word(self.y_str[i])
