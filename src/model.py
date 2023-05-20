import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.join(sys.path[0], '../src'))
import data
import utils

class EncoderRNN(nn.Module):
    def __init__(self, embedding_size = 16, hidden_size = 16, num_layers = 1, cell_type = 'rnn', bidir = False, dropout = 0):
        super(EncoderRNN, self).__init__()

        self.input_size = 128
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.bidir = bidir

        self.embedding = nn.Embedding(self.input_size, embedding_size)

        if (cell_type == 'rnn'):
            self.RNN = nn.RNN(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional = bidir, dropout = dropout)
        elif (cell_type == 'lstm'):
            self.RNN = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional = bidir, dropout = dropout)
        elif (cell_type == 'gru'):
            self.RNN = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional = bidir, dropout = dropout)

        self.initialize_hidden()
    
    def forward(self, input):
        embedded = self.embedding(torch.where(input == 1)[0])
        output, hidden = self.RNN(embedded)
        return output, hidden

    def initialize_hidden(self):
        if(self.bidir):
            self.hidden = torch.zeros(2 * self.num_layers * self.hidden_size).view(2 * self.num_layers, self.hidden_size)
        else:
            self.hidden = torch.zeros(self.num_layers * self.hidden_size).view(self.num_layers, self.hidden_size)

class DecoderRNN(nn.Module):
    def __init__(self, embedding_size = 16, hidden_size = 16, num_layers = 1, cell_type = 'rnn', bidir = False, dropout = 0):
        super(DecoderRNN, self).__init__()

        self.input_size = 128
        self.output_size = 128
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.bidir = bidir
        self.dropout = dropout

        self.embedding = nn.Embedding(self.input_size, self.embedding_size)

        if (cell_type == 'rnn'):
            self.RNN = nn.RNN(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional = bidir, dropout = dropout)
        elif (cell_type == 'lstm'):
            self.RNN = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional = bidir, dropout = dropout)
        elif (cell_type == 'gru'):
            self.RNN = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional = bidir, dropout = dropout)

        self.out = nn.Linear(self.hidden_size, self.output_size)
        # self.out_act = nn.LogSoftmax()
        self.out_act = nn.Softmax()

    def forward(self, input, hidden):
        embedded = self.embedding(torch.where(input == 1)[0])
        embedded = F.relu(embedded)
        output, hidden = self.RNN(embedded, hidden)
        output = self.out(output)
        output = self.out_act(output)
        return output, hidden