import os
import sys
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.join(sys.path[0], '../src'))
import data
import utils
import model
from model import EncoderRNN
from model import DecoderRNN

def MSELoss(y_pred, y):
    one_pos = torch.tensor([torch.where(y == 1)[0]])
    return torch.log(y_pred[0][one_pos])


def train_one_iter(x, y, encoder, decoder, encoder_optim, decoder_optim, criterion):

    # for i in range(50):
        # out, hidden = encoder.forward(x[i])
        # if(x[i][127] == 1):
            # break
    
    out, hidden = encoder.forward(x)

    sos = torch.zeros(128)
    sos[0] = 1
    out = sos

    loss = 0

    for i in range(50):
        out, hidden = decoder.forward(out, hidden)
        target = y[i]
        target = torch.tensor([torch.where(target == 1)[0]])

        # loss += criterion(out, target)
        loss += MSELoss(out, y[i])

        max_index = torch.argmax(out)
        out = torch.zeros(128)
        out[max_index] = 1
        if(target == 127):
            break

    loss.backward()

    encoder_optim.step()
    decoder_optim.step()

    return loss.item()

def train(dataset, encoder, decoder, learning_rate = 0.01, n_iters = 1000):
    # encoder = EncoderRNN(cell_type='lstm', embedding_size=32, hidden_size=32, num_layers=1, bidir= True)
    # decoder = DecoderRNN(cell_type='lstm', embedding_size= 32, hidden_size= 32, num_layers= 1, bidir=True)


    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()

    loss_seq = []

    for i in tqdm(range(n_iters)):
        x, y = next(iter(dataset))
        loss = utils.train_one_iter(x, y, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

        loss_seq.append(loss)

    
    return encoder, decoder, loss_seq

def predict(x, encoder, decoder):
    encoder.initialize_hidden()
    for i in range(50):
        out, hidden = encoder.forward(x[i])
        if(x[i][127] == 1):
            break

    sos = torch.zeros(128)
    sos[0] = 1
    output = torch.zeros(50, 128, dtype=torch.int32)
    out_char = sos

    loss = 0

    for i in range(50):
        out_char, hidden = decoder.forward(out_char, hidden)


        max_index = torch.argmax(out_char)
        out_char = torch.zeros(128)
        out_char[max_index] = 1
        output[i] = out_char
        one_pos = torch.tensor([torch.where(out_char == 1)[0]])
        if(one_pos == 127):
            break
    
    return output