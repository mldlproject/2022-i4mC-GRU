# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class RnnClassifier(nn.Module):
    def __init__(self, device, inputs_shape, num_layers_gru, vocab_size, embed_dim, rnn_hidden_dim, bidirectional, dropout_fc):
        super(RnnClassifier, self).__init__()
        np.random.seed(1)
        random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self.num_layers_gru = num_layers_gru
        self.vocab_size     = vocab_size
        self.embed_dim      = embed_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.bidirectional  = bidirectional
        self.dropout_fc     = dropout_fc
        self.device         = device

        # Embedding layer
        self.word_embeddings = nn.Embedding(self.vocab_size+1, self.embed_dim)

        # Calculate number of directions
        self.num_directions = 2 if self.bidirectional == True else 1
        self.rnn = nn.GRU(self.embed_dim,
                          self.rnn_hidden_dim,
                          num_layers= num_layers_gru,
                          bias = True,
                          bidirectional=self.bidirectional)

        x = torch.ones(inputs_shape, dtype= torch.long)
        self.flatten  = nn.Flatten()        
        self.linear   = nn.Linear(self.shape_data(x), 128)
        self.active   = nn.LeakyReLU()
        self.drop     = nn.Dropout(self.dropout_fc)

        self.linear_out = nn.Linear(128, 1)
        self.sigmoid    = nn.Sigmoid()

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers_gru * self.num_directions, batch_size, self.rnn_hidden_dim).to(self.device)
                    
    def shape_data(self, inputs):
        batch_size, seq_len = inputs.shape
        # Push through embedding layer 
        X = self.word_embeddings(inputs).permute(1, 0, 2)
        # Push through RNN layer
        self.rnn.flatten_parameters()
        rnn_output, self.hidden = self.rnn(X) 
        output = self.flatten(rnn_output.permute(1, 0, 2))
        return output.shape[1]

    def forward(self, inputs):
        batch_size, seq_len = inputs.shape
        # Push through embedding layer
        embed_out = self.word_embeddings(inputs).permute(1, 0, 2)
        # Push through RNN layer
        rnn_output, h_state = self.rnn(embed_out, self.init_hidden(batch_size))

        rnn_flatten = self.flatten(rnn_output.permute(1, 0, 2))
        fc1_out = self.linear(rnn_flatten)
        fc1_out = self.active(fc1_out)
        fc1_out = self.drop(fc1_out)

        fc2_out = self.linear_out(fc1_out)
        output  = self.sigmoid(fc2_out)
        return output