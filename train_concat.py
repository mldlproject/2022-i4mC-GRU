# Import Python libraries
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from utils import *
from torch.utils.data import TensorDataset
from model import *
from training_func import *
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Parameter
type_padded_data = 'no_weight' # w_weight, no_weight
batch_size      = 64
inputs_shape    = (batch_size, 40)
num_layers_gru  = 1
vocab_size      = 40
embed_dim       = 64
rnn_hidden_dim  = 128
bidirectional   = True
dropout_fc      = 0.5
n_epoch         = 2
lr_rate         = 0.0001*2/3

# Model
model_path  = './outputs/saved_model'
model = RnnClassifier(device, inputs_shape, num_layers_gru, vocab_size, embed_dim, rnn_hidden_dim, bidirectional, dropout_fc)
model.load_state_dict(torch.load(model_path + '/model_{}.pt'.format(type_padded_data)))
model = model.cuda()
# Loss function
criteron = nn.BCELoss()
# Adam Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)


# Loading data
data_path   = './process_data/processed_data/encoded_data/encode_padded_data_{}/'.format(type_padded_data)

data_train  = np.load(data_path + '/data_train.npy')
y_train     = np.load(data_path + '/label_train.npy')

data_val    = np.load(data_path + '/data_val.npy')
y_val       = np.load(data_path + '/label_val.npy')

data_test   = np.load(data_path + '/data_test.npy')
y_test      = np.load(data_path + '/label_test.npy')

data_train = np.concatenate((data_train, data_val, data_test), axis=0)
y_train    = np.concatenate((y_train, y_val, y_test), axis=0)

# Shape data 
print('Data Training: ', data_train.shape)
print('Pos: {}, Neg: {}'.format(np.sum(y_train), len(y_train) - np.sum(y_train)))

print('Data Test: ', data_test.shape)
print('Pos: {}, Neg: {}'.format(np.sum(y_test), len(y_test) - np.sum(y_test)))

train_dataset       = TensorDataset(Tensor(data_train).long(), Tensor(y_train))
test_dataset        = TensorDataset(Tensor(data_test).long(), Tensor(y_test))

training_loader     = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, worker_init_fn = np.random.seed(0))
test_loader         = torch.utils.data.DataLoader(test_dataset,  batch_size = batch_size, shuffle = False)

##########################################################################################                 
###                                 Traininig                                          ###           
##########################################################################################
training_loss_list, validation_loss_list, test_loss_list = [], [], []
test_prob_pred, val_prob_pred = [], []

print("Training model for data padded {}".format(type_padded_data))
val_loss_check = 10
for epoch in range(n_epoch):
    #------------------------------------------------
    train_results = train(epoch, model, criteron, optimizer, device, training_loader)
    #------------------------------------------------
    torch.save(model.state_dict(), model_path + '/model_concat_{}.pt'.format(type_padded_data))
    #------------------------------------------------
    test_results = test(epoch, model, criteron, device, test_loader)
    #------------------- 
    training_loss_list.append(train_results)
    #-------------------
    test_loss_list.append(test_results[0])
    test_prob_pred.append(test_results[1])

##########################################################################################                 
###                                 Evaluate                                           ###           
##########################################################################################
print("Performance")
test_probs  = get_prob(test_prob_pred, 1)
matrix      = performance(y_test, test_probs, name= 'test_dataset_{}'.format(type_padded_data))


