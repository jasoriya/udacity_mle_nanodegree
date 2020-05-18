import torch
import torch.nn as nn
import torch.nn.functional as F

## TODO: Complete this classifier
class SimpleNet(nn.Module):
    
    ## TODO: Define the init function
    def __init__(self, input_dim, hidden_dim, output_dim):
        '''Defines layers of a neural network.
           :param input_dim: Number of input features
           :param hidden_dim: Size of hidden layer(s)
           :param output_dim: Number of outputs
         '''
        super(SimpleNet, self).__init__()
        
        # linear layer (input_dim -> hidden_dim)
        self.input_linear = torch.nn.Linear(input_dim, hidden_dim)
        # linear layer (hidden_dim -> hidden_dim)
        self.middle_linear = torch.nn.Linear(hidden_dim, hidden_dim)
        # linear layer (hidden_dim -> output_dim)
        self.output_linear = torch.nn.Linear(hidden_dim, output_dim)
        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
        self.dropout = nn.Dropout(0.2)
    
    ## TODO: Define the feedforward behavior of the network
    def forward(self, x):
        '''Feedforward behavior of the net.
           :param x: A batch of input features
           :return: A single, sigmoid activated value
         '''
        
        # add input layer with relu activation
        x = F.relu(self.input_linear(x))
        # add hidden layer with relu activation
        x = F.relu(self.middle_linear(x))
        # add dropout layer
        x = self.dropout(x)
        # add output layer
        x = self.output_linear(x)
        
        return x