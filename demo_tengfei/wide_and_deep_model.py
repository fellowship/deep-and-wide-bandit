import torch
import torch.nn as nn
from typing import OrderedDict

class Wide_and_Deep(nn.Module):
    def __init__(
        self, 
        context_dim, 
        action_dim, 
        num_users, 
        embed_dim=64, 
        wide_out_dim=1, 
        deep_neurons=[128, 64, 32], 
        activation=nn.ReLU(),
        dropout=0.2, 
        initial_weights=False):
        '''
        Initialize a wide and deep neural network.

        Arguments:
        =====
        context_dim(int):               size of a context, except the user index
        action_dim(int):                size of action space
        num_users(int):                 number of unique users to define the input of embedding layer
        embed_dim(int):                 embedding dimension (default: 64)
        wide_out_dim(int):              output size of wide model (default: 1)
        deep_neurons(list of ints):     a list of layer sizes of deep model (default: [128, 64, 32])
        activation(nn.Module):          an activation module (default: nn.ReLU())
        '''
        super(Wide_and_Deep, self).__init__()

        self.context_dim = context_dim
        # define wide model
        self.wide = nn.Sequential(OrderedDict([
                      ('embedding', nn.Embedding(num_users, embed_dim)),
                      ('fc', nn.Linear(embed_dim, wide_out_dim)),
                      ('activation', activation)
                    ]))
        # define deep model
        deep_dict = OrderedDict([])
        in_features = context_dim
        for i, out_features in enumerate(deep_neurons):
            deep_dict[f"fc{i}"] = nn.Linear(in_features, out_features)
            deep_dict[f"activation{i}"] = activation
            in_features = out_features

        self.deep = nn.Sequential(deep_dict)
        # define the final layer
        self.lastlayer = nn.Linear(wide_out_dim+in_features, action_dim)

    def forward(self, user_ids, contexts):
        '''
        Perform a whole forwad pass through the whole Wide_and_Deep model using user indices and contexts.

        Arguments:
        ====
        user_ids(int/list of ints):     user indices mapped from the RIID
        contexts(array of floats):      contexts related the users
        '''
        # get the representation
        representation = self.get_representation(user_ids, contexts)
        # pass it through the final layer
        out = self.lastlayer(representation)
        
        return out

    def get_representation(self, user_ids, contexts):
        '''
        Perform forwad passes through the wide and deep models using user indices and contexts seperately.
        Return the concatenated outputs BEFORE the last layer.

        Arguments:
        ====
        user_ids(int/list of ints):     user indices mapped from the RIID
        contexts(array of floats):      contexts related the users
        '''
        # convert the user_ids to a 1-dimention LongTensor
        if not isinstance(user_ids, torch.Tensor):
            u = torch.tensor(user_ids).long()
        else:
            u = user_ids.long()
        u = torch.reshape(u, (-1,))
        # convert the contexts to a 2-dimension FloatTensor
        if not isinstance(contexts, torch.Tensor):
            c = torch.tensor(contexts).float()
        else:
            c = contexts.float()
        c = torch.reshape(c, (-1, self.context_dim))

        assert u.shape[0] == c.shape[0]

        # pass u through wide model
        wide_out = self.wide(u)
        # pass c trhough deep model
        deep_out = self.deep(c)
        # conbine the outputs of wide model and deep model
        representation = torch.cat((wide_out, deep_out), dim=1)
        
        return representation
