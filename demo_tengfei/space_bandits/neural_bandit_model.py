"""Define a family of neural network architectures for bandits.
The network accepts different type of optimizers that could lead to different
approximations of the posterior distribution or simply to point estimates.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
from torch.functional import unique
import torch.nn.functional as F

def build_action_mask(actions, num_actions):
    """
    Takes a torch tensor with integer values.
    Returns a one-hot encoded version, where
        each column corresponds to a single action.
    """
    ohe = torch.zeros((len(actions), num_actions))
    actions = actions.reshape(-1, 1)
    return ohe.scatter_(1, actions, 1)

def build_target(rewards, actions, num_actions):
    """
    Takes a torch tensor with floating point values.
    Returns a one-hot encoded version, where
        each column corresponds to a single action.
        The value is the observed reward.
    """
    ohe = torch.zeros((len(actions), num_actions))
    actions = actions.reshape(-1, 1)
    return ohe.scatter_(1, actions, rewards)


class NeuralBanditModel(nn.Module):
    """Implements a neural network for bandit problems."""

    def __init__(self, optimizer, hparams, name):
        """Saves hyper-params and builds a torch NN."""
        super(NeuralBanditModel, self).__init__()
        self.opt_name = optimizer
        self.name = name
        self.hparams = hparams
        self.verbose = self.hparams["verbose"]
        self.times_trained = 0
        self.lr = self.hparams["initial_lr"]
        if self.hparams['activation'] == 'relu':
            self.activation = F.relu
        else:
            act = self.hparams['activation']
            msg = f'activation {act} not implimented'
            raise Exception(msg)
        self.build_model()
        self.optim = self.select_optimizer()
        self.loss = nn.modules.loss.MSELoss()

    def build_layer(self, inp_dim, out_dim):
        """Builds a layer with input x; dropout and layer norm if specified."""

        init_s = self.hparams.get('init_scale', 0.3)
        #these features not currently implemented
        layer_n = self.hparams.get("layer_norm", False)
        dropout = self.hparams.get("use_dropout", False)

        layer = nn.modules.linear.Linear(
            inp_dim,
            out_dim
        )
        nn.init.uniform_(layer.weight, a=-init_s, b=init_s)

        return layer

    def forward(self, user_index, context):
        """forward pass of the Wide and Deep model"""

        wide_out = self.activation(self.wide(self.embedding(user_index)))

        x = context
        for layer in self.deep_layers:
            x = self.activation(layer(x))

        out = self.output_layer(torch.cat((wide_out, x), dim=1))
        return out

    def build_model(self):
        """
        Defines the Wide and Deep model.
        """
        num_user = self.hparams['num_user']
        embed_dim = self.hparams['embed_dim']
        output_size_wide = self.hparams['output_size_wide']
        
        self.embedding = nn.Embedding(num_user, embed_dim)
        self.wide = self.build_layer(embed_dim, output_size_wide) 
        
        self.deep_layers = []
        layer_sizes_deep = self.hparams['layer_sizes_deep']
        context_dim = self.hparams['context_dim']
        
        for i in range(len(layer_sizes_deep)):
            if i==0:
                inp_dim = context_dim
            else:
                inp_dim = layer_sizes_deep[i-1]
            out_dim = layer_sizes_deep[i]
            new_layer = self.build_layer(inp_dim, out_dim)
            name = f'deep layer {len(self.deep_layers)}'
            self.add_module(name, new_layer)
            self.deep_layers.append(new_layer)
        
        self.output_layer = nn.Linear(output_size_wide + layer_sizes_deep[-1], self.hparams['num_actions'])

    def assign_lr(self, lr=None):
        """
        Resets the learning rate to input argument value "lr".
        """
        if lr is None:
            lr = self.lr
        for param_group in self.optim.param_groups:
            param_group['lr'] = lr

    def select_optimizer(self):
        """Selects optimizer. To be extended (SGLD, KFAC, etc)."""
        lr = self.hparams['initial_lr']
        return torch.optim.RMSprop(self.parameters(), lr=lr)

    def scale_weights(self):
        init_s = self.hparams.get('init_scale', 0.3)
        for layer in self.layers:
            nn.init.uniform_(layer.weight, a=-init_s, b=init_s)

    def do_step(self, user_indices, x, y, w, step):

        decay_rate = self.hparams.get('lr_decay_rate', 0.5)
        base_lr = self.hparams.get('initial_lr', 0.1)

        lr = base_lr * (1 / (1 + (decay_rate * step)))
        self.assign_lr(lr)

        y_hat = self.forward(user_indices, x.float())
        y_hat *= w
        ls = self.loss(y_hat, y.float())
        ls.backward()
        clip = self.hparams['max_grad_norm']
        torch.nn.utils.clip_grad_norm_(self.parameters(), clip)

        self.optim.step()
        self.optim.zero_grad()

    def train(self, data, num_steps):
        """Trains the network for num_steps, using the provided data.
        Args:
          data: ContextualDataset object that provides the data.
          num_steps: Number of minibatches to train the network for.
        """

        if self.verbose:
            print("Training {} for {} steps...".format(self.name, num_steps))

        batch_size = self.hparams.get('batch_size', 512)

        data.scale_contexts()
        for step in range(num_steps):
            user_indices, x, y, w = data.get_batch_with_weights(batch_size, scaled=True)
            self.do_step(user_indices, x, y, w, step)

    def get_representation(self, user_indices, contexts):
        """
        Given input contexts, returns representation
        "z" vector.
        """
        if len(user_indices.shape) == 0:
            u = user_indices.unsqueeze(0)
        else:
            u = user_indices
        if len(contexts.shape) == 1:
            c = contexts.unsqueeze(0)
        else:
            c = contexts

        with torch.no_grad():
            wide_out = self.activation(self.wide(self.embedding(u)))
            x = c
            for layer in self.deep_layers:
                x = self.activation(layer(x))
        z = torch.cat((wide_out, x), dim=1)
        return z
