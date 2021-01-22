"""Thompson Sampling with linear posterior over a learnt deep representation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
np.seterr(all='warn')

import warnings

from scipy.stats import invgamma

from .bandit_algorithm import BanditAlgorithm
from .contextual_dataset import ContextualDataset
from .neural_bandit_model import NeuralBanditModel
import torch

import multiprocessing
import os
import pickle
import shutil

#These functions help with multiprocessing random number generation.
mus = None
covs = None

def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def get_mn(i):
    """helper function to parallelize random number generation"""
    mu = mus[i]
    cov = covs[i]
    min_eig = np.min(np.real(np.linalg.eigvals(cov)))
    if min_eig < 0:
        cov -= 10*min_eig * np.eye(*cov.shape)
    return np.random.multivariate_normal(mu, cov)

def parallelize_multivar(mus, covs, n_threads=-1):
    """parallelizes mn computation"""
    if n_threads == -1:
        try:
            cpus = multiprocessing.cpu_count() - 1
        except NotImplementedError:
            cpus = 2   # arbitrary default
    else:
        cpus = n_threads

    with multiprocessing.Pool(processes=cpus) as pool:
        samples = pool.map(get_mn, range(len(mus)))
    return samples


class NeuralBandits(BanditAlgorithm):
    """Full Bayesian linear regression on the last layer of a deep neural net."""

    def __init__(
            self,
            num_actions,
            num_features,
            name='neural_model',
            do_scaling=True,
            init_scale=0.3,
            activation='relu',
            verbose=True,
            optimizer='RMS',
            num_user=200,
            embed_dim=64,
            output_size_wide=1,
            layer_sizes_deep=[50],
            batch_size=512,
            activate_decay=True,
            initial_lr=0.1,
            max_grad_norm=5.0,
            show_training=False,
            freq_summary=1000,
            buffer_s=-1,
            initial_pulls=100,
            reset_lr=True,
            lr_decay_rate=0.5,
            training_freq=1,
            training_freq_network=50,
            training_epochs=100,
            memory_size=-1,
            a0=6,
            b0=6,
            lambda_prior=0.25,
        ):
        """
        A "NeuralLinear" Deep Bayesian contextual bandits model.

            num_actions (int): the number of actions in problem

            num_features (int): the length of context vector, a.k.a. the number of features

            name (string): name for this model instance (default 'neural_model')

            do_scaling (bool): whether to automatically scale features (default True)

            init_scale (float): variance for neural network weights initialization (default 0.3)

            activation (string): activation function for neural network layers (default 'relu')

            verbose (bool): whether to print reports on training steps (default True)

            layer_sizes (list of integers): defines neural network architecture: n_layers = len(layer_sizes), value is per-layer width. (default [50])

            batch_size (integer): batch size for neural network training (default 512)

            activate_decay (bool): whether to use learning rate decay (default True),

            initial_lr (float): initial learning rate for neural network training (default 0.1)

            max_grad_norm (float): maximum gradient value for gradient clipping (default 5.0)

            show_training (bool): whether to show details of neural network training

            freq_summary (int): summary output frequency in number of steps (default 1000)

            buffer_s (int): buffer size for retained examples (default -1)

            initial_pulls (int): number of random pulls before greedy behavior (default 100),

            reset_lr (bool) = whether to reset learning rate on each nn training (default True),

            lr_decay_rate (float): learning rate decay for nn updates (default 0.5)

            training_freq (int): frequency for updates to bayesian linear regressor (default 1)

            training_freq_network (int): frequency of neural network re-trainings (default 50)

            training_epochs (int): number of epochs in each neural network re-training (default 100)

            a0 (int): initial alpha value (default 6)

            b0 (int): initial beta_0 value (default 6)

            lambda_prior (float): lambda prior parameter(default 0.25)
        """
        self.hparams = {
            'num_actions':num_actions,
            'context_dim':num_features,
            'name':name,
            'init_scale':init_scale,
            'activation':activation,
            'verbose':verbose,
            'optimizer':optimizer,
            'num_user':num_user,
            'embed_dim':embed_dim,
            'output_size_wide':output_size_wide,
            'layer_sizes_deep':layer_sizes_deep,
            'batch_size':batch_size,
            'activate_decay':activate_decay,
            'initial_lr':initial_lr,
            'max_grad_norm':max_grad_norm,
            'show_training':show_training,
            'freq_summary':freq_summary,
            'buffer_s':buffer_s,
            'initial_pulls':initial_pulls,
            'reset_lr':reset_lr,
            'lr_decay_rate':lr_decay_rate,
            'training_freq':training_freq,
            'training_freq_network':training_freq_network,
            'training_epochs':training_epochs,
            'memory_size':memory_size,
            'a0':a0,
            'b0':b0,
            'lambda_prior':lambda_prior
        }
        self.do_scaling = do_scaling
        self.name = name
        self.latent_dim = self.hparams['layer_sizes_deep'][-1] + self.hparams['output_size_wide']
        self.num_actions = self.hparams['num_actions']
        self.context_dim = self.hparams['context_dim']
        self.initial_pulls = self.hparams['initial_pulls']
        self.reset_lr = self.hparams['reset_lr']

        self.master_params = dict()

        # Gaussian prior for each beta_i
        self._lambda_prior = self.hparams['lambda_prior']

        self.mu = [
            np.zeros(self.latent_dim)
            for _ in range(self.hparams['num_actions'])
        ]

        self.cov = [(1.0 / self.lambda_prior) * np.eye(self.latent_dim)
                    for _ in range(self.hparams['num_actions'])]

        self.precision = [
            self.lambda_prior * np.eye(self.latent_dim)
            for _ in range(self.hparams['num_actions'])
        ]

        # Inverse Gamma prior for each sigma2_i
        self._a0 = self.hparams['a0']
        self._b0 = self.hparams['b0']

        self.a = [self._a0 for _ in range(self.hparams['num_actions'])]
        self.b = [self._b0 for _ in range(self.hparams['num_actions'])]

        # Regression and NN Update Frequency
        self.update_freq_lr = self.hparams['training_freq']
        self.update_freq_nn = self.hparams['training_freq_network']

        self.t = 0
        self.optimizer_n = optimizer

        self.num_epochs = self.hparams['training_epochs']

        memory_size = self.hparams['memory_size']

        self.data_h = ContextualDataset(
                            self.hparams['context_dim'],
                            self.hparams['num_actions'],
                            intercept=False,
                            memory_size=memory_size
        )

        self.latent_h = ContextualDataset(
                            self.latent_dim,
                            self.hparams['num_actions'],
                            intercept=False,
                            memory_size=memory_size
        )

        self.bnn = NeuralBanditModel(optimizer, self.hparams, f'{name}-bnn')

    def get_representation(self, user_index, context):
        """
        Returns the latent feature vector from the neural network.
        This vector is called z in the Google Brain paper.
        """
        if not isinstance(context, torch.Tensor):
            c = torch.tensor(context).float()
        else:
            c = context.float()
        if not isinstance(user_index, torch.Tensor):
            u = torch.tensor(user_index).long()
        else:
            u = user_index.long()

        z = self.bnn.get_representation(u, c)
        return z

    def expected_values(self, user_index, context, scale=False):
        """
        Computes expected values from context. Does not consider uncertainty.
        Args:
          context: Context for which the action need to be chosen.
        Returns:
          expected reward vector.
        """
        if scale:
            context = self.data_h.scale_contexts(contexts=context)

        # Compute last-layer representation for the current context
        z_context = self.get_representation(user_index, context).numpy()

        # Compute sampled expected values, intercept is last component of beta
        vals = [
            np.dot(self.mu[i], z_context.T)
            for i in range(self.hparams['num_actions'])
        ]
        return np.array(vals)

    def _sample(self, user_index, context, parallelize=False, n_threads=-1):
        # Sample sigma2, and beta conditional on sigma2
        n_rows = len(context)
        d = self.mu[0].shape[0]
        a_projected = np.repeat(np.array(self.a)[np.newaxis, :], n_rows, axis=0)
        sigma2_s = self.b * invgamma.rvs(a_projected)
        if n_rows == 1:
            sigma2_s = sigma2_s.reshape(1, -1)
        beta_s = []
        try:
            for i in range(self.hparams['num_actions']):
                global mus
                global covs
                mus = np.repeat(self.mu[i][np.newaxis, :], n_rows, axis=0)
                s2s = sigma2_s[:, i]
                rep = np.repeat(s2s[:, np.newaxis], d, axis=1)
                rep = np.repeat(rep[:, :, np.newaxis], d, axis=2)
                covs = np.repeat(self.cov[i][np.newaxis, :, :], n_rows, axis=0)
                covs = rep * covs
                if parallelize:
                    multivariates = parallelize_multivar(mus, covs, n_threads=n_threads)
                else:
                    multivariates = [np.random.multivariate_normal(mus[j], covs[j]) for j in range(n_rows)]
                beta_s.append(multivariates)
        except np.linalg.LinAlgError as e:
            # Sampling could fail if covariance is not positive definite
            # Todo: Fix This
            print('Exception when sampling from {}.'.format(self.name))
            print('Details: {} | {}.'.format(e.message, e.args))
            for i in range(self.hparams['num_actions']):
                multivariates = [np.random.multivariate_normal(np.zeros((d)), np.eye(d)) for j in range(n_rows)]
                beta_s.append(multivariates)
        beta_s = np.array(beta_s)

        # Compute last-layer representation for the current context
        z_context = self.get_representation(user_index, context).numpy()

        # Apply Thompson Sampling
        vals = [
            (beta_s[i, :, :] * z_context).sum(axis=-1)
            for i in range(self.hparams['num_actions'])
        ]
        return np.array(vals)

    def action(self, user_index, context):
        """Samples beta's from posterior, and chooses best action accordingly."""
        # Round robin until each action has been selected "initial_pulls" times
        if self.t < self.num_actions * self.initial_pulls:
            return self.t % self.num_actions
        else:
            context = context.reshape(-1, self.hparams['context_dim'])
            if self.do_scaling:
                context = self.data_h.scale_contexts(contexts=context)
            vals = self._sample(user_index, context)
        return np.argmax(vals)

    def update(self, user_index, context, action, reward):
        """Updates the posterior using linear bayesian regression formula."""
        self.t += 1
        self.data_h.add(user_index, context, action, reward)
        c = context.reshape((1, self.context_dim))
        z_context = self.get_representation(user_index, c)
        self.latent_h.add(user_index, z_context, action, reward)
        # Retrain the network on the original data (data_h)
        if self.t % self.update_freq_nn == 0:
            self._retrain_nn()

        if self.t % self.update_freq_lr == 0:
            self._update_actions()

    def _retrain_nn(self):
        """Retrain the network on original data (data_h)"""
        if self.reset_lr:
            self.bnn.assign_lr()

        #uncomment following lines for better stuff.
        #steps = round(self.num_epochs * (len(self.data_h)/self.hparams['batch_size']))
        #print(f"training for {steps} steps.")
        self.bnn.train(self.data_h, self.num_epochs)
        self._replace_latent_h()

    def _replace_latent_h(self):
        # Update the latent representation of every datapoint collected so far
        if self.do_scaling:
            self.data_h.scale_contexts()
        ctx = self.data_h.get_contexts(scaled=self.do_scaling)
        user_indices = self.data_h.user_indices.copy()

        new_z = self.get_representation(user_indices, ctx)

        self.latent_h._replace_data(user_indices=user_indices, contexts=new_z)

    def _update_actions(self):
        """
        Update the Bayesian Linear Regression on
        stored latent variables.
        """

        # Find all the actions to update
        actions_to_update = self.latent_h.actions

        for action_v in np.unique(actions_to_update):

            # Update action posterior with formulas: \beta | z,y ~ N(mu_q, cov_q)
            user_indices, z, y = self.latent_h.get_data(action_v)
            user_indices = user_indices.numpy()
            z = z.numpy()
            y = y.numpy()

            # The algorithm could be improved with sequential formulas (cheaper)
            s = np.dot(z.T, z)

            # Some terms are removed as we assume prior mu_0 = 0.
            precision_a = s + self.lambda_prior * np.eye(self.latent_dim)
            cov_a = np.linalg.inv(precision_a)
            mu_a = np.dot(cov_a, np.dot(z.T, y))

            # Inverse Gamma posterior update
            a_post = self.a0 + z.shape[0] / 2.0
            b_upd = 0.5 * np.dot(y.T, y)
            b_upd -= 0.5 * np.dot(mu_a.T, np.dot(precision_a, mu_a))
            b_post = self.b0 + b_upd

            # Store new posterior distributions
            self.mu[action_v] = mu_a
            self.cov[action_v] = cov_a
            self.precision[action_v] = precision_a
            self.a[action_v] = a_post
            self.b[action_v] = b_post

    def fit(self, user_indices, contexts, actions, rewards):
        """Inputs bulk data for training.
        Args:
          contexts: Set of observed contexts.
          actions: Corresponding list of actions.
          rewards: Corresponding list of rewards.
        """
        data_length = len(rewards)
        self.data_h._ingest_data(user_indices, contexts, actions, rewards)
        #create latent representations of data
        self._replace_latent_h()
        self.latent_h.actions = self.data_h.actions.copy()
        self.latent_h.rewards = self.data_h.rewards
        #update count
        self.t += data_length
        #update posterior on ingested data
        self._retrain_nn()
        self._update_actions()


    def predict(self, user_indices, contexts, thompson=True, parallelize=True, n_threads=-1):
        """Takes a list or array-like of contexts and batch predicts on them"""
        contexts = contexts.reshape(-1, self.hparams['context_dim'])
        if self.do_scaling:
            contexts = self.data_h.scale_contexts(contexts=contexts)

        if thompson:
            reward_matrix = self._sample(contexts, parallelize=parallelize, n_threads=n_threads)
        else:
            reward_matrix = self.expected_values(user_indices, contexts)
        return np.argmax(reward_matrix, axis=0)

    def save(self, path):
        """saves model to path"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @property
    def a0(self):
        return self._a0

    @property
    def b0(self):
        return self._b0

    @property
    def lambda_prior(self):
        return self._lambda_prior
