import numpy as np

import torch
import torch.nn as nn

from mushroom_rl.policy import Policy
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.torch import to_float_tensor
from mushroom_rl.utils.parameters import to_parameter

from itertools import chain

class TorchPolicy(Policy):
    """
    Interface for a generic PyTorch policy.
    A PyTorch policy is a policy implemented as a neural network using PyTorch.
    Functions ending with '_t' use tensors as input, and also as output when
    required.

    """
    def __init__(self, use_cuda):
        """
        Constructor.

        Args:
            use_cuda (bool): whether to use cuda or not.

        """
        self._use_cuda = use_cuda

        self._add_save_attr(_use_cuda='primitive')

    def __call__(self, state, action):
        s = to_float_tensor(np.atleast_2d(state), self._use_cuda)
        a = to_float_tensor(np.atleast_2d(action), self._use_cuda)

        return np.exp(self.log_prob_t(s, a).item())

    def draw_action(self, state):
        # handling context
        with torch.no_grad():
            c, state = state[0], state[1]
            s = to_float_tensor(np.atleast_2d(state), self._use_cuda)
            a = self.draw_action_t([c, s])

        return torch.squeeze(a, dim=0).detach().cpu().numpy()

    def distribution(self, state):
        """
        Compute the policy distribution in the given states.

        Args:
            state (np.ndarray): the set of states where the distribution is
                computed.

        Returns:
            The torch distribution for the provided states.

        """
        s = to_float_tensor(state, self._use_cuda)

        return self.distribution_t(s)

    def entropy(self, state=None):
        """
        Compute the entropy of the policy.

        Args:
            state (np.ndarray, None): the set of states to consider. If the
                entropy of the policy can be computed in closed form, then
                ``state`` can be None.

        Returns:
            The value of the entropy of the policy.

        """
        s = to_float_tensor(state, self._use_cuda) if state is not None else None

        return self.entropy_t(s).detach().cpu().numpy().item()

    def draw_action_t(self, state):
        """
        Draw an action given a tensor.

        Args:
            state (torch.Tensor): set of states.

        Returns:
            The tensor of the actions to perform in each state.

        """
        raise NotImplementedError

    def log_prob_t(self, state, action):
        """
        Compute the logarithm of the probability of taking ``action`` in
        ``state``.

        Args:
            state (torch.Tensor): set of states.
            action (torch.Tensor): set of actions.

        Returns:
            The tensor of log-probability.

        """
        raise NotImplementedError

    def entropy_t(self, state):
        """
        Compute the entropy of the policy.

        Args:
            state (torch.Tensor): the set of states to consider. If the
                entropy of the policy can be computed in closed form, then
                ``state`` can be None.

        Returns:
            The tensor value of the entropy of the policy.

        """
        raise NotImplementedError

    def distribution_t(self, state):
        """
        Compute the policy distribution in the given states.

        Args:
            state (torch.Tensor): the set of states where the distribution is
                computed.

        Returns:
            The torch distribution for the provided states.

        """
        raise NotImplementedError

    def set_weights(self, weights):
        """
        Setter.

        Args:
            weights (np.ndarray): the vector of the new weights to be used by
                the policy.

        """
        raise NotImplementedError

    def get_weights(self):
        """
        Getter.

        Returns:
             The current policy weights.

        """
        raise NotImplementedError

    def parameters(self):
        """
        Returns the trainable policy parameters, as expected by torch
        optimizers.

        Returns:
            List of parameters to be optimized.

        """
        raise NotImplementedError

    def reset(self):
        pass

    @property
    def use_cuda(self):
        """
        True if the policy is using cuda_tensors.
        """
        return self._use_cuda
    
class MTBoltzmannTorchPolicy(TorchPolicy):
    """
    Torch policy implementing a Multi-Task Boltzmann policy.

    """
    class CategoricalWrapper(torch.distributions.Categorical):
        def __init__(self, logits):
            super().__init__(logits=logits)

        def log_prob(self, value):
            return super().log_prob(value.squeeze())

    def __init__(self, network, input_shape, output_shape, beta, use_cuda=False, **params):
        """
        Constructor.

        Args:
            network (object): the network class used to implement the mean
                regressor;
            input_shape (tuple): the shape of the state space;
            output_shape (tuple): the shape of the action space;
            beta ((float, Parameter)): the inverse of the temperature distribution. As
                the temperature approaches infinity, the policy becomes more and
                more random. As the temperature approaches 0.0, the policy becomes
                more and more greedy.
            params (dict): parameters used by the network constructor.

        """
        super().__init__(use_cuda)

        self._action_dim = output_shape[0]
        self._predict_params = dict()

        self._logits = Regressor(TorchApproximator, input_shape, output_shape,
                                 network=network, use_cuda=use_cuda, **params)
        self._beta = to_parameter(beta)

        self._add_save_attr(
            _action_dim='primitive',
            _predict_params='pickle',
            _beta='mushroom',
            _logits='mushroom',
        )

    def draw_action_t(self, state):
        # state should be a list
        action = self.distribution_t(state).sample().detach()

        if len(action.shape) > 1:
            return action
        else:
            return action.unsqueeze(0)

    def log_prob_t(self, state, action):
        # state should be a list
        return self.distribution_t(state).log_prob(action)[:, None]

    def entropy_t(self, state):
        # state should be a list
        return torch.mean(self.distribution_t(state).entropy())

    def distribution_t(self, state):
        c, state = state[0], state[1]
        logits = self._logits(state, c=c, **self._predict_params, output_tensor=True) * self._beta(state.cpu().numpy()) # add .cpu()
        return MTBoltzmannTorchPolicy.CategoricalWrapper(logits)

    def set_weights(self, weights):
        self._logits.set_weights(weights)

    def get_weights(self):
        return self._logits.get_weights()

    def parameters(self):
        return self._logits.model.network.parameters()

    def set_beta(self, beta):
        self._beta = to_parameter(beta)
