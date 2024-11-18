import numpy as np

import torch
import torch.optim as optim

from mushroom_rl.algorithms.actor_critic.deep_actor_critic import DeepAC
from mushroom_rl.policy import Policy
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.replay_memory import ReplayMemory
from mushroom_rl.utils.torch import to_float_tensor
from mushroom_rl.utils.parameters import to_parameter

from moore.utils.networks_sac import count_parameters

from copy import deepcopy
from itertools import chain

import os

class SACPolicy(Policy):
    """
    Class used to implement the policy used by the Soft Actor-Critic
    algorithm. The policy is a Gaussian policy squashed by a tanh.
    This class implements the compute_action_and_log_prob and the
    compute_action_and_log_prob_t methods, that are fundamental for
    the internals calculations of the SAC algorithm.
    
    This modified version of the SACPolicy supports the following:
    - Predicting the mu and sigma with a joint/ disjoint model(s).
    - A function that get the shared parameters of the policy model.
    """
    def __init__(self, approximator, min_a, max_a, log_std_min, log_std_max):
        """
        Constructor.

        Args:
            mu_approximator (Regressor): a regressor computing mean in given a
                state;
            sigma_approximator (Regressor): a regressor computing the variance
                in given a state;
            min_a (np.ndarray): a vector specifying the minimum action value
                for each component;
            max_a (np.ndarray): a vector specifying the maximum action value
                for each component.
            log_std_min ([float, Parameter]): min value for the policy log std;
            log_std_max ([float, Parameter]): max value for the policy log std.

        """
        # NOTE: handling the shared_mu_sigma architecture
        self._shared_mu_sigma = not isinstance(approximator, tuple)

        if self._shared_mu_sigma:
            self._approximator = approximator
        else:
            self._mu_approximator = approximator[0]
            self._sigma_approximator = approximator[1]


        self._delta_a = to_float_tensor(.5 * (max_a - min_a), self.use_cuda)
        self._central_a = to_float_tensor(.5 * (max_a + min_a), self.use_cuda)

        self._log_std_min = to_parameter(log_std_min)
        self._log_std_max = to_parameter(log_std_max)

        self._eps_log_prob = 1e-6

        if self.use_cuda:
            self._delta_a = self._delta_a.cuda()
            self._central_a = self._central_a.cuda()

        # NOTE: handling the shared_mu_sigma architecture
        if not self._shared_mu_sigma:
            self._add_save_attr(
                _mu_approximator='mushroom',
                _sigma_approximator='mushroom',
                _delta_a='torch',
                _central_a='torch',
                _log_std_min='mushroom',
                _log_std_max='mushroom',
                _eps_log_prob='primitive',
                _shared_mu_sigma='primitive',
            )
        else:
            self._add_save_attr(
                _approximator='mushroom',
                _delta_a='torch',
                _central_a='torch',
                _log_std_min='mushroom',
                _log_std_max='mushroom',
                _eps_log_prob='primitive',
                _shared_mu_sigma='primitive',
            )

    def __call__(self, state, action):
        raise NotImplementedError

    def draw_action(self, state):
        return self.compute_action_and_log_prob_t(
            state, compute_log_prob=False).detach().cpu().numpy()

    def compute_action_and_log_prob(self, state):
        """
        Function that samples actions using the reparametrization trick and
        the log probability for such actions.

        Args:
            state (np.ndarray): the state in which the action is sampled.

        Returns:
            The actions sampled and the log probability as numpy arrays.

        """
        a, log_prob = self.compute_action_and_log_prob_t(state)
        return a.detach().cpu().numpy(), log_prob.detach().cpu().numpy()

    def compute_action_and_log_prob_t(self, state, compute_log_prob=True):
        """
        Function that samples actions using the reparametrization trick and,
        optionally, the log probability for such actions.

        Args:
            state (np.ndarray): the state in which the action is sampled;
            compute_log_prob (bool, True): whether to compute the log
            probability or not.

        Returns:
            The actions sampled and, optionally, the log probability as torch
            tensors.

        """
        dist = self.distribution(state)
        a_raw = dist.rsample()
        a = torch.tanh(a_raw)
        a_true = a * self._delta_a + self._central_a

        if compute_log_prob:
            log_prob = dist.log_prob(a_raw).sum(dim=1)
            log_prob -= torch.log(1. - a.pow(2) + self._eps_log_prob).sum(dim=1)
      
            return a_true, log_prob
        else:
            return a_true

    def distribution(self, state):
        """
        Compute the policy distribution in the given states.

        Args:
            state (np.ndarray): the set of states where the distribution is
                computed.

        Returns:
            The torch distribution for the provided states.

        """
        # modified for sharing mu and sigma
        if self._shared_mu_sigma:
            a = self._approximator.predict(state, output_tensor=True)
            if a.ndim == 1:
                mu, log_sigma = a[:a.shape[-1]//2], a[a.shape[-1]//2:]
            else:
                mu, log_sigma = a[:, :a.shape[-1]//2], a[:, a.shape[-1]//2:]
        else:
            mu = self._mu_approximator.predict(state, output_tensor=True)
            log_sigma = self._sigma_approximator.predict(state, output_tensor=True)
        # Bound the log_std
        log_sigma = torch.clamp(log_sigma, self._log_std_min(), self._log_std_max())
        
        return torch.distributions.Normal(mu, log_sigma.exp())

    def entropy(self, state=None):
        """
        Compute the entropy of the policy.

        Args:
            state (np.ndarray): the set of states to consider.

        Returns:
            The value of the entropy of the policy.

        """

        return torch.mean(self.distribution(state).entropy()).detach().cpu().numpy().item()

    def reset(self):
        pass

    def set_weights(self, weights):
        """
        Setter.

        Args:
            weights (np.ndarray): the vector of the new weights to be used by
                the policy.

        """
        # NOTE: handling the shared_mu_sigma architecture
        if self._shared_mu_sigma:
            self._approximator.set_weights(weights)
        else:
            mu_weights = weights[:self._mu_approximator.weights_size]
            sigma_weights = weights[self._mu_approximator.weights_size:]

            self._mu_approximator.set_weights(mu_weights)
            self._sigma_approximator.set_weights(sigma_weights)

    def get_weights(self):
        """
        Getter.

        Returns:
             The current policy weights.

        """
        # NOTE: handling the shared_mu_sigma architecture
        if self._shared_mu_sigma:
            return self._approximator.get_weights()
        
        mu_weights = self._mu_approximator.get_weights()
        sigma_weights = self._sigma_approximator.get_weights()

        return np.concatenate([mu_weights, sigma_weights])

    @property
    def use_cuda(self):
        """
        True if the policy is using cuda_tensors.
        """
        # NOTE: handling the shared_mu_sigma architecture
        return self._mu_approximator.model.use_cuda if not self._shared_mu_sigma else self._approximator.model.use_cuda

    def parameters(self):
        """
        Returns the trainable policy parameters, as expected by torch
        optimizers.

        Returns:
            List of parameters to be optimized.

        """
        # NOTE: handling the shared_mu_sigma architecture
        if self._shared_mu_sigma:
            return chain(self._approximator.model.network.parameters())
        
        return chain(self._mu_approximator.model.network.parameters(),
                     self._sigma_approximator.model.network.parameters())
    
    def get_shared_weights(self):
        # NOTE: only in case of shared_mu_sigma
        return self._approximator.model.network.get_shared_weights()
    
    def get_state_dict(self):
        state_dict = {}

        if self._shared_mu_sigma:
            state_dict["policy_approximator_state_dict"] = self._approximator.model.network.state_dict()
        else:
            state_dict["policy_mu_approximator_state_dict"] = self._mu_approximator.model.network.state_dict()
            state_dict["policy_sigma_approximator_state_dict"] = self._sigma_approximator.model.network.state_dict()
        
        return state_dict

    def load_state_dict(self, state_dict):
        if self._shared_mu_sigma:
            self._approximator.model.network.load_state_dict(state_dict["policy_approximator_state_dict"])
        else:
            self._mu_approximator.model.network.load_state_dict(state_dict["policy_mu_approximator_state_dict"])
            self._sigma_approximator.model.network.load_state_dict(state_dict["policy_sigma_approximator_state_dict"])
    
###########################################################################################################################################################################################
class SAC(DeepAC):
    """
    Soft Actor-Critic algorithm.
    "Soft Actor-Critic Algorithms and Applications".
    Haarnoja T. et al.. 2019.

    This is a modified version of SAC which supports the following:
    - Predicting the mu and log_sigma with a joint/ disjoint model(s).
    - A function which outputs the shared critic parameters.
    - models_summary funtion which reports the summary of all the models in addition to the number of parameters.
    """
    def __init__(self, mdp_info, actor_optimizer, critic_params, batch_size,
                 initial_replay_size, max_replay_size, warmup_transitions, tau, lr_alpha, 
                 actor_params = None, actor_mu_params = None, actor_sigma_params = None, 
                 log_std_min=-20, log_std_max=2, target_entropy=None, log_alpha = None, 
                 critic_fit_params=None, policy_class = SACPolicy, shared_mu_sigma = False):
        """
        Constructor.

        Args:
            actor_mu_params (dict): parameters of the actor mean approximator
                to build;
            actor_sigma_params (dict): parameters of the actor sigm
                approximator to build;
            actor_optimizer (dict): parameters to specify the actor
                optimizer algorithm;
            critic_params (dict): parameters of the critic approximator to
                build;
            batch_size ((int, Parameter)): the number of samples in a batch;
            initial_replay_size (int): the number of samples to collect before
                starting the learning;
            max_replay_size (int): the maximum number of samples in the replay
                memory;
            warmup_transitions ([int, Parameter]): number of samples to accumulate in the
                replay memory to start the policy fitting;
            tau ([float, Parameter]): value of coefficient for soft updates;
            lr_alpha ([float, Parameter]): Learning rate for the entropy coefficient;
            log_std_min ([float, Parameter]): Min value for the policy log std;
            log_std_max ([float, Parameter]): Max value for the policy log std;
            target_entropy (float, None): target entropy for the policy, if
                None a default value is computed ;
            critic_fit_params (dict, None): parameters of the fitting algorithm
                of the critic approximator.
            TODO: add the description for the newly added variables
        """
        self._critic_fit_params = dict() if critic_fit_params is None else critic_fit_params

        self._batch_size = to_parameter(batch_size)
        self._warmup_transitions = to_parameter(warmup_transitions)
        self._tau = to_parameter(tau)
        self._shared_mu_sigma = shared_mu_sigma
        
        if target_entropy is None:
            self._target_entropy = -np.prod(mdp_info.action_space.shape).astype(np.float32)
        else:
            self._target_entropy = target_entropy

        self._replay_memory = ReplayMemory(initial_replay_size, max_replay_size)

        if 'n_models' in critic_params.keys():
            assert critic_params['n_models'] == 2
        else:
            critic_params['n_models'] = 2

        target_critic_params = deepcopy(critic_params)
        self._critic_approximator = Regressor(TorchApproximator,
                                              **critic_params)
        self._target_critic_approximator = Regressor(TorchApproximator,
                                                     **target_critic_params)

        if self._shared_mu_sigma:
            actor_approximator = Regressor(TorchApproximator,
                                            **actor_params)
        else:
            actor_mu_approximator = Regressor(TorchApproximator,
                                            **actor_mu_params)
            actor_sigma_approximator = Regressor(TorchApproximator,
                                                **actor_sigma_params)

        if self._shared_mu_sigma:   
            policy = policy_class(actor_approximator,
                            mdp_info.action_space.low,
                            mdp_info.action_space.high,
                            log_std_min,
                            log_std_max)
        else:
            policy = policy_class((actor_mu_approximator, actor_sigma_approximator),
                            mdp_info.action_space.low,
                            mdp_info.action_space.high,
                            log_std_min,
                            log_std_max)

        self._init_target(self._critic_approximator,
                          self._target_critic_approximator)

        if log_alpha is None:
            self._log_alpha = torch.tensor(0., dtype=torch.float32)
        else:
            self._log_alpha = torch.tensor(log_alpha, dtype=torch.float32)


        if policy.use_cuda:
            self._log_alpha = self._log_alpha.cuda().requires_grad_()
        else:
            self._log_alpha.requires_grad_()

        self._alpha_optim = optim.Adam([self._log_alpha], lr=lr_alpha)

        policy_parameters = policy.parameters()

        self._add_save_attr(
            _critic_fit_params='pickle',
            _batch_size='mushroom',
            _warmup_transitions='mushroom',
            _tau='mushroom',
            _shared_mu_sigma='primitive',
            _target_entropy='primitive',
            _replay_memory='mushroom',
            _critic_approximator='mushroom',
            _target_critic_approximator='mushroom',
            _log_alpha='torch',
            _alpha_optim='torch'
        )

        super().__init__(mdp_info, policy, actor_optimizer, policy_parameters)

    def fit(self, dataset, **info):
        self._replay_memory.add(dataset)
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, _ = \
                self._replay_memory.get(self._batch_size())

            if self._replay_memory.size > self._warmup_transitions():
                action_new, log_prob = self.policy.compute_action_and_log_prob_t(state)
                loss = self._loss(state, action_new, log_prob)
                self._optimize_actor_parameters(loss)
                self._update_alpha(log_prob.detach())

            q_next = self._next_q(next_state, absorbing)
            q = reward + self.mdp_info.gamma * q_next

            self._critic_approximator.fit(state, action, q,
                                          **self._critic_fit_params)

            self._update_target(self._critic_approximator,
                                self._target_critic_approximator)

    def _loss(self, state, action_new, log_prob):
        q_0 = self._critic_approximator(state, action_new,
                                        output_tensor=True, idx=0)
        q_1 = self._critic_approximator(state, action_new,
                                        output_tensor=True, idx=1)

        q = torch.min(q_0, q_1)

        return (self._alpha * log_prob - q).mean()

    def _update_alpha(self, log_prob):
        alpha_loss = - (self._log_alpha * (log_prob + self._target_entropy)).mean()
        self._alpha_optim.zero_grad()
        alpha_loss.backward()
        self._alpha_optim.step()

    def _next_q(self, next_state, absorbing):
        """
        Args:
            next_state (np.ndarray): the states where next action has to be
                evaluated;
            absorbing (np.ndarray): the absorbing flag for the states in
                ``next_state``.

        Returns:
            Action-values returned by the critic for ``next_state`` and the
            action returned by the actor.

        """
        a, log_prob_next = self.policy.compute_action_and_log_prob(next_state)

        q = self._target_critic_approximator.predict(
            next_state, a, prediction='min') - self._alpha_np * log_prob_next
        q *= 1 - absorbing

        return q

    def _post_load(self):
        self._update_optimizer_parameters(self.policy.parameters())

    @property
    def _alpha(self):
        return self._log_alpha.exp()

    @property
    def _alpha_np(self):
        return self._alpha.detach().cpu().numpy()
    
    def get_log_alpha(self):
        '''
            get the value of alpha of context c for logging
        '''
        return self._log_alpha.detach().cpu().numpy()
    
    def models_summary(self):
        if self._logger is not None:
            n_parameters = 0
            if self._shared_mu_sigma:
                self._logger.info("ActorNetwork:")
                _n_parameters = count_parameters(self.policy._approximator.model.network)
                n_parameters+=_n_parameters
                self._logger.info(f"Number of Parameters: {_n_parameters}")
                self._logger.info(self.policy._approximator.model.network)
            else:
                self._logger.info("ActorNetwork Mu:")
                _n_parameters = count_parameters(self.policy._mu_approximator.model.network)
                n_parameters+=_n_parameters
                self._logger.info(f"Number of Parameters: {_n_parameters}")
                self._logger.info(self.policy._mu_approximator.model.network)
                ###########################################################
                self._logger.info("ActorNetwork Sigma:")
                _n_parameters = count_parameters(self.policy._sigma_approximator.model.network)
                n_parameters+=_n_parameters
                self._logger.info(f"Number of Parameters: {_n_parameters}")
                self._logger.info(self.policy._sigma_approximator.model.network)
            ###########################################################
            for i in range(len(self._critic_approximator)):
                self._logger.info(f"CriticNetwork {i+1}:")
                _n_parameters = count_parameters(self._critic_approximator[i].network)
                n_parameters+=_n_parameters
                self._logger.info(f"Number of Parameters: {_n_parameters}")
                self._logger.info(self._critic_approximator[i].network)
            #############################################################
            for i in range(len(self._target_critic_approximator)):
                self._logger.info(f"Target CriticNetwork {i+1}:")
                _n_parameters = count_parameters(self._target_critic_approximator[i].network)
                n_parameters+=_n_parameters
                self._logger.info(f"Number of Parameters: {_n_parameters}")
                self._logger.info(self._target_critic_approximator[i].network)
            #############################################################
            self._logger.info(f"Total Number of Parameters: {n_parameters}")

    def get_critic_weights(self):
        weights = {}
        for i in range(len(self._critic_approximator)):
            weights[f"critic_{i+1}_weights"] = self._critic_approximator[i].get_weights()
        for i in range(len(self._target_critic_approximator)):
            weights[f"target_critic_{i+1}_weights"] = self._target_critic_approximator[i].get_weights()
        return weights

    def set_critic_weights(self, weights_dir):
        for i in range(len(self._critic_approximator)):
            weights = np.load(os.path.join(weights_dir, f"critic_{i+1}_weights.npy"))
            self._critic_approximator[i].set_weights(weights)
        for i in range(len(self._target_critic_approximator)):
            weights = np.load(os.path.join(weights_dir, f"target_critic_{i+1}_weights.npy"))
            self._target_critic_approximator[i].set_weights(weights)

    def get_critic_shared_weights(self):
        i = np.random.randint(0, 2)
        return self._critic_approximator[i].network.get_shared_weights()

    def get_state_dict(self):
        # policy/ actor
        policy_state_dict = self.policy.get_state_dict()
        policy_optimizer_state_dict = self._optimizer.state_dict()

        # alpha
        log_alpha_state_dict = {"log_alpha": self.get_alpha_np(), "log_alpha_optimizer_state_dict": self._alpha_optim.state_dict()}

        # critic
        critic_state_dict = {"critic_0_state_dict": self._critic_approximator[0].network.state_dict(),
                             "critic_1_state_dict": self._critic_approximator[1].network.state_dict(),
                             "critic_0_optimizer_state_dict": self._critic_approximator[0]._optimizer.state_dict(),
                             "critic_1_optimizer_state_dict": self._critic_approximator[1]._optimizer.state_dict(),
                             }
        

        return {**policy_state_dict, **policy_optimizer_state_dict, **log_alpha_state_dict, **critic_state_dict}
    

    def load_state_dict(self, state_dict):
        # policy/ actor
        self.policy.load_state_dict(state_dict)
        # TODO: complete the function (if needed)
