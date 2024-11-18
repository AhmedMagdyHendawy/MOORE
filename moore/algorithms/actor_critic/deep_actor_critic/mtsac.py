import numpy as np
# deeplearning frameworks
import torch
import torch.optim as optim
# framework
from . import SAC, SACPolicy
from moore.utils.replay_memory import ReplayMemory


class MTSACPolicy(SACPolicy):

    def distribution(self, state):
        """
        Compute the policy distribution in the given states.

        Args:
            state (np.ndarray): the set of states where the distribution is
                computed.

        Returns:
            The torch distribution for the provided states.

        """
        idx, state = state[0], state[1]
        # modified for sharing mu and sigma
        if self._shared_mu_sigma:
            a = self._approximator.predict(state, c=idx, output_tensor=True)
            if a.ndim == 1:
                mu, log_sigma = a[:a.shape[-1]//2], a[a.shape[-1]//2:]
            else:
                mu, log_sigma = a[:, :a.shape[-1]//2], a[:, a.shape[-1]//2:]
        else:
            mu = self._mu_approximator.predict(state, c=idx, output_tensor=True)
            log_sigma = self._sigma_approximator.predict(state, c=idx, output_tensor=True)
        # Bound the log_std
        log_sigma = torch.clamp(log_sigma, self._log_std_min(), self._log_std_max())

        return torch.distributions.Normal(mu, log_sigma.exp())


class MTSAC(SAC):
    def __init__(self, mdp_info, actor_optimizer, critic_params, batch_size,
                 initial_replay_size, max_replay_size, warmup_transitions, tau, lr_alpha,
                 actor_params=None, actor_mu_params=None, actor_sigma_params=None, 
                 log_std_min=-20, log_std_max=2, target_entropy=None, log_alpha = None, 
                 critic_fit_params=None, shared_mu_sigma = False, n_contexts = 1):
        
        super().__init__(mdp_info, actor_optimizer, critic_params, batch_size,
                        initial_replay_size, max_replay_size, warmup_transitions, tau, lr_alpha,
                        actor_params=actor_params, actor_mu_params=actor_mu_params, actor_sigma_params=actor_sigma_params,
                        log_std_min=log_std_min, log_std_max=log_std_max, target_entropy=target_entropy, critic_fit_params=critic_fit_params, 
                        policy_class=MTSACPolicy, shared_mu_sigma=shared_mu_sigma)
        
        self._n_contexts = n_contexts
        
        self._replay_memory = [ReplayMemory(initial_replay_size, max_replay_size) for _ in range(n_contexts)]

        if log_alpha is None:
            self._log_alpha = torch.tensor([0.]*n_contexts, dtype=torch.float32)
        else:
            assert len(log_alpha) == n_contexts
            self._log_alpha = torch.tensor(log_alpha, dtype=torch.float32)

        if self.policy.use_cuda:
            self._log_alpha = self._log_alpha.cuda().requires_grad_()
        else:
            self._log_alpha.requires_grad_()

        self._alpha_optim = optim.Adam([self._log_alpha], lr=lr_alpha)
        
        self._add_save_attr(_n_contexts='primitive')
        
    def fit(self, dataset, **info):
        contexts = np.array([d[0][0] for d in dataset]).ravel().astype(np.int64)
        unique_contexts = np.unique(contexts)
        for c in unique_contexts:
            idxs = np.argwhere(contexts == c).ravel()
            d = [dataset[idx] for idx in idxs]
            self._replay_memory[c].add(d)
        
        fit_condition = np.all([rm.initialized for rm in self._replay_memory])

        if fit_condition:
            state_idx = []
            state = []
            action = []
            reward = []
            next_state = []
            absorbing = []

            for i in range(len(self._replay_memory)):
                state_i, action_i, reward_i, next_state_i,\
                    absorbing_i, _ = self._replay_memory[i].get(
                        self._batch_size())
                state_idx.append(np.ones(self._batch_size(), dtype=np.int64) * i)
                state.append(state_i)
                action.append(action_i)
                reward.append(reward_i)
                next_state.append(next_state_i)
                absorbing.append(absorbing_i)

            state_idx = np.vstack(state_idx).reshape(-1)
            state = np.vstack(state)
            action = np.vstack(action)
            reward = np.hstack(reward)
            next_state = np.vstack(next_state)
            absorbing = np.hstack(absorbing)

            if self._replay_memory[0].size > self._warmup_transitions(): #first or any
                action_new, log_prob = self.policy.compute_action_and_log_prob_t([state_idx, state])
                loss = self._loss(state, action_new, state_idx, log_prob)
                self._optimize_actor_parameters(loss)
                self._update_alpha(log_prob.detach(), state_idx)

            q_next = self._next_q(next_state, absorbing, state_idx)
            q = reward + self.mdp_info.gamma * q_next

            self._critic_approximator.fit(state, action, q, c = state_idx,
                                          **self._critic_fit_params)
            # TODO: double check this step
            self._update_target(self._critic_approximator,
                                self._target_critic_approximator)
           
    def _loss(self, state, action_new, state_idx, log_prob):

        q_0 = self._critic_approximator(state, action_new, c = state_idx,
                                        output_tensor=True, idx=0)
        q_1 = self._critic_approximator(state, action_new, c = state_idx,
                                        output_tensor=True, idx=1)

        q = torch.min(q_0, q_1)

        return (self._alpha(state_idx) * log_prob - q).mean()

    def _update_alpha(self, log_prob, state_idx):
        alpha_loss = - (self._log_alpha_disentangled(state_idx) * (log_prob + self._target_entropy)).mean()
        self._alpha_optim.zero_grad()
        alpha_loss.backward()
        self._alpha_optim.step()

        return alpha_loss
    
    def _next_q(self, next_state, absorbing, state_idx):
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

        a, log_prob_next = self.policy.compute_action_and_log_prob([state_idx, next_state])

        q = self._target_critic_approximator.predict(
            next_state, a, c = state_idx, prediction='min') - self._alpha_np(state_idx) * log_prob_next
        q *= 1 - absorbing

        return q
    
    # TODO: improve
    def _log_alpha_disentangled(self, c):
        log_alpha = torch.zeros(size=(c.shape[0],))
        c = c.astype(int)

        if self.policy.use_cuda:
            log_alpha = log_alpha.cuda()

        for _, ci in enumerate(np.unique(c)):
            ci_idx = np.argwhere(c == ci).ravel()
            log_alpha_i = self._log_alpha[ci]
            log_alpha[ci_idx] = log_alpha_i

        return log_alpha
    
    # TODO: improve  
    def _alpha(self, c):
        
        log_alpha = self._log_alpha_disentangled(c)
        return log_alpha.exp()

    def _alpha_np(self, c):
        return self._alpha(c).detach().cpu().numpy()

    def get_log_alpha(self, c):
        '''
            get the value of log_alpha of context c for logging
        '''
        return self._log_alpha[c].detach().cpu().numpy()

    def set_log_alpha(self, log_alpha):
        for c in range(self._n_contexts()):
            self._log_alpha[c] = log_alpha[c]