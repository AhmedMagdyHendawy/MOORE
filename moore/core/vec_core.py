from tqdm import tqdm

from collections import defaultdict

import numpy as np

class VecCore(object):
    def __init__(self, agent, mdp, callbacks_fit=None, callback_step=None):
        self.agent = agent
        self.mdp = mdp
        self._n_mdp = self.mdp.num_envs #change it later
        self.callbacks_fit = callbacks_fit if callbacks_fit is not None else list()
        self.callback_step = callback_step if callback_step is not None else lambda x: None

        self._state = np.zeros((self._n_mdp, self.mdp.info.observation_space.shape[0]))
        
        self.eval = False
        
        self._total_episodes_counter = 0
        self._total_steps_counter = 0
        self._current_episodes_counter = 0
        self._current_steps_counter = 0
        self._episode_steps = 0 #[None for _ in range(self._n_mdp)]
        self._n_episodes = None
        self._n_steps_per_fit = None
        self._n_episodes_per_fit = None

        self.current_idx = 0

    def learn(self, n_steps=None, n_episodes=None, n_steps_per_fit=None,
            n_episodes_per_fit=None, render=False, quiet=False):
        """
        This function moves the agent in the environment and fits the policy
        using the collected samples. The agent can be moved for a given number
        of steps or a given number of episodes and, independently from this
        choice, the policy can be fitted after a given number of steps or a
        given number of episodes. By default, the environment is reset.

        Args:
            n_steps (int, None): number of steps to move the agent;
            n_episodes (int, None): number of episodes to move the agent;
            n_steps_per_fit (int, None): number of steps between each fit of the
                policy;
            n_episodes_per_fit (int, None): number of episodes between each fit
                of the policy;
            render (bool, False): whether to render the environment or not;
            quiet (bool, False): whether to show the progress bar or not.

        """
        assert (n_episodes_per_fit is not None and n_steps_per_fit is None)\
            or (n_episodes_per_fit is None and n_steps_per_fit is not None)

        self._n_steps_per_fit = n_steps_per_fit
        self._n_episodes_per_fit = n_episodes_per_fit

        if n_steps_per_fit is not None:
            fit_condition =\
                lambda: self._current_steps_counter >= self._n_steps_per_fit
        else:
            fit_condition = lambda: self._current_episodes_counter\
                                    >= self._n_episodes_per_fit

        self._run(n_steps, n_episodes, fit_condition, render, quiet, get_env_info=False)

    def evaluate(self, initial_states=None, n_steps=None, n_episodes=None,
                 render=False, quiet=False, get_env_info=False):
        """
        This function moves the agent in the environment using its policy.
        The agent is moved for a provided number of steps, episodes, or from
        a set of initial states for the whole episode. By default, the
        environment is reset.

        Args:
            initial_states (np.ndarray, None): the starting states of each
                episode;
            n_steps (int, None): number of steps to move the agent;
            n_episodes (int, None): number of episodes to move the agent;
            render (bool, False): whether to render the environment or not;
            quiet (bool, False): whether to show the progress bar or not;
            get_env_info (bool, False): whether to return the environment
                info list or not.

        Returns:
            The collected dataset and, optionally, an extra dataset of
            environment info, collected at each step.

        """
        fit_condition = lambda: False

        return self._run(n_steps, n_episodes, fit_condition, render, quiet, get_env_info,
                         initial_states)

    def _run(self, n_steps, n_episodes, fit_condition, render, quiet, get_env_info,
             initial_states=None):
        assert n_episodes is not None and n_steps is None and initial_states is None\
            or n_episodes is None and n_steps is not None and initial_states is None\
            or n_episodes is None and n_steps is None and initial_states is not None

        self._n_episodes = len(
            initial_states) if initial_states is not None else n_episodes

        if n_steps is not None:
            move_condition =\
                lambda: self._total_steps_counter < n_steps

            steps_progress_bar = tqdm(total=n_steps,
                                      dynamic_ncols=True, disable=quiet,
                                      leave=False)
            episodes_progress_bar = tqdm(disable=True)
        else:
            move_condition =\
                lambda: self._total_episodes_counter < self._n_episodes

            steps_progress_bar = tqdm(disable=True)
            episodes_progress_bar = tqdm(total=self._n_episodes,
                                         dynamic_ncols=True, disable=quiet,
                                         leave=False)

        dataset, dataset_info = self._run_impl(move_condition, fit_condition, steps_progress_bar,
                                                episodes_progress_bar, render, initial_states)

        if get_env_info:
            return dataset, dataset_info
        else:
            return dataset
    
    def _run_eval_impl(self, move_condition, fit_condition, steps_progress_bar,
                  episodes_progress_bar, render, initial_states):
        self._total_episodes_counter = 0
        self._total_steps_counter = 0
        self._current_episodes_counter = 0
        self._current_steps_counter = 0

        dataset = list()
        dataset_info = defaultdict(list)

        last = True
        while move_condition():
            if last:
                self.eval_reset(self.current_idx, initial_states)

            sample, step_info = self._eval_step(self.current_idx, render)

            self.callback_step([sample])

            self._total_steps_counter += 1
            self._current_steps_counter += 1
            steps_progress_bar.update(1)

            if sample[-1]:
                self._total_episodes_counter += 1
                self._current_episodes_counter += 1
                episodes_progress_bar.update(1)

            dataset.append(sample)

            for key, value in step_info.items():
                dataset_info[key].append(value)

            if fit_condition():
                self.agent.fit(dataset, **dataset_info)
                self._current_episodes_counter = 0
                self._current_steps_counter = 0

                for c in self.callbacks_fit:
                    c(dataset)

                dataset = list()
                dataset_info = defaultdict(list)

            last = sample[-1]

        self.agent.stop()
        self.mdp.env_method("stop", indices=self.current_idx)

        steps_progress_bar.close()
        episodes_progress_bar.close()

        return dataset, dataset_info
    
    def _run_impl(self, move_condition, fit_condition, steps_progress_bar,
                  episodes_progress_bar, render, initial_states):
        
        # eval
        if self.eval:
            # TODO: do it better
            return self._run_eval_impl(move_condition, fit_condition, steps_progress_bar,
                                     episodes_progress_bar, render, initial_states)
        
        # self._total_episodes_counter = 0
        self._total_steps_counter = 0
        # self._current_episodes_counter = 0
        self._current_steps_counter = 0

        dataset = list()
        dataset_info = defaultdict(list)

        last = np.array([True]*self._n_mdp)
        self._state = np.zeros((self._n_mdp, self.mdp.info.observation_space.shape[0]))

        while move_condition():
            idx = np.argwhere(last == True).flatten()
            if len(idx) > 0 and len(idx) < self._n_mdp:
                self.reset(initial_states, idx = idx)
                last[idx] = False
            elif len(idx) == self._n_mdp:
                self.reset(initial_states)
                last = [False]*self._n_mdp

            sample, last, step_info = self._step(render)
            
            last = np.array(last)

            for step_info_i in step_info:
                for key, value in step_info_i.items():
                    dataset_info[key].append(value)

            dataset+=sample

            self._total_steps_counter += 1
            self._current_steps_counter += 1
            steps_progress_bar.update(1)

            if fit_condition():
                self.agent.fit(dataset, **dataset_info)
                self._current_episodes_counter = 0
                self._current_steps_counter = 0

                for c in self.callbacks_fit:
                    c(dataset)

                dataset = list()
                dataset_info = defaultdict(list)

        self.agent.stop()
        self.mdp.stop()

        steps_progress_bar.close()
        episodes_progress_bar.close()

        return dataset, dataset_info
    
    def _step(self, render):
        idx = np.arange(self._n_mdp)
        action = self.agent.draw_action([idx, np.stack(self._state, axis=0)])
        next_state, reward, absorbing, step_info = self.mdp.step(action)

        self._episode_steps += 1

        if render:
            self.mdp.render()

        last = np.logical_not(np.logical_and(self._episode_steps < self.mdp.info.horizon,np.logical_not(absorbing)))

        state = self._state
        next_state = self._preprocess(next_state.copy())
        self._state = next_state

        sample = [([i, state[i]], action[i], reward[i], [i, next_state[i]], absorbing[i], last[i]) for i in range(self._n_mdp)]

        return sample, last, step_info

    def reset(self, initial_states=None, idx = None):

        if initial_states is None\
            or self._total_episodes_counter == self._n_episodes:
            initial_state = [None]*self._n_mdp if idx is None else None
        else:
            initial_state = initial_states[self._total_episodes_counter]

        self.agent.episode_start()

        if idx is None:
            self._state = self._preprocess(self.mdp.reset(initial_state).copy())
        else:
            state = np.stack(self.mdp.env_method("reset", initial_state, indices=idx), axis = 0)
            self._state[idx, :] = self._preprocess(state.copy())
        
        self.agent.next_action = None
        self._episode_steps = 0
    
    def _eval_step(self, i, render):
        action = self.agent.draw_action([i, self._state])
        next_state, reward, absorbing, step_info = self.mdp.env_method("step", action, indices = i)[0]

        self._episode_steps += 1

        if render:
            self.mdp.env_name("render")

        last = not(
            self._episode_steps < self.mdp.info.horizon and not absorbing)

        state = self._state
        next_state = self._preprocess(next_state.copy())
        self._state= next_state

        return ([i, state], action, reward, [i, next_state], absorbing, last), step_info

    def eval_reset(self, i, initial_states=None):

        if initial_states is None\
            or self._total_episodes_counter == self._n_episodes:
            initial_state = None
        else:
            initial_state = initial_states[self._total_episodes_counter]

        self.agent.episode_start()

        self._state = self._preprocess(self.mdp.env_method("reset", initial_state, indices = i)[0].copy())
        
        self.agent.next_action = None
        self._episode_steps = 0

    def _preprocess(self, state):
        """
        Method to apply state preprocessors.

        Args:
            state (np.ndarray): the state to be preprocessed.

        Returns:
             The preprocessed state.

        """
        for p in self.agent.preprocessors:
            state = p(state)

        return state
