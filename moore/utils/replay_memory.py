from mushroom_rl.utils.replay_memory import ReplayMemory


class ReplayMemory(ReplayMemory):
    def add(self, dataset):
        for i in range(len(dataset)):
            self._states[self._idx] = dataset[i][0][1]
            self._actions[self._idx] = dataset[i][1]
            self._rewards[self._idx] = dataset[i][2]
            self._next_states[self._idx] = dataset[i][3][1]
            self._absorbing[self._idx] = dataset[i][4]
            self._last[self._idx] = dataset[i][5]

            self._idx += 1
            if self._idx == self._max_size:
                self._full = True
                self._idx = 0


